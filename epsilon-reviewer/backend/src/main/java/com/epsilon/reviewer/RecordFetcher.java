package com.epsilon.reviewer;

import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;

/** 天鳳の牌譜URLとアップロードを扱い、入力サイズと転送先を制限します。 */
public final class RecordFetcher implements AutoCloseable {
  public static final int MAX_UPLOAD = 16 * 1024 * 1024;
  private final HttpClient client =
      HttpClient.newBuilder()
          .connectTimeout(Duration.ofSeconds(10))
          .followRedirects(HttpClient.Redirect.NEVER)
          .build();

  public SourceRecord fetch(String url) throws IOException, InterruptedException {
    URI input;
    try {
      input = URI.create(url);
    } catch (IllegalArgumentException e) {
      throw new ApiException(400, "invalid_url", "Invalid record URL.");
    }
    if (input.getHost() == null
        || input.getUserInfo() != null
        || input.getPort() != -1
        || !("https".equals(input.getScheme()) || "http".equals(input.getScheme()))) {
      throw new ApiException(400, "invalid_url", "Expected a Tenhou record URL.");
    }
    String host = input.getHost().toLowerCase(Locale.ROOT);
    if (!host.equals("tenhou.net") && !host.equals("www.tenhou.net"))
      throw new ApiException(400, "unsupported_url", "Only Tenhou record URLs are supported.");
    Map<String, String> query = query(input.getRawQuery());
    String id = query.get("log");
    if (id == null && "/0/log/".equals(input.getPath())) id = input.getRawQuery();
    if (id == null || !id.matches("[0-9]{10}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{8}"))
      throw new ApiException(400, "invalid_record_id", "Invalid Tenhou log ID.");
    return new SourceRecord(
        download(URI.create("https://tenhou.net/0/log/?" + id)), id + ".xml", "tenhou");
  }

  private byte[] download(URI uri) throws IOException, InterruptedException {
    for (int redirect = 0; redirect <= 3; redirect++) {
      requirePublicRecordEndpoint(uri);
      var request =
          HttpRequest.newBuilder(uri)
              .timeout(Duration.ofSeconds(30))
              .header("User-Agent", "EpsilonReviewer/1.0")
              .GET()
              .build();
      var response = BoundedHttp.get(client, request, MAX_UPLOAD);
      {
        if (response.statusCode() >= 300 && response.statusCode() < 400) {
          URI next =
              uri.resolve(
                  response
                      .headers()
                      .firstValue("Location")
                      .orElseThrow(
                          () ->
                              new ApiException(
                                  502,
                                  "fetch_failed",
                                  "Record response is missing a redirect destination.")));
          if (!Objects.equals(uri.getHost(), next.getHost()) || uri.getPort() != next.getPort())
            throw new ApiException(
                502, "unsafe_redirect", "Redirect outside the record service was rejected.");
          uri = next;
          continue;
        }
        if (response.statusCode() == 403 || response.statusCode() == 401)
          throw new ApiException(
              422, "record_access_denied", "Record service denied access to this record.");
        if (response.statusCode() == 404)
          throw new ApiException(
              404, "record_not_found", "Record was not found by the upstream service.");
        if (response.statusCode() != 200)
          throw new ApiException(502, "fetch_failed", "Record service request failed.");
        return response.body();
      }
    }
    throw new ApiException(502, "too_many_redirects", "Record service redirect limit exceeded.");
  }

  private static void requirePublicRecordEndpoint(URI uri) throws UnknownHostException {
    boolean allowed =
        "https".equals(uri.getScheme())
            && uri.getUserInfo() == null
            && "tenhou.net".equals(uri.getHost())
            && uri.getPort() == -1;
    if (!allowed)
      throw new ApiException(400, "invalid_url", "Record fetch destination is not allowed.");
    for (InetAddress address : InetAddress.getAllByName(uri.getHost())) {
      if (address.isAnyLocalAddress()
          || address.isLoopbackAddress()
          || address.isLinkLocalAddress()
          || address.isSiteLocalAddress()
          || address.isMulticastAddress())
        throw new ApiException(
            502, "invalid_address", "Record service resolved to a disallowed address.");
    }
  }

  public static SourceRecord uploaded(byte[] bytes, String name) {
    String fileName = name == null ? "record.mjson" : name;
    if (fileName.length() > 255
        || fileName.contains("/")
        || fileName.contains("\\")
        || fileName.chars().anyMatch(c -> c < 32))
      throw new ApiException(400, "invalid_file_name", "Invalid record file name.");
    String formatName = fileName.toLowerCase(Locale.ROOT);
    if (formatName.endsWith(".gz")) formatName = formatName.substring(0, formatName.length() - 3);
    if (formatName.endsWith(".epsilon-review.json")
        || formatName.endsWith(".epsilon-reviewer.json"))
      throw new ApiException(
          415,
          "result_import_disabled",
          "Analysis result import is disabled; provide the original record.");
    if (bytes.length > MAX_UPLOAD)
      throw new ApiException(413, "record_too_large", "Record file exceeds 16 MiB.");
    String prefix =
        new String(bytes, 0, Math.min(bytes.length, 1024), StandardCharsets.UTF_8).stripLeading();
    if (prefix.startsWith("{")
        && (prefix.contains("\"formatVersion\"") || prefix.contains("\"resultId\"")))
      throw new ApiException(
          415,
          "result_import_disabled",
          "Analysis result import is disabled; provide the original record.");
    return new SourceRecord(bytes, fileName, "auto");
  }

  public static byte[] readBounded(InputStream input, int maximum) throws IOException {
    var out = new ByteArrayOutputStream();
    byte[] buffer = new byte[8192];
    for (int read; (read = input.read(buffer)) != -1; ) {
      if (out.size() > maximum - read)
        throw new ApiException(413, "record_too_large", "Record data exceeds the size limit.");
      out.write(buffer, 0, read);
    }
    return out.toByteArray();
  }

  private static Map<String, String> query(String raw) {
    Map<String, String> result = new HashMap<>();
    if (raw != null)
      for (String part : raw.split("&")) {
        String[] pair = part.split("=", 2);
        if (pair.length == 2)
          result.put(
              URLDecoder.decode(pair[0], StandardCharsets.UTF_8),
              URLDecoder.decode(pair[1], StandardCharsets.UTF_8));
      }
    return result;
  }

  @Override
  public void close() {
    client.close();
  }

  public record SourceRecord(byte[] bytes, String fileName, String source) {}
}

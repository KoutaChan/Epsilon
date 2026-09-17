package com.epsilon.reviewer;

import com.google.gson.*;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.*;
import java.util.*;
import java.util.function.IntSupplier;
import java.util.zip.GZIPInputStream;

/** HTTPの入出力と、Host・Origin・Cookieの検証を行う。 */
final class ReviewerHttp {
  private static final Gson JSON =
      new GsonBuilder().disableHtmlEscaping().serializeNulls().create();
  private static final String COOKIE = "epsilon_reviewer_owner";
  private final URI publicOrigin;
  private final Path webDirectory;
  private final IntSupplier port;
  private final SecureRandom random = new SecureRandom();

  ReviewerHttp(URI publicOrigin, Path webDirectory, IntSupplier port) {
    this.publicOrigin = publicOrigin;
    this.webDirectory = webDirectory == null ? null : webDirectory.toAbsolutePath().normalize();
    this.port = port;
  }

  void prepare(HttpExchange exchange) {
    var headers = exchange.getResponseHeaders();
    headers.set("X-Content-Type-Options", "nosniff");
    headers.set("Referrer-Policy", "no-referrer");
    headers.set("Cache-Control", "no-store");
    headers.set(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'"
            + " data: blob:; font-src 'self'; connect-src 'self'; worker-src 'self' blob:;"
            + " base-uri 'none'; frame-ancestors 'none'; form-action 'self'");
    boolean sharedPage = hasSharedReviewQuery(exchange.getRequestURI());
    if (sharedPage || isSharedResultApi(exchange.getRequestURI().getPath()))
      headers.set("X-Robots-Tag", "noindex");
    boolean sharedNavigation =
        sharedPage
            && exchange.getRequestMethod().equals("GET")
            && "navigate".equals(exchange.getRequestHeaders().getFirst("Sec-Fetch-Mode"))
            && "document".equals(exchange.getRequestHeaders().getFirst("Sec-Fetch-Dest"));
    requireSameOriginRequest(exchange, sharedNavigation);
  }

  private void requireSameOriginRequest(HttpExchange exchange, boolean sharedNavigation) {
    String authority = exchange.getRequestHeaders().getFirst("Host");
    String expectedAuthority = publicAuthority();
    boolean local = exchange.getRemoteAddress().getAddress().isLoopbackAddress();
    boolean authorityAllowed = Objects.equals(authority, expectedAuthority);
    if (local)
      authorityAllowed |=
          Objects.equals(authority, "localhost:" + port.getAsInt())
              || Objects.equals(authority, "127.0.0.1:" + port.getAsInt())
              || Objects.equals(authority, "[::1]:" + port.getAsInt());
    if (!authorityAllowed)
      throw new ApiException(
          403, "host_not_allowed", "Request host does not match the configured public origin.");
    String origin = exchange.getRequestHeaders().getFirst("Origin");
    if (origin != null && !origin.equals(publicOrigin.getScheme() + "://" + authority))
      throw new ApiException(403, "origin_not_allowed", "Cross-origin requests are not allowed.");
    // 外部ページからの共有リンクの移動だけを許可し、Host と Origin の境界は維持する。
    if (sharedNavigation) return;
    String site = exchange.getRequestHeaders().getFirst("Sec-Fetch-Site");
    if (site != null && !Set.of("same-origin", "none").contains(site))
      throw new ApiException(403, "origin_not_allowed", "Cross-origin requests are not allowed.");
  }

  String sharedResultUrl(String id) {
    return publicOrigin.getScheme() + "://" + publicAuthority() + "/?share=" + id;
  }

  private String publicAuthority() {
    String authority = publicOrigin.getRawAuthority();
    return authority.endsWith(":0")
        ? authority.substring(0, authority.length() - 1) + port.getAsInt()
        : authority;
  }

  static boolean isSharedResultApi(String path) {
    return path.equals("/api/shares") || path.startsWith("/api/shares/");
  }

  private static boolean hasSharedReviewQuery(URI uri) {
    if (!uri.getPath().equals("/") || uri.getRawQuery() == null) return false;
    for (String parameter : uri.getRawQuery().split("&")) {
      String key = parameter.split("=", 2)[0];
      if (URLDecoder.decode(key, StandardCharsets.UTF_8).equals("share")) return true;
    }
    return false;
  }

  String owner(HttpExchange exchange) {
    String key = null;
    for (String cookie : exchange.getRequestHeaders().getOrDefault("Cookie", List.of())) {
      for (String item : cookie.split(";")) {
        String[] pair = item.trim().split("=", 2);
        if (pair.length == 2 && pair[0].equals(COOKIE) && pair[1].matches("[A-Za-z0-9_-]{43}"))
          key = pair[1];
      }
    }
    if (key == null) {
      byte[] bytes = new byte[32];
      random.nextBytes(bytes);
      key = Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
      exchange
          .getResponseHeaders()
          .add(
              "Set-Cookie",
              COOKIE
                  + "="
                  + key
                  + "; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000"
                  + ("https".equals(publicOrigin.getScheme()) ? "; Secure" : ""));
    }
    try {
      return HexFormat.of()
          .formatHex(
              MessageDigest.getInstance("SHA-256").digest(key.getBytes(StandardCharsets.US_ASCII)));
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError(e);
    }
  }

  void serveUi(HttpExchange exchange, String path) throws IOException {
    if (!exchange.getRequestMethod().equals("GET") || webDirectory == null)
      throw new ApiException(404, "not_found", "Web UI directory is not configured.");
    Path file = webDirectory.resolve(path.substring(1)).normalize();
    if (!file.startsWith(webDirectory)) throw new ApiException(404, "not_found", "File not found.");
    if (!Files.isRegularFile(file)) {
      if (path.contains(".")) throw new ApiException(404, "not_found", "File not found.");
      file = webDirectory.resolve("index.html");
    }
    if (!Files.isRegularFile(file) || !file.toRealPath().startsWith(webDirectory.toRealPath()))
      throw new ApiException(404, "not_found", "Web UI build is unavailable.");
    String name = file.getFileName().toString();
    String type =
        name.endsWith(".js")
            ? "text/javascript; charset=utf-8"
            : name.endsWith(".css")
                ? "text/css; charset=utf-8"
                : name.endsWith(".svg")
                    ? "image/svg+xml"
                    : name.endsWith(".html")
                        ? "text/html; charset=utf-8"
                        : name.endsWith(".woff2")
                            ? "font/woff2"
                            : name.endsWith(".png") ? "image/png" : "application/octet-stream";
    bytes(exchange, 200, type, Files.readAllBytes(file));
  }

  static JsonObject requestJson(HttpExchange exchange) throws IOException {
    String contentType = exchange.getRequestHeaders().getFirst("Content-Type");
    if (contentType == null || !contentType.toLowerCase(Locale.ROOT).startsWith("application/json"))
      throw new ApiException(
          415, "unsupported_content_type", "Expected application/json content type.");
    byte[] bytes = RecordFetcher.readBounded(exchange.getRequestBody(), 16 * 1024);
    try {
      return JsonParser.parseString(new String(bytes, StandardCharsets.UTF_8)).getAsJsonObject();
    } catch (RuntimeException e) {
      throw new ApiException(400, "invalid_json", "Invalid JSON request body.");
    }
  }

  static String readRequiredString(JsonObject body, String key) {
    if (!body.has(key)
        || !body.get(key).isJsonPrimitive()
        || !body.getAsJsonPrimitive(key).isString()
        || body.get(key).getAsString().isBlank())
      throw new ApiException(400, "missing_field", key + " is required.");
    return body.get(key).getAsString();
  }

  static void json(HttpExchange exchange, int status, Object value) throws IOException {
    bytes(
        exchange,
        status,
        "application/json; charset=utf-8",
        JSON.toJson(value).getBytes(StandardCharsets.UTF_8));
  }

  static void error(HttpExchange exchange, int status, String code, String message)
      throws IOException {
    json(exchange, status, Map.of("error", new ApiError(code, message)));
  }

  /** 保存済み gzip をそのまま送り、非対応クライアントだけ逐次展開する。 */
  static void streamResult(
      HttpExchange exchange, ResultStore.ResultContent content, String attachmentName)
      throws IOException {
    var headers = exchange.getResponseHeaders();
    boolean attachment = attachmentName != null;
    boolean compressed = attachment || acceptsGzip(exchange);
    headers.set(
        "Content-Type", attachment ? "application/gzip" : "application/json; charset=utf-8");
    if (attachment)
      headers.set("Content-Disposition", "attachment; filename=\"" + attachmentName + "\"");
    else {
      headers.set("Vary", "Accept-Encoding");
      if (compressed) headers.set("Content-Encoding", "gzip");
    }
    if (compressed) {
      exchange.sendResponseHeaders(200, content.size());
      content.input().transferTo(exchange.getResponseBody());
    } else {
      try (var json = new GZIPInputStream(content.input())) {
        exchange.sendResponseHeaders(200, 0);
        json.transferTo(exchange.getResponseBody());
      }
    }
  }

  private static boolean acceptsGzip(HttpExchange exchange) {
    Double gzip = null, wildcard = null;
    for (String header : exchange.getRequestHeaders().getOrDefault("Accept-Encoding", List.of())) {
      for (String entry : header.split(",")) {
        String[] parameters = entry.trim().split(";");
        String coding = parameters[0].trim();
        if (!coding.equalsIgnoreCase("gzip") && !coding.equals("*")) continue;
        double quality = 1;
        for (int i = 1; i < parameters.length; i++) {
          String parameter = parameters[i].trim();
          if (parameter.startsWith("q=")) {
            try {
              quality = Double.parseDouble(parameter.substring(2));
            } catch (NumberFormatException failure) {
              quality = 0;
            }
          }
        }
        if (coding.equalsIgnoreCase("gzip")) gzip = quality;
        else wildcard = quality;
      }
    }
    Double quality = gzip != null ? gzip : wildcard;
    return quality != null && quality > 0 && quality <= 1;
  }

  static void bytes(HttpExchange exchange, int status, String type, byte[] value)
      throws IOException {
    exchange.getResponseHeaders().set("Content-Type", type);
    exchange.sendResponseHeaders(status, value.length);
    exchange.getResponseBody().write(value);
  }
}

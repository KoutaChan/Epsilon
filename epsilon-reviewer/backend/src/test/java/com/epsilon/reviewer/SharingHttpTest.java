package com.epsilon.reviewer;

import com.epsilon.reviewer.dto.ModelInfo;
import com.epsilon.reviewer.dto.RecordMetadata;
import com.epsilon.reviewer.dto.ReviewResult;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 共有リンクの永続性、所有者権限との分離、外部サイトからのアクセス制限、圧縮転送と非公開結果の扱いを検証する。 */
public final class SharingHttpTest {
  @Test
  public void webResultHasAnAutomaticStableLinkAcrossRestartAndDeletionInvalidatesIt()
      throws Exception {
    Path data = Files.createTempDirectory("reviewer-sharing-lifecycle-");
    String id = UUID.randomUUID().toString();
    Owner owner = ownerFromCookieKey("A".repeat(43));
    try (var client = HttpClient.newHttpClient()) {
      saveResult(data, owner, id, false);
      try (var server = startServer(data, null)) {
        URI base = createServerBaseUri(server);
        String expected = base + "/?share=" + id;
        for (int attempt = 0; attempt < 2; attempt++) {
          var response =
              sendRequest(client, createRequest(base, "/api/results/" + id + "/share", owner));
          Assert.assertEquals(response.statusCode(), 200);
          Assert.assertEquals(readJsonBody(response).get("url").getAsString(), expected);
        }
        var shared = sendRequest(client, createRequest(base, "/api/shares/" + id, null));
        Assert.assertEquals(shared.statusCode(), 200);
        Assert.assertEquals(readJsonBody(shared).get("resultId").getAsString(), id);
        Assert.assertFalse(shared.headers().firstValue("Set-Cookie").isPresent());
      }
      try (var restarted = startServer(data, null)) {
        URI base = createServerBaseUri(restarted);
        var link = sendRequest(client, createRequest(base, "/api/results/" + id + "/share", owner));
        Assert.assertEquals(link.statusCode(), 200);
        Assert.assertEquals(readJsonBody(link).get("url").getAsString(), base + "/?share=" + id);
        Assert.assertEquals(
            sendRequest(client, createRequest(base, "/api/shares/" + id, null)).statusCode(), 200);
        Assert.assertEquals(
            sendRequest(client, createRequest(base, "/api/results/" + id, owner).DELETE())
                .statusCode(),
            204);
        assertError(
            sendRequest(client, createRequest(base, "/api/shares/" + id, null)),
            404,
            "share_not_found");
        assertError(
            sendRequest(client, createRequest(base, "/api/results/" + id + "/share", owner)),
            404,
            "result_not_found");
      }
    } finally {
      removeTestDirectory(data);
    }
  }

  @Test
  public void publicResultIsCookieIndependentAndGrantsNoOwnerPermissions() throws Exception {
    Path data = Files.createTempDirectory("reviewer-sharing-permissions-");
    Owner owner = ownerFromCookieKey("A".repeat(43));
    Owner visitor = ownerFromCookieKey("B".repeat(42) + "A");
    String id = UUID.randomUUID().toString();
    saveResult(data, owner, id, false);
    try (var server = startServer(data, null);
        var client = HttpClient.newHttpClient()) {
      URI base = createServerBaseUri(server);
      var original = sendRequest(client, createRequest(base, "/api/results/" + id, owner));
      Assert.assertEquals(original.statusCode(), 200);
      for (Owner viewer : new Owner[] {null, visitor}) {
        var shared = sendRequest(client, createRequest(base, "/api/shares/" + id, viewer));
        Assert.assertEquals(shared.statusCode(), 200);
        Assert.assertEquals(shared.body(), original.body());
        Assert.assertEquals(shared.headers().firstValue("X-Robots-Tag").orElseThrow(), "noindex");
        Assert.assertFalse(shared.headers().firstValue("Set-Cookie").isPresent());
        for (String suffix : List.of("", "/export", "/source", "/share")) {
          assertError(
              sendRequest(client, createRequest(base, "/api/results/" + id + suffix, viewer)),
              404,
              "result_not_found");
        }
        assertError(
            sendRequest(client, createRequest(base, "/api/results/" + id, viewer).DELETE()),
            404,
            "result_not_found");
      }
      var history = sendRequest(client, createRequest(base, "/api/history", visitor));
      Assert.assertEquals(history.statusCode(), 200);
      Assert.assertTrue(readJsonBody(history).getAsJsonArray("results").isEmpty());
      assertError(
          sendRequest(
              client,
              createRequest(base, "/api/records", visitor)
                  .header("Content-Type", "application/json")
                  .POST(HttpRequest.BodyPublishers.ofString("{\"resultId\":\"" + id + "\"}"))),
          404,
          "result_not_found");
      for (String path : List.of("/api/shares/" + id, "/api/results/" + id + "/share")) {
        for (String method : List.of("POST", "DELETE")) {
          Assert.assertEquals(
              sendRequest(
                      client,
                      createRequest(base, path, owner)
                          .method(method, HttpRequest.BodyPublishers.noBody()))
                  .statusCode(),
              404);
        }
      }
      Assert.assertEquals(
          sendRequest(client, createRequest(base, "/api/shares/" + id, null)).statusCode(), 200);
      for (String missing : List.of("invalid", UUID.randomUUID().toString(), id + "/source")) {
        var rejected = sendRequest(client, createRequest(base, "/api/shares/" + missing, null));
        assertError(rejected, 404, "share_not_found");
        Assert.assertFalse(rejected.headers().firstValue("Set-Cookie").isPresent());
      }
    } finally {
      removeTestDirectory(data);
    }
  }

  @Test
  public void crossSiteShareNavigationDoesNotRelaxApiOriginOrHostBoundaries() throws Exception {
    Path data = Files.createTempDirectory("reviewer-sharing-navigation-");
    Path web = Files.createDirectory(data.resolve("web"));
    Files.writeString(web.resolve("index.html"), "<main>Shared review fixture</main>");
    Owner owner = ownerFromCookieKey("A".repeat(43));
    String id = UUID.randomUUID().toString();
    saveResult(data, owner, id, false);
    try (var server = startServer(data, web);
        var client = HttpClient.newHttpClient()) {
      URI base = createServerBaseUri(server);
      for (String query :
          List.of(
              "share=" + id,
              "%73hare=" + id,
              "share=invalid",
              "share=",
              "share=" + id + "&share=" + id,
              "share=" + id + "&other=1")) {
        var page =
            sendRequest(
                client,
                createRequest(base, "/?" + query, null)
                    .header("Sec-Fetch-Site", "cross-site")
                    .header("Sec-Fetch-Mode", "navigate")
                    .header("Sec-Fetch-Dest", "document"));
        Assert.assertEquals(page.statusCode(), 200);
        Assert.assertEquals(page.body(), "<main>Shared review fixture</main>");
        Assert.assertEquals(page.headers().firstValue("X-Robots-Tag").orElseThrow(), "noindex");
        Assert.assertFalse(page.headers().firstValue("Set-Cookie").isPresent());
      }
      for (String path : List.of("/", "/api/shares/" + id, "/api/results/" + id + "/share")) {
        assertError(
            sendRequest(
                client,
                createRequest(base, path, owner)
                    .header("Sec-Fetch-Site", "cross-site")
                    .header("Sec-Fetch-Mode", "navigate")
                    .header("Sec-Fetch-Dest", "document")),
            403,
            "origin_not_allowed");
      }
      for (String[] context : new String[][] {{"cors", "document"}, {"navigate", "iframe"}}) {
        assertError(
            sendRequest(
                client,
                createRequest(base, "/?share=" + id, null)
                    .header("Sec-Fetch-Site", "cross-site")
                    .header("Sec-Fetch-Mode", context[0])
                    .header("Sec-Fetch-Dest", context[1])),
            403,
            "origin_not_allowed");
      }
      assertError(
          sendRequest(
              client,
              createRequest(base, "/api/shares/" + id, null)
                  .header("Origin", "https://example.com")),
          403,
          "origin_not_allowed");
      assertError(
          sendRequest(
              client,
              createRequest(base, "/?share=" + id, null)
                  .header("Sec-Fetch-Site", "cross-site")
                  .header("Sec-Fetch-Mode", "navigate")
                  .header("Sec-Fetch-Dest", "document")
                  .header("Origin", "https://example.com")),
          403,
          "origin_not_allowed");
      assertError(
          sendRequest(
              client,
              createRequest(base, "/api/results/" + id, owner)
                  .header("Sec-Fetch-Site", "cross-site")
                  .DELETE()),
          403,
          "origin_not_allowed");
      assertError(
          sendRequest(
              client,
              createRequest(base, "/?share=" + id, null)
                  .header("Sec-Fetch-Site", "cross-site")
                  .header("Sec-Fetch-Mode", "navigate")
                  .header("Sec-Fetch-Dest", "document")
                  .POST(HttpRequest.BodyPublishers.noBody())),
          403,
          "origin_not_allowed");
      try (var socket = new Socket("127.0.0.1", server.port())) {
        socket.setSoTimeout(5000);
        socket
            .getOutputStream()
            .write(
                ("GET /?share="
                        + id
                        + " HTTP/1.1\r\nHost: attacker.example\r\n"
                        + "Sec-Fetch-Site: cross-site\r\nSec-Fetch-Mode: navigate\r\n"
                        + "Sec-Fetch-Dest: document\r\nConnection: close\r\n\r\n")
                    .getBytes(StandardCharsets.US_ASCII));
        String rejected =
            new String(socket.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        Assert.assertTrue(rejected.startsWith("HTTP/1.1 403"));
        Assert.assertTrue(rejected.contains("host_not_allowed"));
      }
      Assert.assertEquals(
          sendRequest(client, createRequest(base, "/api/results/" + id, owner)).statusCode(), 200);
    } finally {
      removeTestDirectory(data);
    }
  }

  @Test
  public void gzipNegotiationStreamsTheSameResultAndExportsRemainRawCompressedFiles()
      throws Exception {
    Path data = Files.createTempDirectory("reviewer-sharing-gzip-");
    Owner owner = ownerFromCookieKey("A".repeat(43));
    String id = UUID.randomUUID().toString();
    saveResult(data, owner, id, false);
    Path resultFile =
        data.resolve("results").resolve(owner.hash()).resolve(id).resolve("result.json.gz");
    byte[] stored = Files.readAllBytes(resultFile);
    byte[] expected;
    try (var gzip = new java.util.zip.GZIPInputStream(new java.io.ByteArrayInputStream(stored))) {
      expected = gzip.readAllBytes();
    }
    try (var server = startServer(data, null);
        var client = HttpClient.newHttpClient()) {
      URI base = createServerBaseUri(server);
      for (String path : List.of("/api/results/" + id, "/api/shares/" + id)) {
        Owner viewer = path.startsWith("/api/shares/") ? null : owner;
        for (String encoding : List.of("gzip", "br, gzip;q=0.5", "*;q=1", "GZIP")) {
          var response =
              client.send(
                  createRequest(base, path, viewer).header("Accept-Encoding", encoding).build(),
                  HttpResponse.BodyHandlers.ofByteArray());
          Assert.assertEquals(response.statusCode(), 200);
          Assert.assertEquals(response.body(), stored);
          Assert.assertEquals(
              response.headers().firstValue("Content-Encoding").orElseThrow(), "gzip");
          Assert.assertEquals(
              response.headers().firstValue("Vary").orElseThrow(), "Accept-Encoding");
          Assert.assertEquals(
              response.headers().firstValueAsLong("Content-Length").orElseThrow(), stored.length);
          Assert.assertFalse(response.headers().firstValue("Set-Cookie").isPresent());
        }
        for (String encoding : List.of("identity", "br", "gzip;q=0, *;q=1", "gzip;q=0.000")) {
          var response =
              client.send(
                  createRequest(base, path, viewer).header("Accept-Encoding", encoding).build(),
                  HttpResponse.BodyHandlers.ofByteArray());
          Assert.assertEquals(response.statusCode(), 200);
          Assert.assertEquals(response.body(), expected);
          Assert.assertFalse(response.headers().firstValue("Content-Encoding").isPresent());
          Assert.assertEquals(
              response.headers().firstValue("Vary").orElseThrow(), "Accept-Encoding");
          Assert.assertFalse(response.headers().firstValue("Content-Length").isPresent());
          Assert.assertFalse(response.headers().firstValue("Set-Cookie").isPresent());
          Assert.assertEquals(
              JsonParser.parseString(new String(response.body(), StandardCharsets.UTF_8))
                  .getAsJsonObject()
                  .get("formatVersion")
                  .getAsInt(),
              ReviewResult.CURRENT_FORMAT_VERSION);
        }
      }
      for (String encoding : List.of("identity", "gzip")) {
        var exported =
            client.send(
                createRequest(base, "/api/results/" + id + "/export", owner)
                    .header("Accept-Encoding", encoding)
                    .build(),
                HttpResponse.BodyHandlers.ofByteArray());
        Assert.assertEquals(exported.statusCode(), 200);
        Assert.assertEquals(exported.body(), stored);
        Assert.assertEquals(
            exported.headers().firstValue("Content-Type").orElseThrow(), "application/gzip");
        Assert.assertFalse(exported.headers().firstValue("Content-Encoding").isPresent());
        Assert.assertEquals(
            exported.headers().firstValue("Content-Disposition").orElseThrow(),
            "attachment; filename=\"" + id + ".epsilon-reviewer.json.gz\"");
      }
    } finally {
      removeTestDirectory(data);
    }
  }

  @Test
  public void desktopResultsRemainPrivateAndHaveNoSharingLink() throws Exception {
    Path data = Files.createTempDirectory("reviewer-sharing-desktop-");
    Owner owner = ownerFromCookieKey("A".repeat(43));
    String id = UUID.randomUUID().toString();
    saveResult(data, owner, id, true);
    try (var server = startServer(data, null);
        var client = HttpClient.newHttpClient()) {
      URI base = createServerBaseUri(server);
      Assert.assertEquals(
          sendRequest(client, createRequest(base, "/api/results/" + id, owner)).statusCode(), 200);
      assertError(
          sendRequest(client, createRequest(base, "/api/results/" + id + "/share", owner)),
          409,
          "share_unavailable");
      var rejected = sendRequest(client, createRequest(base, "/api/shares/" + id, null));
      assertError(rejected, 404, "share_not_found");
      Assert.assertFalse(rejected.headers().firstValue("Set-Cookie").isPresent());
    } finally {
      removeTestDirectory(data);
    }
  }

  private static ReviewerServer startServer(Path data, Path web) throws Exception {
    var server =
        new ReviewerServer(
            new InetSocketAddress("127.0.0.1", 0),
            URI.create("http://127.0.0.1:0"),
            null,
            data,
            web);
    server.start();
    return server;
  }

  private static URI createServerBaseUri(ReviewerServer server) {
    return URI.create("http://127.0.0.1:" + server.port());
  }

  private static HttpRequest.Builder createRequest(URI base, String path, Owner owner) {
    var request = HttpRequest.newBuilder(base.resolve(path));
    if (owner != null) request.header("Cookie", owner.cookie());
    return request;
  }

  private static HttpResponse<String> sendRequest(HttpClient client, HttpRequest.Builder request)
      throws Exception {
    return client.send(request.build(), HttpResponse.BodyHandlers.ofString());
  }

  private static JsonObject readJsonBody(HttpResponse<String> response) {
    return JsonParser.parseString(response.body()).getAsJsonObject();
  }

  private static void assertError(HttpResponse<String> response, int status, String code) {
    Assert.assertEquals(response.statusCode(), status, response.body());
    Assert.assertEquals(
        readJsonBody(response).getAsJsonObject("error").get("code").getAsString(), code);
  }

  private static Owner ownerFromCookieKey(String key) throws Exception {
    String cookie = "epsilon_reviewer_owner=" + key;
    String hash =
        HexFormat.of()
            .formatHex(
                MessageDigest.getInstance("SHA-256")
                    .digest(key.getBytes(StandardCharsets.US_ASCII)));
    return new Owner(cookie, hash);
  }

  private static void saveResult(Path data, Owner owner, String id, boolean desktop)
      throws Exception {
    var result =
        new ReviewResult(
            ReviewResult.CURRENT_FORMAT_VERSION,
            id,
            "2026-09-15T00:00:00Z",
            new RecordMetadata(
                "mjai", "fixture.mjai", List.of("A", "B", "C", "D"), null, "unavailable", 1),
            new ModelInfo("epsilon", "fixed", "Epsilon", "1", "epsilon"),
            List.of());
    new ResultStore(data.resolve("results"))
        .save(
            owner.hash(),
            result,
            new RecordFetcher.SourceRecord(new byte[] {1, 2}, "fixture.mjai", "mjai"),
            desktop);
  }

  private static void removeTestDirectory(Path directory) throws Exception {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }

  private record Owner(String cookie, String hash) {}
}

package com.epsilon.reviewer;

import com.epsilon.reviewer.dto.*;
import com.google.gson.*;
import java.io.ByteArrayOutputStream;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Tests cookie ownership, external request rejection, and compressed invalid uploads. */
public final class ReviewerServerTest {
  @Test
  public void localRequestsPreserveOwnershipAndPublicShareOrigin() throws Exception {
    Path data = Files.createTempDirectory("reviewer-http-test-");
    try (var server =
            new ReviewerServer(
                new InetSocketAddress("127.0.0.1", 0),
                URI.create("http://reviewer.example:8080"),
                null,
                data,
                null);
        var client = HttpClient.newHttpClient()) {
      server.start();
      URI base = URI.create("http://127.0.0.1:" + server.port());
      var initial =
          client.send(
              HttpRequest.newBuilder(base.resolve("/api/models")).build(),
              HttpResponse.BodyHandlers.ofString());
      Assert.assertEquals(initial.statusCode(), 200);
      Assert.assertEquals(
          JsonParser.parseString(initial.body()).getAsJsonObject().getAsJsonArray("models").size(),
          0);
      String setCookie = initial.headers().firstValue("Set-Cookie").orElseThrow();
      Assert.assertTrue(setCookie.contains("HttpOnly"));
      Assert.assertTrue(setCookie.contains("SameSite=Lax"));
      Assert.assertFalse(setCookie.contains("Domain="));
      Assert.assertFalse(setCookie.contains("Secure"));
      String cookie = setCookie.split(";", 2)[0], key = cookie.split("=", 2)[1];
      String owner =
          HexFormat.of()
              .formatHex(
                  MessageDigest.getInstance("SHA-256")
                      .digest(key.getBytes(StandardCharsets.US_ASCII)));
      String id = UUID.randomUUID().toString();
      var stored =
          new ReviewResult(
              ReviewResult.CURRENT_FORMAT_VERSION,
              id,
              "2026-09-15T00:00:00Z",
              new RecordMetadata(
                  "tenhou", "game.xml", List.of("A", "B", "C", "D"), null, "unavailable", 1),
              new ModelInfo("epsilon", "fixed", "Epsilon", "1", "epsilon"),
              List.of());
      String source =
          """
          <mjloggm><GO type="169"/><UN n0="A" n1="B" n2="C" n3="D"/>
          <INIT seed="0,0,0,0,0,132" ten="250,250,250,250" oya="0"
          hai0="0,1,2,3,4,5,6,7,8,9,10,11,12"
          hai1="13,14,15,16,17,18,19,20,21,22,23,24,25"
          hai2="26,27,28,29,30,31,32,33,34,35,36,37,38"
          hai3="39,40,41,42,43,44,45,46,47,48,49,50,51"/>
          <T128/><D128/><RYUUKYOKU sc="250,0,250,0,250,0,250,0"
          owari="250,0,250,0,250,0,250,0"/></mjloggm>
          """;
      new ResultStore(data.resolve("results"))
          .save(
              owner,
              stored,
              new RecordFetcher.SourceRecord(
                  source.getBytes(StandardCharsets.UTF_8), "game.xml", "tenhou"),
              false);
      for (String host : List.of("127.0.0.1", "localhost")) {
        URI local = URI.create("http://" + host + ":" + server.port());
        var models =
            client.send(
                HttpRequest.newBuilder(local.resolve("/api/models")).build(),
                HttpResponse.BodyHandlers.ofString());
        Assert.assertEquals(models.statusCode(), 200);
        var prepared =
            client.send(
                HttpRequest.newBuilder(local.resolve("/api/records"))
                    .header("Origin", local.toString())
                    .header("Sec-Fetch-Site", "same-origin")
                    .header("Content-Type", "application/json")
                    .header("Cookie", cookie)
                    .POST(HttpRequest.BodyPublishers.ofString("{\"resultId\":\"" + id + "\"}"))
                    .build(),
                HttpResponse.BodyHandlers.ofString());
        Assert.assertEquals(prepared.statusCode(), 201, prepared.body());
        Assert.assertEquals(
            JsonParser.parseString(prepared.body())
                .getAsJsonObject()
                .getAsJsonObject("metadata")
                .get("roundCount")
                .getAsInt(),
            1);
        var shared =
            client.send(
                HttpRequest.newBuilder(local.resolve("/api/results/" + id + "/share"))
                    .header("Cookie", cookie)
                    .build(),
                HttpResponse.BodyHandlers.ofString());
        Assert.assertEquals(shared.statusCode(), 200);
        Assert.assertEquals(
            JsonParser.parseString(shared.body()).getAsJsonObject().get("url").getAsString(),
            "http://reviewer.example:8080/?share=" + id);
      }
      for (String suffix : List.of("", "/export", "/source")) {
        var denied =
            client.send(
                HttpRequest.newBuilder(base.resolve("/api/results/" + id + suffix)).build(),
                HttpResponse.BodyHandlers.ofString());
        Assert.assertEquals(denied.statusCode(), 404);
        var allowed =
            client.send(
                HttpRequest.newBuilder(base.resolve("/api/results/" + id + suffix))
                    .header("Cookie", cookie)
                    .build(),
                HttpResponse.BodyHandlers.ofString());
        Assert.assertEquals(allowed.statusCode(), 200);
        if (suffix.equals("/export"))
          Assert.assertTrue(
              allowed
                  .headers()
                  .firstValue("Content-Disposition")
                  .orElseThrow()
                  .contains(".epsilon-reviewer.json.gz"));
      }
      var deniedDelete =
          client.send(
              HttpRequest.newBuilder(base.resolve("/api/results/" + id)).DELETE().build(),
              HttpResponse.BodyHandlers.ofString());
      Assert.assertEquals(deniedDelete.statusCode(), 404);
      var delete =
          client.send(
              HttpRequest.newBuilder(base.resolve("/api/results/" + id))
                  .header("Cookie", cookie)
                  .DELETE()
                  .build(),
              HttpResponse.BodyHandlers.ofString());
      Assert.assertEquals(delete.statusCode(), 204);
    } finally {
      delete(data);
    }
  }

  @Test
  public void crossSiteMutationsAndExternalResultUploadsAreRejected() throws Exception {
    Path data = Files.createTempDirectory("reviewer-input-test-");
    try (var server =
            new ReviewerServer(
                new InetSocketAddress("127.0.0.1", 0),
                URI.create("http://reviewer.example:8080"),
                null,
                data,
                null);
        var client = HttpClient.newHttpClient()) {
      server.start();
      URI endpoint = URI.create("http://127.0.0.1:" + server.port() + "/api/records");
      var crossSite =
          client.send(
              HttpRequest.newBuilder(endpoint)
                  .header("Origin", "https://example.com")
                  .header("Content-Type", "application/json")
                  .POST(HttpRequest.BodyPublishers.ofString("{}"))
                  .build(),
              HttpResponse.BodyHandlers.ofString());
      Assert.assertEquals(crossSite.statusCode(), 403);
      JsonObject error =
          JsonParser.parseString(crossSite.body()).getAsJsonObject().getAsJsonObject("error");
      Assert.assertEquals(error.get("code").getAsString(), "origin_not_allowed");
      Assert.assertTrue(error.get("message").getAsString().chars().allMatch(c -> c < 128));
      for (String host : List.of("attacker.example", "localhost:" + (server.port() + 1))) {
        try (var socket = new Socket("127.0.0.1", server.port())) {
          socket.setSoTimeout(5000);
          socket
              .getOutputStream()
              .write(
                  ("GET /api/models HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n")
                      .getBytes(StandardCharsets.US_ASCII));
          String rejected =
              new String(socket.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
          Assert.assertTrue(rejected.startsWith("HTTP/1.1 403"), rejected);
          Assert.assertTrue(rejected.contains("host_not_allowed"), rejected);
        }
      }
      var result =
          client.send(
              HttpRequest.newBuilder(endpoint)
                  .header("Content-Type", "application/octet-stream")
                  .header("X-File-Name", "renamed.mjson")
                  .POST(
                      HttpRequest.BodyPublishers.ofString(
                          "{\"formatVersion\":1,\"resultId\":\"external\",\"rounds\":[]}"))
                  .build(),
              HttpResponse.BodyHandlers.ofString());
      Assert.assertEquals(result.statusCode(), 415);
      var json =
          client.send(
              HttpRequest.newBuilder(endpoint)
                  .header("Content-Type", "application/json")
                  .POST(
                      HttpRequest.BodyPublishers.ofString(
                          "{\"formatVersion\":1,\"resultId\":\"external\",\"rounds\":[]}"))
                  .build(),
              HttpResponse.BodyHandlers.ofString());
      Assert.assertEquals(json.statusCode(), 415);
      var localUrl =
          client.send(
              HttpRequest.newBuilder(endpoint)
                  .header("Content-Type", "application/json")
                  .POST(
                      HttpRequest.BodyPublishers.ofString("{\"url\":\"http://127.0.0.1/private\"}"))
                  .build(),
              HttpResponse.BodyHandlers.ofString());
      Assert.assertEquals(localUrl.statusCode(), 400);
      Assert.assertEquals(
          new ResultStore(data.resolve("results")).history("a".repeat(64)).size(), 0);
    } finally {
      delete(data);
    }
  }

  @Test
  public void compressedAndRenamedResultsFailDuringPreparationWithoutSavingData() throws Exception {
    Path data = Files.createTempDirectory("reviewer-result-input-");
    try (var server =
            new ReviewerServer(
                new InetSocketAddress("127.0.0.1", 0),
                URI.create("http://127.0.0.1:0"),
                null,
                data,
                null);
        var client = HttpClient.newHttpClient()) {
      server.start();
      URI base = URI.create("http://127.0.0.1:" + server.port());
      byte[] result =
          (" ".repeat(2048) + "{\"formatVersion\":1,\"resultId\":\"external\",\"rounds\":[]}")
              .getBytes(StandardCharsets.UTF_8);
      var compressed = new ByteArrayOutputStream();
      try (var gzip = new GZIPOutputStream(compressed)) {
        gzip.write(result);
      }
      for (byte[] bytes : List.of(result, compressed.toByteArray())) {
        var rejected =
            client.send(
                HttpRequest.newBuilder(base.resolve("/api/records"))
                    .header("Content-Type", "application/octet-stream")
                    .header("X-File-Name", "renamed.mjai")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(bytes))
                    .build(),
                HttpResponse.BodyHandlers.ofString());
        Assert.assertEquals(rejected.statusCode(), 422);
        Assert.assertEquals(
            JsonParser.parseString(rejected.body())
                .getAsJsonObject()
                .getAsJsonObject("error")
                .get("code")
                .getAsString(),
            "invalid_record");
        String cookie = rejected.headers().firstValue("Set-Cookie").orElseThrow().split(";", 2)[0];
        var history =
            client.send(
                HttpRequest.newBuilder(base.resolve("/api/history"))
                    .header("Cookie", cookie)
                    .build(),
                HttpResponse.BodyHandlers.ofString());
        Assert.assertEquals(history.statusCode(), 200);
        Assert.assertTrue(
            JsonParser.parseString(history.body())
                .getAsJsonObject()
                .getAsJsonArray("results")
                .isEmpty());
      }
      try (var paths = Files.walk(data.resolve("results"))) {
        Assert.assertTrue(
            paths.noneMatch(path -> path.getFileName().toString().equals("result.json.gz")));
      }
    } finally {
      delete(data);
    }
  }

  private static void delete(Path directory) throws Exception {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}

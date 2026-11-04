package com.epsilon.reviewer.http;

import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.engine.Format6;
import com.epsilon.reviewer.store.OwnerId;
import com.epsilon.reviewer.store.ResultId;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpRequest;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 共有の道は Cookie を読まず発行せず、持ち主の権利を与えない。削除で 404 になり、他のサイトからの共有リンクの移動だけを通し、Host と Origin の
 * 守りは緩めない。共有リンクのもしもはサーバの保存の牌譜で答える。
 */
public class SharingTest {
  @Test
  public void sharesNeitherReadNorSetTheCookieAndGrantNoOwnerRights() {
    try (var server = TestServer.start()) {
      ResultId id = server.reviewer.analyze(TestReviewer.OWNER, "kakan-a", TestReviewer.FIXTURE);
      String path = "/api/shares/" + id.value();
      var original = server.send(server.request("/api/results/" + id.value(), TestReviewer.OWNER));
      for (OwnerId viewer : new OwnerId[] {null, TestReviewer.OTHER}) {
        var shared = server.send(server.request(path, viewer));
        Assert.assertEquals(shared.statusCode(), 200);
        Assert.assertEquals(shared.body(), original.body());
        Assert.assertEquals(shared.headers().firstValue("X-Robots-Tag").orElseThrow(), "noindex");
        Assert.assertFalse(shared.headers().firstValue("Set-Cookie").isPresent());
      }
      for (String suffix : List.of("", "/download", "/share"))
        TestServer.assertError(
            server.send(server.request("/api/results/" + id.value() + suffix, TestReviewer.OTHER)),
            404,
            "result_not_found");
      TestServer.assertError(
          server.send(
              server.json("/api/results/" + id.value() + "/record", TestReviewer.OTHER, "")),
          404,
          "result_not_found");
      TestServer.assertError(
          server.send(server.request("/api/results/" + id.value(), TestReviewer.OTHER).DELETE()),
          404,
          "result_not_found");
      Assert.assertTrue(
          TestServer.body(server.send(server.request("/api/history", TestReviewer.OTHER)))
              .getAsJsonArray("results")
              .isEmpty());
      for (String missing :
          List.of(
              "invalid",
              "00000000-0000-4000-8000-000000000000",
              id.value().toUpperCase(Locale.ROOT))) {
        var refused = server.send(server.request("/api/shares/" + missing, null));
        TestServer.assertError(refused, 404, "share_not_found");
        Assert.assertFalse(refused.headers().firstValue("Set-Cookie").isPresent());
      }
      TestServer.assertError(
          server.send(server.request(path, null).DELETE()), 404, "not_found");
      TestServer.assertError(
          server.send(
              server.request(path + "/record", null).POST(HttpRequest.BodyPublishers.noBody())),
          404,
          "not_found");

      String body = server.send(server.request(path, null)).body();
      Format6.Located decision =
          HttpApiTest.firstChoice(Format6.decode(JsonParser.parseString(body).getAsJsonObject()));
      String query =
          "{\"round\":"
              + decision.round()
              + ",\"eventIndex\":"
              + decision.step().eventIndex()
              + ",\"seat\":"
              + decision.step().actor()
              + ",\"edits\":{}}";
      var answer = server.send(server.json(path + "/scenario", null, query));
      Assert.assertEquals(answer.statusCode(), 200, answer.body());
      Assert.assertFalse(answer.headers().firstValue("Set-Cookie").isPresent());
      Assert.assertEquals(TestServer.body(answer), decision.step().candidateSet());

      Assert.assertEquals(
          server
              .send(server.request("/api/results/" + id.value(), TestReviewer.OWNER).DELETE())
              .statusCode(),
          204);
      TestServer.assertError(server.send(server.request(path, null)), 404, "share_not_found");
      TestServer.assertError(
          server.send(server.json(path + "/scenario", null, query)), 404, "share_not_found");
    }
  }

  @Test
  public void onlyACrossSiteNavigationToTheSharedPageIsAllowed() throws IOException {
    Path web = TestFiles.directory("reviewer-web-");
    Files.writeString(web.resolve("index.html"), "<main>Shared review fixture</main>");
    try (var server = TestServer.start(URI.create("http://127.0.0.1:0"), web)) {
      ResultId id = server.reviewer.analyze(TestReviewer.OWNER, "kakan-a", TestReviewer.FIXTURE);
      for (String query :
          List.of(
              "share=" + id.value(),
              "%73hare=" + id.value(),
              "share=invalid",
              "share=",
              "share=" + id.value() + "&share=" + id.value(),
              "share=" + id.value() + "&other=1")) {
        var page = server.send(navigation(server, "/?" + query));
        Assert.assertEquals(page.statusCode(), 200, query);
        Assert.assertEquals(page.body(), "<main>Shared review fixture</main>");
        Assert.assertEquals(page.headers().firstValue("X-Robots-Tag").orElseThrow(), "noindex");
        Assert.assertFalse(page.headers().firstValue("Set-Cookie").isPresent());
      }
      for (String path :
          List.of("/", "/api/shares/" + id.value(), "/api/results/" + id.value() + "/share"))
        TestServer.assertError(server.send(navigation(server, path)), 403, "origin_not_allowed");
      TestServer.assertError(
          server.send(
              server
                  .request("/?share=" + id.value(), null)
                  .header("Sec-Fetch-Site", "cross-site")
                  .header("Sec-Fetch-Mode", "cors")
                  .header("Sec-Fetch-Dest", "document")),
          403,
          "origin_not_allowed");
      TestServer.assertError(
          server.send(
              navigation(server, "/?share=" + id.value()).header("Origin", "https://example.com")),
          403,
          "origin_not_allowed");
      TestServer.assertError(
          server.send(
              server
                  .request("/api/results/" + id.value(), TestReviewer.OWNER)
                  .header("Sec-Fetch-Site", "cross-site")
                  .DELETE()),
          403,
          "origin_not_allowed");
      try (var socket = new Socket("127.0.0.1", server.server.port())) {
        socket.setSoTimeout(5000);
        socket
            .getOutputStream()
            .write(
                ("GET /?share="
                        + id.value()
                        + " HTTP/1.1\r\nHost: attacker.example\r\n"
                        + "Sec-Fetch-Site: cross-site\r\nSec-Fetch-Mode: navigate\r\n"
                        + "Sec-Fetch-Dest: document\r\nConnection: close\r\n\r\n")
                    .getBytes(StandardCharsets.US_ASCII));
        String reply = new String(socket.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        Assert.assertTrue(reply.startsWith("HTTP/1.1 403"), reply);
        Assert.assertTrue(reply.contains("host_not_allowed"), reply);
      }
      var owned = server.request("/api/results/" + id.value(), TestReviewer.OWNER);
      Assert.assertEquals(server.send(owned).statusCode(), 200);
    } finally {
      TestFiles.delete(web);
    }
  }

  @Test
  public void shareLinksUseTheConfiguredOrigin() {
    try (var server = TestServer.start(URI.create("http://reviewer.example:8080"), null)) {
      ResultId id = server.reviewer.analyze(TestReviewer.OWNER, "kakan-a", TestReviewer.FIXTURE);
      var link =
          server.send(
              server.request("/api/results/" + id.value() + "/share", TestReviewer.OWNER));
      Assert.assertEquals(
          TestServer.body(link).get("url").getAsString(),
          "http://reviewer.example:8080/?share=" + id.value());
    }
  }

  private static HttpRequest.Builder navigation(TestServer server, String path) {
    return server
        .request(path, null)
        .header("Sec-Fetch-Site", "cross-site")
        .header("Sec-Fetch-Mode", "navigate")
        .header("Sec-Fetch-Dest", "document");
  }
}

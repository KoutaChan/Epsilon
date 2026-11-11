package com.epsilon.reviewer.http;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.engine.Format6;
import com.epsilon.reviewer.store.OwnerId;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.http.HttpRequest;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/** HTTP のパスがある全ての操作を、テスト用のモデルで端から端まで通し、ステータス・本文・ヘッダーを確かめる。 */
public class HttpApiTest {
  private static final OwnerId OWNER = TestReviewer.OWNER;
  private TestServer server;

  @BeforeClass
  public void start() {
    server = TestServer.start();
  }

  @AfterClass(alwaysRun = true)
  public void stop() {
    server.close();
  }

  @Test
  public void everyCommandRunsEndToEnd() {
    var models = server.send(server.request("/api/models", null));
    Assert.assertEquals(models.statusCode(), 200);
    Assert.assertEquals(models.headers().firstValue("Cache-Control").orElseThrow(), "no-store");
    Assert.assertFalse(models.headers().firstValue("Set-Cookie").isPresent());
    JsonObject catalog = TestServer.body(models);
    Assert.assertEquals(catalog.getAsJsonArray("models").size(), 3);
    Assert.assertEquals(
        catalog.getAsJsonArray("models").get(0).getAsJsonObject().toString(),
        "{\"modelId\":\"fixture\",\"revision\":\"r1\",\"displayName\":\"Fixture\","
            + "\"version\":\"1\",\"seats\":4,\"available\":true}");
    Assert.assertEquals(
        catalog.getAsJsonObject("defaultModel").toString(),
        "{\"modelId\":\"fixture\",\"revision\":\"r1\"}");

    var imported =
        server.send(
            server
                .request("/api/records", OWNER)
                .header("Content-Type", "application/octet-stream")
                .POST(HttpRequest.BodyPublishers.ofByteArray(TestFiles.played(Variant.TENHOU_4P, 0))));
    Assert.assertEquals(imported.statusCode(), 200, imported.body());
    JsonObject record = TestServer.body(imported);
    Assert.assertEquals(record.getAsJsonObject("record").get("source").getAsString(), "tenhou");
    Assert.assertEquals(record.getAsJsonObject("record").getAsJsonArray("names").size(), 4);

    var started =
        server.send(
            server.json(
                "/api/analyses",
                OWNER,
                "{\"recordId\":\""
                    + record.get("recordId").getAsString()
                    + "\",\"modelId\":\"fixture\",\"revision\":\"r1\"}"));
    Assert.assertEquals(started.statusCode(), 200, started.body());
    String jobId = TestServer.body(started).get("jobId").getAsString();
    Assert.assertEquals(
        TestServer.body(started).getAsJsonObject("model").get("displayName").getAsString(),
        "Fixture");
    JsonObject job = awaitJob(jobId);
    Assert.assertEquals(job.get("status").getAsString(), "completed", job.toString());
    Assert.assertEquals(job.get("progress").getAsInt(), 100);
    Assert.assertFalse(job.has("error"));
    String id = job.get("resultId").getAsString();
    var cancelled = server.send(server.json("/api/jobs/" + jobId + "/cancel", OWNER, "{}"));
    Assert.assertEquals(TestServer.body(cancelled).get("status").getAsString(), "completed");

    var history = server.send(server.request("/api/history", OWNER));
    JsonObject entry = TestServer.body(history).getAsJsonArray("results").get(0).getAsJsonObject();
    Assert.assertEquals(entry.get("resultId").getAsString(), id);
    Assert.assertEquals(entry.getAsJsonObject("model").get("modelId").getAsString(), "fixture");

    var result =
        server.bytes(server.request("/api/results/" + id, OWNER).header("Accept-Encoding", "gzip"));
    Assert.assertEquals(result.statusCode(), 200);
    Assert.assertEquals(result.headers().firstValue("Content-Encoding").orElseThrow(), "gzip");
    JsonObject document = JsonParser.parseString(gunzip(result.body())).getAsJsonObject();
    Assert.assertEquals(document.get("format").getAsInt(), 6);
    Assert.assertEquals(document.get("resultId").getAsString(), id);

    var restored = server.send(server.json("/api/results/" + id + "/record", OWNER, ""));
    Assert.assertEquals(restored.statusCode(), 200, restored.body());
    Assert.assertTrue(TestServer.body(restored).has("recordId"));

    var download = server.bytes(server.request("/api/results/" + id + "/download", OWNER));
    Assert.assertEquals(download.statusCode(), 200);
    Assert.assertEquals(
        download.headers().firstValue("Content-Type").orElseThrow(), "application/gzip");
    Assert.assertEquals(download.body(), result.body());

    var share = server.send(server.request("/api/results/" + id + "/share", OWNER));
    Assert.assertEquals(
        TestServer.body(share).get("url").getAsString(), server.base() + "/?share=" + id);
    var shared = server.bytes(server.request("/api/shares/" + id, null));
    Assert.assertEquals(shared.statusCode(), 200);
    Assert.assertFalse(shared.headers().firstValue("Set-Cookie").isPresent());

    Format6.Located decision = firstChoice(Format6.decode(document));
    String query =
        "{\"round\":"
            + decision.round()
            + ",\"eventIndex\":"
            + decision.step().eventIndex()
            + ",\"seat\":"
            + decision.step().actor()
            + ",\"edits\":{}}";
    for (String path :
        new String[] {"/api/results/" + id + "/scenario", "/api/shares/" + id + "/scenario"}) {
      var answer = server.send(server.json(path, path.contains("shares") ? null : OWNER, query));
      Assert.assertEquals(answer.statusCode(), 200, answer.body());
      Assert.assertEquals(TestServer.body(answer), decision.step().candidateSet(), path);
    }

    var deleted = server.send(server.request("/api/results/" + id, OWNER).DELETE());
    Assert.assertEquals(deleted.statusCode(), 204);
    TestServer.assertError(
        server.send(server.request("/api/results/" + id, OWNER)), 404, "result_not_found");
    TestServer.assertError(
        server.send(server.request("/api/shares/" + id, null)), 404, "share_not_found");
  }

  @Test
  public void requestsWithTheWrongShapeHaveStableCodes() {
    TestServer.assertError(server.send(server.request("/api/nothing", OWNER)), 404, "not_found");
    TestServer.assertError(
        server.send(server.request("/api/models", null).POST(HttpRequest.BodyPublishers.noBody())),
        404,
        "not_found");
    TestServer.assertError(
        server.send(
            server
                .request("/api/records/tenhou", OWNER)
                .header("Content-Type", "text/plain")
                .POST(HttpRequest.BodyPublishers.ofString("{\"url\":\"x\"}"))),
        400,
        "invalid_request");
    TestServer.assertError(
        server.send(
            server.json(
                "/api/records/tenhou", OWNER, "{\"url\":\"" + "a".repeat(17 * 1024) + "\"}")),
        400,
        "invalid_request");
    TestServer.assertError(
        server.send(
            server.json(
                "/api/records/tenhou",
                OWNER,
                "{\"url\":\"https://game.mahjongsoul.com/?paipu=x\"}")),
        400,
        "unsupported_url");
    TestServer.assertError(
        server.send(
            server.json("/api/records/tenhou", OWNER, "{\"url\":\"https://tenhou.net/0/?log=x\"}")),
        400,
        "invalid_url");
    TestServer.assertError(
        server.send(
            server
                .request("/api/records", OWNER)
                .header("Content-Type", "application/octet-stream")
                .POST(HttpRequest.BodyPublishers.ofString("{\"format\":6,\"resultId\":\"x\"}"))),
        422,
        "invalid_record");
    // 長さを書かない本文（chunked）で、上限より 1 バイト多く送る。サーバは本文を読み切ってから断る。
    TestServer.assertError(
        server.send(
            server
                .request("/api/records", OWNER)
                .header("Content-Type", "application/octet-stream")
                .POST(
                    HttpRequest.BodyPublishers.ofInputStream(
                        () -> new ByteArrayInputStream(new byte[16 * 1024 * 1024 + 1])))),
        413,
        "record_too_large");
    TestServer.assertError(
        server.send(
            server.json(
                "/api/analyses",
                OWNER,
                "{\"recordId\":\"00000000-0000-4000-8000-000000000000\","
                    + "\"modelId\":\"fixture\",\"revision\":\"r1\"}")),
        404,
        "record_expired");
    TestServer.assertError(
        server.send(
            server.json(
                "/api/analyses",
                OWNER,
                "{\"recordId\":\"x\",\"modelId\":\"fixture\",\"revision\":\"r1\"}")),
        404,
        "record_expired");
    TestServer.assertError(
        server.send(server.json("/api/analyses", OWNER, "{\"recordId\":\"x\",\"modelId\":\"m\"}")),
        400,
        "invalid_request");
    TestServer.assertError(
        server.send(server.request("/api/jobs/00000000-0000-4000-8000-000000000000", OWNER)),
        404,
        "job_not_found");
    TestServer.assertError(
        server.send(server.request("/api/results/NOT-A-UUID", OWNER)), 404, "result_not_found");
    String scenario = "/api/results/00000000-0000-4000-8000-000000000000/scenario";
    TestServer.assertError(
        server.send(server.json(scenario, OWNER, "{\"round\":0,\"eventIndex\":1,\"seat\":0}")),
        400,
        "invalid_request");
    TestServer.assertError(
        server.send(
            server.json(
                scenario,
                OWNER,
                "{\"resultId\":\"x\",\"round\":0,\"eventIndex\":1,\"seat\":0,\"edits\":{}}")),
        400,
        "invalid_request");
    TestServer.assertError(
        server.send(
            server.json(scenario, OWNER, "{\"round\":0,\"eventIndex\":1,\"seat\":4,\"edits\":{}}")),
        400,
        "invalid_request");
    TestServer.assertError(
        server.send(
            server.json(scenario, OWNER, "{\"round\":0,\"eventIndex\":1,\"seat\":0,\"edits\":{}}")),
        404,
        "result_not_found");
  }

  /** 候補が 2 つ以上ある打牌の判断のうち、最初のもの。 */
  static Format6.Located firstChoice(Format6.Document document) {
    return document.decisions().stream()
        .filter(located -> located.step().candidates().size() > 1)
        .filter(located -> located.step().chosenType().equals("DAHAI"))
        .findFirst()
        .orElseThrow();
  }

  static String gunzip(byte[] gzip) {
    try (var in = new GZIPInputStream(new ByteArrayInputStream(gzip))) {
      return new String(in.readAllBytes(), StandardCharsets.UTF_8);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  private JsonObject awaitJob(String jobId) {
    long deadline = System.nanoTime() + Duration.ofSeconds(30).toNanos();
    while (System.nanoTime() < deadline) {
      JsonObject job = TestServer.body(server.send(server.request("/api/jobs/" + jobId, OWNER)));
      String status = job.get("status").getAsString();
      if (status.equals("completed") || status.equals("failed") || status.equals("cancelled"))
        return job;
      try {
        Thread.sleep(20);
      } catch (InterruptedException failure) {
        Thread.currentThread().interrupt();
        throw new AssertionError(failure);
      }
    }
    throw new AssertionError("The job did not finish.");
  }
}

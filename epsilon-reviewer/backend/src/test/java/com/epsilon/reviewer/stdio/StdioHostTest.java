package com.epsilon.reviewer.stdio;

import com.epsilon.reviewer.Reviewer;
import com.epsilon.reviewer.ReviewerConfig;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.TestReviewer;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.Pipe;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/** stdio の操作を管で通す: models・importBytes の本文・解析・result の本文・失敗の答え・知らない操作・stdin の終わりで閉じる。 */
public class StdioHostTest {
  @Test
  public void aScriptedSessionAnswersEveryRequestAndClosesAtTheEnd() throws Exception {
    Path directory = TestFiles.directory("reviewer-stdio-");
    try {
      ReviewerConfig config = TestReviewer.config(directory);
      Pipe requests = Pipe.open();
      Pipe replies = Pipe.open();
      var host =
          new StdioHost(
              Reviewer.open(config),
              new FrameReader(Channels.newInputStream(requests.source())),
              new FrameWriter(Channels.newOutputStream(replies.sink())));
      AtomicReference<Throwable> failure = new AtomicReference<>();
      Thread engine =
          Thread.ofPlatform()
              .start(
                  () -> {
                    try {
                      host.run();
                    } catch (Throwable thrown) {
                      failure.set(thrown);
                    }
                  });
      var session = new Session(requests, replies);

      JsonObject models = session.call("models", "{}", null);
      Assert.assertTrue(models.get("ok").getAsBoolean(), models.toString());
      Assert.assertEquals(models.getAsJsonObject("response").getAsJsonArray("models").size(), 3);

      JsonObject imported = session.call("importBytes", "{}", TestFiles.golden("kakan-a"));
      String recordId = imported.getAsJsonObject("response").get("recordId").getAsString();
      JsonObject started =
          session.call(
              "startAnalysis",
              "{\"recordId\":\"" + recordId + "\",\"modelId\":\"fixture\",\"revision\":\"r1\"}",
              null);
      String jobId = started.getAsJsonObject("response").get("jobId").getAsString();
      JsonObject job = null;
      long deadline = System.nanoTime() + Duration.ofSeconds(30).toNanos();
      while (System.nanoTime() < deadline) {
        job =
            session.call("job", "{\"jobId\":\"" + jobId + "\"}", null).getAsJsonObject("response");
        if (job.get("status").getAsString().equals("completed")) break;
        Thread.sleep(20);
      }
      Assert.assertEquals(job.get("status").getAsString(), "completed");
      String resultId = job.get("resultId").getAsString();
      String resultRequest = "{\"resultId\":\"" + resultId + "\"}";
      JsonObject result = session.call("result", resultRequest, null);
      Assert.assertTrue(result.get("ok").getAsBoolean());
      Assert.assertFalse(result.has("response"));
      try (var gzip = new GZIPInputStream(new ByteArrayInputStream(session.lastBody))) {
        JsonObject document =
            JsonParser.parseString(new String(gzip.readAllBytes(), StandardCharsets.UTF_8))
                .getAsJsonObject();
        Assert.assertEquals(document.get("resultId").getAsString(), resultId);
      }
      Assert.assertEquals(
          session
              .call("history", "{}", null)
              .getAsJsonObject("response")
              .getAsJsonArray("results")
              .size(),
          1);

      refused(session.call("downloadResult", resultRequest, null), "invalid_request");
      refused(session.call("nope", "{}", null), "invalid_request");
      refused(session.call("job", "{\"jobId\":\"x\"}", null), "job_not_found");
      refused(session.call("history", "{\"extra\":1}", null), "invalid_request");
      refused(session.call("models", "{}", new byte[] {1}), "invalid_request");
      refused(
          session.call("importFile", "{\"path\":\"relative.xml\"}", null), "invalid_request");
      JsonObject deleted = session.call("deleteResult", resultRequest, null);
      Assert.assertTrue(deleted.get("ok").getAsBoolean());
      Assert.assertFalse(deleted.has("response"));

      requests.sink().close();
      engine.join(Duration.ofSeconds(30));
      Assert.assertFalse(engine.isAlive());
      Assert.assertNull(failure.get());
      // stdin の終わりで閉じたので、同じ保存先をもう一度開ける。
      Reviewer.open(config).close();
    } finally {
      TestFiles.delete(directory);
    }
  }

  private static void refused(JsonObject reply, String code) {
    Assert.assertFalse(reply.get("ok").getAsBoolean(), reply.toString());
    Assert.assertEquals(reply.getAsJsonObject("error").get("code").getAsString(), code);
  }

  /** 要求を 1 つずつ送り、同じ id の答えを待つ（答えは順に来るとは限らない）。 */
  private static final class Session {
    private final FrameWriter out;
    private final FrameReader in;
    private final Map<Long, Frame> early = new HashMap<>();
    private long next = 1;
    byte[] lastBody;

    Session(Pipe requests, Pipe replies) {
      out = new FrameWriter(Channels.newOutputStream(requests.sink()));
      in = new FrameReader(Channels.newInputStream(replies.source()));
    }

    JsonObject call(String command, String request, byte[] body) throws IOException {
      long id = next++;
      JsonObject header = new JsonObject();
      header.addProperty("id", id);
      header.addProperty("command", command);
      header.add("request", JsonParser.parseString(request));
      out.write(header, body);
      Frame frame = early.remove(id);
      while (frame == null) {
        Frame received = in.read().orElseThrow();
        long replyId = received.header().get("id").getAsLong();
        if (replyId == id) frame = received;
        else early.put(replyId, received);
      }
      lastBody = frame.body();
      return frame.header();
    }
  }
}

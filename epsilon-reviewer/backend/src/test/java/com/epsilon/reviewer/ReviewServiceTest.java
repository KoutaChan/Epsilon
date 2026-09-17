package com.epsilon.reviewer;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.reviewer.engine.ReviewEngine;
import com.epsilon.reviewer.model.ModelCatalog;
import com.google.gson.*;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/** モデルの読み込み失敗、解析の完了、待機中ジョブの取消しに対する結果保存の状態を検証する。 */
public final class ReviewServiceTest {
  @Test
  public void failedModelLoadReportsStableErrorAndLeavesNoResult() throws Exception {
    Path directory = Files.createTempDirectory("reviewer-failed-job-");
    try {
      Path checkpoint = Files.createDirectory(directory.resolve("checkpoint"));
      Files.writeString(checkpoint.resolve("incomplete.txt"), "not a model checkpoint");
      Path config = directory.resolve("models.json");
      Files.writeString(
          config,
          """
          {"models":[{"modelId":"broken","revision":"r1","displayName":"Broken",
          "version":"1","series":"epsilon-nano","checkpoint":"checkpoint",
          "options":{"device":"cpu"}}]}
          """);
      var store = new ResultStore(directory.resolve("results"));
      try (var service =
          new ReviewService(
              new ReviewEngine(),
              new ModelCatalog(config),
              store,
              directory.resolve("temporary"))) {
        String owner = "a".repeat(64);
        var source =
            new RecordFetcher.SourceRecord(
                xml().getBytes(StandardCharsets.UTF_8), "record.xml", "tenhou");
        var prepared = service.prepare(owner, source);
        var accepted = service.analyze(owner, prepared.recordId(), "broken", "r1", false);
        var failed = await(service, owner, accepted.jobId());
        Assert.assertEquals(failed.status(), "failed");
        Assert.assertEquals(failed.error().code(), "analysis_failed");
        Assert.assertTrue(failed.error().message().chars().allMatch(c -> c < 128));
        Assert.assertNull(failed.resultId());
        Assert.assertTrue(store.history(owner).isEmpty());
        Assert.assertEquals(service.cancel(owner, accepted.jobId()).status(), "failed");
      }
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  @Test(groups = "native")
  public void realAnalysisCommitsBeforeCompletionAndCancelledQueuedJobLeavesNoResult()
      throws Exception {
    Path directory = Files.createTempDirectory("reviewer-jobs-test-");
    try {
      Path checkpoint = directory.resolve("checkpoint");
      try (Model model =
          NetworkFactory.createDecisionModel(Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
        EpsilonDecisionCheckpointManager.saveInitial(model, checkpoint);
      }
      Path config = directory.resolve("models.json");
      Files.writeString(
          config,
          "{\"models\":[{\"modelId\":\"fixture\",\"revision\":\"fixed\",\"displayName\":\"Fixture\",\"version\":\"1\",\"series\":\"epsilon-nano\",\"checkpoint\":\"checkpoint\",\"options\":{\"device\":\"cpu\"}}]}");
      var catalog = new ModelCatalog(config);
      var store = new ResultStore(directory.resolve("results"));
      try (var service =
          new ReviewService(new ReviewEngine(), catalog, store, directory.resolve("temporary"))) {
        String owner = "a".repeat(64), other = "b".repeat(64);
        var source =
            new RecordFetcher.SourceRecord(
                xml().getBytes(StandardCharsets.UTF_8), "record.xml", "tenhou");
        String first = service.prepare(owner, source).recordId();
        String second = service.prepare(other, source).recordId();
        String firstJob = service.analyze(owner, first, "fixture", "fixed", false).jobId();
        String secondJob = service.analyze(other, second, "fixture", "fixed", true).jobId();
        Assert.expectThrows(ApiException.class, () -> service.job(other, firstJob));
        service.cancel(other, secondJob);
        ReviewService.JobSnapshot complete = await(service, owner, firstJob);
        Assert.assertEquals(complete.status(), "completed", complete.toString());
        String resultId = complete.resultId();
        JsonObject result;
        try (var content = store.openResult(owner, resultId);
            var gzip = new GZIPInputStream(content.input());
            var reader = new InputStreamReader(gzip, StandardCharsets.UTF_8)) {
          result = JsonParser.parseReader(reader).getAsJsonObject();
        }
        Assert.assertEquals(result.getAsJsonObject("model").get("revision").getAsString(), "fixed");
        Assert.assertFalse(result.getAsJsonArray("rounds").isEmpty());
        Assert.assertEquals(store.history(owner).size(), 1);
        Assert.assertEquals(store.source(owner, resultId).bytes(), source.bytes());
        Assert.assertEquals(await(service, other, secondJob).status(), "cancelled");
        Assert.assertEquals(store.history(other).size(), 0);
        try (var paths = Files.list(directory.resolve("temporary"))) {
          Assert.assertEquals(paths.count(), 0L);
        }
      }
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  private static ReviewService.JobSnapshot await(ReviewService service, String owner, String jobId)
      throws Exception {
    long deadline = System.nanoTime() + Duration.ofSeconds(30).toNanos();
    while (System.nanoTime() < deadline) {
      ReviewService.JobSnapshot job = service.job(owner, jobId);
      if (Set.of("completed", "cancelled", "failed").contains(job.status())) return job;
      Thread.sleep(10);
    }
    throw new AssertionError("Analysis job did not complete before the deadline.");
  }

  private static String xml() {
    StringBuilder result =
        new StringBuilder(
            "<mjloggm><GO type=\"169\"/><UN n0=\"A\" n1=\"B\" n2=\"C\" n3=\"D\"/><INIT"
                + " seed=\"0,0,0,0,0,132\" ten=\"250,250,250,250\" oya=\"0\"");
    for (int seat = 0; seat < 4; seat++) {
      List<String> hand = new ArrayList<>();
      for (int i = 0; i < 13; i++) hand.add(Integer.toString(seat * 13 + i));
      result.append(" hai").append(seat).append("=\"").append(String.join(",", hand)).append("\"");
    }
    return result
        .append(
            "/><T128/><D128/><RYUUKYOKU sc=\"250,0,250,0,250,0,250,0\""
                + " owari=\"250,0,250,0,250,0,250,0\"/></mjloggm>")
        .toString();
  }
}

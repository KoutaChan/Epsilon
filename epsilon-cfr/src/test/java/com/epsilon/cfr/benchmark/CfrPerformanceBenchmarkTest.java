package com.epsilon.cfr.benchmark;

import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrEpisodeFrame;
import com.epsilon.cfr.data.CfrEpisodeWriter;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 少数半荘の診断が主経路を過不足なく数え、元データを変更しないことを確かめる。 */
public class CfrPerformanceBenchmarkTest {
  @Test
  public void boundedInspectionCountsRealCandidatesAndPaddingWithoutReadingOtherMatches()
      throws Exception {
    Path directory = Files.createTempDirectory("cfr-shape-diagnostic-");
    try {
      CfrDecision forced = decision(0, 1);
      CfrDecision choices = decision(1, 3);
      try (var writer = new CfrEpisodeWriter(directory)) {
        writer.accept(
            new CfrEpisodeFrame(new CfrEpisode(7, List.of(forced), new double[4]), 0, false));
        writer.accept(
            new CfrEpisodeFrame(new CfrEpisode(7, List.of(choices), new double[4]), 1, true));
      }
      byte[] original = Files.readAllBytes(directory.resolve("00000007.bin"));
      Files.write(directory.resolve("b.bin"), new byte[] {1});
      Files.write(directory.resolve("0.bin.part"), new byte[] {2});

      Map<String, Object> report = CfrPerformanceBenchmark.inspectMatches(directory, 1);
      Assert.assertEquals(report.get("availableMatches"), 2);
      Assert.assertEquals(report.get("inspectedMatches"), 1);
      Map<String, Object> observations = map(report.get("observations"));
      Assert.assertEquals(observations.get("rows"), 2L);
      Assert.assertEquals(observations.get("legalActions"), 4L);
      Assert.assertEquals(observations.get("actualTransitions"), 4L);
      Assert.assertEquals(observations.get("oneTransitionPerActionRows"), 2L);
      Assert.assertEquals(observations.get("legalActionCountRows"), Map.of(1, 1L, 3, 1L));
      Assert.assertEquals(observations.get("transitionCountActions"), Map.of(1, 4L));
      Assert.assertEquals(observations.get("naturalBucketRows"), Map.of("1x1", 1L, "4x1", 1L));
      Assert.assertEquals(observations.get("storedActionSlots"), 17L);
      Assert.assertEquals(observations.get("storedTransitionSlots"), 17L);
      Assert.assertEquals(observations.get("naturalActionSlots"), 5L);
      Assert.assertEquals(observations.get("naturalTransitionSlots"), 5L);
      long dense = (long) forced.observation().size() + choices.observation().size();
      long padding = 13L * (int) report.get("bytesPerTransition");
      Assert.assertEquals(observations.get("densePayloadBytes"), dense);
      Assert.assertEquals(observations.get("transitionPaddingBytes"), padding);
      Assert.assertEquals(
          observations.get("compactPayloadBytesExcludingMetadata"), dense - padding);
      Assert.assertEquals(observations.get("structuralZeroRuns"), 6L);
      Assert.assertEquals(map(report.get("byStoredBucket")).keySet(), Set.of("1x1", "16x1"));
      var match = map(((List<?>) report.get("matches")).getFirst());
      Assert.assertEquals(match.get("trajectories"), 2);
      Assert.assertEquals(match.get("decisions"), 2L);
      Assert.assertEquals(Files.readAllBytes(directory.resolve("00000007.bin")), original);
      Assert.assertEquals(Files.readAllBytes(directory.resolve("b.bin")), new byte[] {1});
      Assert.assertEquals(Files.readAllBytes(directory.resolve("0.bin.part")), new byte[] {2});
      Assert.expectThrows(
          IOException.class, () -> CfrPerformanceBenchmark.inspectMatches(directory, 2));
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> CfrPerformanceBenchmark.inspectMatches(directory, 0));
    } finally {
      clean(directory);
    }
  }

  @Test
  public void inspectionRejectsDecisionMetadataThatDisagreesWithTheObservation() throws Exception {
    Path directory = Files.createTempDirectory("cfr-shape-invalid-");
    try {
      CfrDecision original = decision(0, 3);
      CfrDecision inconsistent =
          new CfrDecision(
              0, original.seat(), new int[] {1}, 0, new double[] {1}, 1, original.observation());
      try (var writer = new CfrEpisodeWriter(directory)) {
        writer.accept(
            new CfrEpisodeFrame(new CfrEpisode(7, List.of(inconsistent), new double[4]), 0, true));
      }
      Assert.expectThrows(
          IOException.class, () -> CfrPerformanceBenchmark.inspectMatches(directory, 32));
    } finally {
      clean(directory);
    }
  }

  @Test
  public void transitionSuffixPaddingMergesWithUnusedActionsAndNaturalBytesMatchEncoding()
      throws Exception {
    Path directory = Files.createTempDirectory("cfr-shape-transition-padding-");
    try {
      CfrDecision padded = decision(0, 2, new DecisionBucket(4, 16));
      CfrDecision natural = decision(0, 2, new DecisionBucket(2, 1));
      try (var writer = new CfrEpisodeWriter(directory)) {
        writer.accept(
            new CfrEpisodeFrame(new CfrEpisode(7, List.of(padded), new double[4]), 0, true));
      }
      Map<String, Object> report = CfrPerformanceBenchmark.inspectMatches(directory, 32);
      Map<String, Object> observations = map(report.get("observations"));
      Assert.assertEquals(observations.get("actualTransitions"), 2L);
      Assert.assertEquals(observations.get("storedTransitionSlots"), 64L);
      Assert.assertEquals(
          observations.get("transitionPaddingBytes"), 62L * (int) report.get("bytesPerTransition"));
      Assert.assertEquals(observations.get("structuralZeroRuns"), 12L);
      Assert.assertEquals(
          observations.get("naturalBucketDensePayloadBytes"), (long) natural.observation().size());
    } finally {
      clean(directory);
    }
  }

  private static CfrDecision decision(long id, int actions) {
    return decision(id, actions, DecisionBucket.forInferenceCounts(actions, 1));
  }

  private static CfrDecision decision(long id, int actions, DecisionBucket bucket) {
    var engine = new GameEngine(772);
    var step = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var point = step.decisions().getFirst();
    var context =
        new DecisionContext(
            7,
            id,
            point.player(),
            engine.observation(point.player()),
            point.legalActions().subList(0, actions));
    var builder = DecisionBatchBuilder.inference(1, bucket);
    builder.encodeObservedInferenceRow(0, context.observation(), context.legalActions());
    int[] ids = IntStream.range(0, actions).toArray();
    double[] policy = new double[actions];
    Arrays.fill(policy, 1d / actions);
    return new CfrDecision(
        id,
        point.player(),
        ids,
        0,
        policy,
        1d / actions,
        CfrObservationCodec.encodePayload(builder.buildEncodedInferenceRows(), 0));
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> map(Object value) {
    return (Map<String, Object>) value;
  }

  private static void clean(Path directory) throws IOException {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}

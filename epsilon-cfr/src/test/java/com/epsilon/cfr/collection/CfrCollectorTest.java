package com.epsilon.cfr.collection;

import static com.epsilon.cfr.collection.CfrCollectorTestSupport.configuration;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.cfr.algorithm.LcfrTargets;
import com.epsilon.cfr.collection.CfrCollectorTestSupport.Collector;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.Action;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrCollectorTest {
  private static final CfrPolicy UNIFORM =
      batch ->
          batch.stream()
              .map(
                  context -> {
                    Assert.assertEquals(context.observation().observer(), context.seat());
                    Assert.expectThrows(
                        IllegalArgumentException.class,
                        () -> context.observation().hand((context.seat() + 1) % 4).count(0));
                    double[] probabilities = new double[context.legalActions().size()];
                    Arrays.fill(probabilities, 1.0 / probabilities.length);
                    return new Prediction(
                        probabilities, CfrObservationCodec.encodePayload(context));
                  })
              .toList();

  @Test
  public void realRoundsAreDeterministicAcrossBatchAndConcurrencySizes() {
    List<Long> ids = List.of(3L, 7L, 9L, 11L);
    BoundaryValue boundary = prefix -> new double[] {0.1, -0.1, 0.2, -0.2};
    var serial = collector(CfrCollector.Mode.HAND_GRP, 1, 1, true).collect(ids, UNIFORM, boundary);
    var batched =
        collector(CfrCollector.Mode.HAND_GRP, 16, 4, true).collect(ids, UNIFORM, boundary);
    Assert.assertEquals(batched.stats().games(), 4L);
    Assert.assertEquals(serial.stats().largestBatch(), 1);
    Assert.assertTrue(batched.stats().largestBatch() >= 1 && batched.stats().largestBatch() <= 16);
    Assert.assertEquals(batched.stats().hands(), batched.stats().trajectories());
    Assert.assertTrue(batched.stats().hands() > batched.stats().games());
    Assert.assertEquals(batched.episodes().size(), serial.episodes().size());
    for (int i = 0; i < serial.episodes().size(); i++)
      equalEpisode(serial.episodes().get(i), batched.episodes().get(i));
    boolean[] observed = new boolean[4];
    int responses = 0;
    for (CfrEpisode episode : batched.episodes()) {
      for (CfrDecision decision : episode.decisions()) {
        observed[decision.seat()] = true;
        if (Arrays.stream(decision.legalActionIds())
            .mapToObj(Action::fromIndex)
            .anyMatch(action -> action.type().isResponse())) responses++;
      }
    }
    Assert.assertEquals(observed, new boolean[] {true, true, true, true});
    Assert.assertTrue(responses > 0);
  }

  @Test(timeOut = 20000)
  public void compatibleActionBucketsShareBatchesWithoutChangingLimitsOrTeachers() {
    List<Long> ids = List.of(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L);
    BoundaryValue boundary = ignored -> new double[4];
    AtomicInteger multipleTransitions = new AtomicInteger();
    AtomicInteger mixedActionBuckets = new AtomicInteger();
    CfrPolicy policy =
        new CfrPolicy() {
          @Override
          public boolean needsInferenceWork() {
            return false;
          }

          @Override
          public List<Prediction> evaluate(List<DecisionContext> batch) {
            var bucket =
                DecisionBatchBuilder.selectInferenceBucket(batch.getFirst().legalActions());
            Assert.assertTrue(batch.size() <= (bucket.actionTransitionCapacity() > 1 ? 2 : 16));
            if (bucket.actionTransitionCapacity() > 1) multipleTransitions.incrementAndGet();
            boolean mixed = false;
            for (var context : batch) {
              var actualBucket = DecisionBatchBuilder.selectInferenceBucket(context.legalActions());
              Assert.assertEquals(
                  actualBucket.actionTransitionCapacity(), bucket.actionTransitionCapacity());
              mixed |= actualBucket.legalActionCapacity() != bucket.legalActionCapacity();
            }
            if (mixed) mixedActionBuckets.incrementAndGet();
            return UNIFORM.evaluate(batch);
          }
        };
    var configured =
        new Collector(
            new CfrCollector.Config(
                CfrCollector.Mode.HAND_GRP,
                .1,
                16,
                8,
                901L,
                true,
                true,
                2,
                new DecisionDuelArenaSettings(3, 5, 1),
                new InferenceBatchingSettings(60_000_000),
                EpsilonUtilityProfile.TENHOU));
    var actual = configured.collect(ids, policy, boundary);
    var expected =
        collector(CfrCollector.Mode.HAND_GRP, 1, 1, true).collect(ids, UNIFORM, boundary);
    Assert.assertEquals(actual.stats().games(), (long) ids.size());
    Assert.assertTrue(
        multipleTransitions.get() > 0, "The fixture must contain meld follow-up choices");
    Assert.assertTrue(
        mixedActionBuckets.get() > 0, "Compatible action capacities must share an inference batch");
    Assert.assertEquals(actual.episodes().size(), expected.episodes().size());
    for (int episode = 0; episode < expected.episodes().size(); episode++) {
      equalEpisode(actual.episodes().get(episode), expected.episodes().get(episode));
      var before = LcfrTargets.generate(expected.episodes().get(episode), 3, ids.size());
      var after = LcfrTargets.generate(actual.episodes().get(episode), 3, ids.size());
      Assert.assertEquals(after.size(), before.size());
      for (int row = 0; row < before.size(); row++)
        Assert.assertTrue(after.get(row).contentEquals(before.get(row)), "LCFR target changed");
    }
  }

  @Test
  public void matchesKeepPublicRootsAcrossRoundBoundariesAndCanStreamOnlyOutcomes() {
    List<CfrCollector.CollectedEpisode> completed = new ArrayList<>();
    var stats =
        collector(CfrCollector.Mode.MATCH_MC, 8, 2, false)
            .collect(List.of(20L, 21L), UNIFORM, null, completed::add);
    Assert.assertEquals(stats.games(), 2L);
    Assert.assertEquals(stats.trajectories(), 2L);
    Assert.assertTrue(stats.hands() > 2);
    for (var item : completed) {
      Assert.assertTrue(item.episode().decisions().isEmpty());
      Assert.assertTrue(item.observedRoots().size() > 1);
      for (int i = 0; i < item.observedRoots().size(); i++) {
        Assert.assertEquals(item.observedRoots().get(i).publicPrefix().length, (i + 1) * 7);
      }
      double sum = Arrays.stream(item.episode().utilities()).sum();
      Assert.assertEquals(sum, 0.0, 1e-12);
      Assert.assertEquals(
          EpsilonGrpRanks.decode(EpsilonGrpRanks.encode(item.finalRanks())), item.finalRanks());
    }
  }

  @Test
  public void configuredUtilitiesPreserveRealizedRanksAndCalibrationLabels() {
    int[] expectedRanks = null;
    for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
      if (!profile.rankBased()) continue;
      List<CfrCollector.CollectedEpisode> completed = new ArrayList<>();
      configuredCollector(CfrCollector.Mode.MATCH_MC, false, profile)
          .collect(List.of(20L), UNIFORM, null, completed::add);
      var result = completed.getFirst();
      int[] ranks = result.finalRanks();
      if (expectedRanks == null) expectedRanks = ranks.clone();
      Assert.assertEquals(ranks, expectedRanks, "Changing utility must not change the game");
      Assert.assertEquals(EpsilonGrpRanks.decode(EpsilonGrpRanks.encode(ranks)), expectedRanks);
      for (int seat = 0; seat < 4; seat++)
        Assert.assertEquals(
            result.episode().utility(seat), profile.utilityForRankDouble(ranks[seat]));
      Arrays.fill(ranks, -1);
      Assert.assertEquals(result.finalRanks(), expectedRanks, "Published ranks must be immutable");
    }
  }

  @Test
  public void placementBoundaryUsesItsOwnRangeBeforeCentering() {
    BoundaryValue boundary =
        prefix ->
            prefix.length == 7
                ? new double[] {-1.5, .5, -.5, 1.5}
                : new double[] {1.5, .5, -.5, -1.5};
    List<CfrCollector.CollectedEpisode> completed = new ArrayList<>();
    configuredCollector(CfrCollector.Mode.HAND_GRP, true, EpsilonUtilityProfile.PLACEMENT)
        .collect(List.of(0L), UNIFORM, boundary, completed::add);
    Assert.assertEquals(completed.getFirst().episode().utilities(), new double[] {3, 0, 0, -3});
    Assert.assertNull(
        completed.getFirst().finalRanks(), "A GRP boundary has no realized final ranks");
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            configuredCollector(CfrCollector.Mode.HAND_GRP, true, EpsilonUtilityProfile.TENHOU)
                .collect(List.of(0L), UNIFORM, boundary));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> configuredCollector(CfrCollector.Mode.MATCH_MC, false, EpsilonUtilityProfile.SCORE));
  }

  @Test
  public void terminalUtilityUsesCommonRankDefinitionIncludingTies() {
    int[] scores = {20000, 30000, 20000, 30000};
    int[] ranks = {2, 0, 3, 1};
    for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
      if (!profile.rankBased()) continue;
      double[] values = CfrCollector.terminalUtilities(scores, profile);
      for (int seat = 0; seat < 4; seat++)
        Assert.assertEquals(values[seat], profile.utilityForRankDouble(ranks[seat]));
    }
  }

  @Test
  public void batchesFrozenBoundaryValuesWithoutChangingCenteredTrajectories() {
    List<Long> ids = List.of(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L);
    BoundaryValue scalar =
        prefix -> {
          double progress = prefix.length * .0001;
          return new double[] {.1 + progress, -.1, .2 - progress, -.2};
        };
    AtomicInteger largest = new AtomicInteger();
    AtomicInteger evaluated = new AtomicInteger();
    BoundaryValue batched =
        new BoundaryValue() {
          @Override
          public double[] evaluate(float[] prefix) {
            throw new AssertionError("Collection must use the batch boundary API");
          }

          @Override
          public List<double[]> evaluateBatch(List<float[]> prefixes) {
            largest.accumulateAndGet(prefixes.size(), Math::max);
            evaluated.addAndGet(prefixes.size());
            return prefixes.stream().map(scalar::evaluate).toList();
          }
        };
    var actual = collector(CfrCollector.Mode.HAND_GRP, 16, 8, true).collect(ids, UNIFORM, batched);
    var expected = collector(CfrCollector.Mode.HAND_GRP, 1, 1, true).collect(ids, UNIFORM, scalar);
    Assert.assertTrue(largest.get() > 0 && largest.get() <= 16);
    Assert.assertEquals(
        (long) evaluated.get(), 1 + actual.stats().hands() - actual.stats().games());
    Assert.assertEquals(actual.episodes().size(), expected.episodes().size());
    for (int episode = 0; episode < expected.episodes().size(); episode++)
      equalEpisode(actual.episodes().get(episode), expected.episodes().get(episode));

    BoundaryValue malformed =
        new BoundaryValue() {
          @Override
          public double[] evaluate(float[] prefix) {
            return new double[4];
          }

          @Override
          public List<double[]> evaluateBatch(List<float[]> prefixes) {
            return List.of();
          }
        };
    Assert.expectThrows(
        IllegalStateException.class,
        () -> collector(CfrCollector.Mode.HAND_GRP, 16, 8, true).collect(ids, UNIFORM, malformed));
  }

  @Test
  public void handFramesCoverTheSameWholeGameAndResetLocalReachAndCentering() {
    BoundaryValue boundary = prefix -> new double[] {prefix.length * .0001, 0, 0, 0};
    var frames = new ArrayList<CfrCollector.CollectedEpisode>();
    var stats =
        collector(CfrCollector.Mode.HAND_GRP, 8, 1, true)
            .collect(List.of(111L), UNIFORM, boundary, frames::add);
    var match =
        collector(CfrCollector.Mode.MATCH_MC, 8, 1, true)
            .collect(List.of(111L), UNIFORM, null)
            .episodes()
            .getFirst();
    Assert.assertEquals(stats.games(), 1L);
    Assert.assertEquals(stats.hands(), (long) frames.size());
    Assert.assertEquals(stats.trajectories(), stats.hands());
    Assert.assertTrue(frames.size() > 1);
    var combined = new ArrayList<CfrDecision>();
    var unique = new HashSet<CfrSampleId>();
    for (int index = 0; index < frames.size(); index++) {
      var frame = frames.get(index);
      boolean last = index == frames.size() - 1;
      Assert.assertEquals(frame.trajectoryIndex(), index);
      Assert.assertEquals(frame.gameEnded(), last);
      Assert.assertEquals(frame.observedRoots().size(), 1);
      Assert.assertEquals(frame.observedRoots().getFirst().roundOrdinal(), index + 1);
      double[] start = boundary.evaluate(frame.observedRoots().getFirst().publicPrefix());
      double[] end =
          last
              ? match.utilities()
              : boundary.evaluate(frames.get(index + 1).observedRoots().getFirst().publicPrefix());
      for (int seat = 0; seat < 4; seat++)
        Assert.assertEquals(frame.episode().utility(seat), end[seat] - start[seat], 1e-12);
      if (last) Assert.assertNotNull(frame.finalRanks());
      else Assert.assertNull(frame.finalRanks());
      var rows = LcfrTargets.generate(frame.episode(), 3, frames.size());
      Assert.assertEquals(
          rows.getFirst().logAverageWeight(), Math.log(3) - Math.log(frames.size()), 1e-12);
      for (var row : rows) Assert.assertTrue(unique.add(row.id()));
      combined.addAll(frame.episode().decisions());
    }
    equalEpisode(new CfrEpisode(111L, combined, match.utilities()), match);
  }

  @Test
  public void malformedPolicyFailsInsteadOfDroppingTrajectories() {
    CfrPolicy invalid =
        batch ->
            batch.stream()
                .map(
                    context ->
                        new Prediction(
                            new double[context.legalActions().size()],
                            ObservationPayload.copyOf(new byte[0])))
                .toList();
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            collector(CfrCollector.Mode.HAND_GRP, 8, 1, true)
                .collect(List.of(0L), invalid, prefix -> new double[4]));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () ->
            collector(CfrCollector.Mode.HAND_GRP, 8, 1, true)
                .collect(List.of(0L, 0L), UNIFORM, prefix -> new double[4]));
  }

  @Test
  public void gameIdsSelectIndependentWallsAndKeepRealRoundOrdinals() {
    Map<Long, ObservationPayload> first = new HashMap<>();
    Map<Long, Integer> ordinals = new HashMap<>();
    AtomicInteger laterRound = new AtomicInteger();
    CfrPolicy inspect =
        batch -> {
          for (DecisionContext context : batch) {
            if (context.decisionId() == 1)
              first.put(context.episodeId(), CfrObservationCodec.encodePayload(context));
            Assert.assertTrue(context.roundOrdinal() >= 1);
            int previous = ordinals.getOrDefault(context.episodeId(), 1);
            Assert.assertTrue(context.roundOrdinal() >= previous);
            ordinals.put(context.episodeId(), context.roundOrdinal());
            if (context.roundOrdinal() > 1) laterRound.incrementAndGet();
          }
          return UNIFORM.evaluate(batch);
        };
    new Collector(configuration(CfrCollector.Mode.MATCH_MC, 0, 8, 4, 103L, false, false))
        .collect(List.of(0L, 1L, 2L, 3L), inspect, null);
    Assert.assertEquals(first.size(), 4);
    for (long id = 1; id < 4; id++) Assert.assertNotEquals(first.get(id), first.get(0L));
    Assert.assertTrue(laterRound.get() > 0);
  }

  @Test
  public void failedBatchDrainsOtherBorrowedObservationsBeforeCollectionReturns() throws Exception {
    var waiting = new CopyOnWriteArrayList<CompletableFuture<List<Prediction>>>();
    CountDownLatch submitted = new CountDownLatch(4);
    CfrPolicy asynchronous =
        new CfrPolicy() {
          @Override
          public List<Prediction> evaluate(List<DecisionContext> batch) {
            return UNIFORM.evaluate(batch);
          }

          @Override
          public CompletableFuture<List<Prediction>> submitBatch(
              List<DecisionContext> batch, DecisionInferenceIngress ingress) {
            ingress.handoff(this).move();
            var result = new CompletableFuture<List<Prediction>>();
            waiting.add(result);
            submitted.countDown();
            return result;
          }
        };
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var collected =
          executor.submit(
              () ->
                  collector(CfrCollector.Mode.HAND_GRP, 1, 4, true)
                      .collect(List.of(0L, 1L, 2L, 3L), asynchronous, ignored -> new double[4]));
      try {
        Assert.assertTrue(
            submitted.await(10, TimeUnit.SECONDS), "All four batches must be submitted");
        waiting
            .getFirst()
            .completeExceptionally(new IllegalStateException("injected inference failure"));
        Assert.assertFalse(
            collected.isDone(),
            "Failed collection must retain borrowed observations until all work drains");
      } finally {
        for (var result : waiting)
          result.completeExceptionally(new IllegalStateException("drained"));
      }
      Assert.expectThrows(ExecutionException.class, () -> collected.get(10, TimeUnit.SECONDS));
      Assert.assertTrue(waiting.stream().allMatch(CompletableFuture::isDone));
    }
  }

  private static Collector collector(
      CfrCollector.Mode mode, int batch, int concurrent, boolean retain) {
    return new Collector(configuration(mode, 0.1, batch, concurrent, 901L, true, retain));
  }

  private static Collector configuredCollector(
      CfrCollector.Mode mode, boolean center, EpsilonUtilityProfile profile) {
    return new Collector(
        new CfrCollector.Config(
            mode,
            .1,
            8,
            1,
            901L,
            center,
            false,
            8,
            new DecisionDuelArenaSettings(1, 2, 64),
            new InferenceBatchingSettings(0),
            profile));
  }

  private static void equalEpisode(CfrEpisode left, CfrEpisode right) {
    Assert.assertEquals(left.episodeId(), right.episodeId());
    Assert.assertEquals(left.utilities(), right.utilities());
    Assert.assertEquals(left.decisions().size(), right.decisions().size());
    for (int i = 0; i < left.decisions().size(); i++) {
      CfrDecision a = left.decisions().get(i), b = right.decisions().get(i);
      Assert.assertEquals(a.decisionId(), b.decisionId());
      Assert.assertEquals(a.seat(), b.seat());
      Assert.assertEquals(a.legalActionIds(), b.legalActionIds());
      Assert.assertEquals(a.selectedSlot(), b.selectedSlot());
      Assert.assertEquals(a.targetPolicy(), b.targetPolicy());
      Assert.assertEquals(a.behaviorProbability(), b.behaviorProbability());
      Assert.assertEquals(a.observation(), b.observation());
    }
  }
}

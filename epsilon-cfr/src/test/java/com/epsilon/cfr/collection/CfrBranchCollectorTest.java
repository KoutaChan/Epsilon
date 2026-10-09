package com.epsilon.cfr.collection;

import static com.epsilon.cfr.collection.CfrCollectorTestSupport.configuration;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.cfr.algorithm.LcfrTargets;
import com.epsilon.cfr.collection.CfrCollectorTestSupport.Collector;
import com.epsilon.cfr.collection.CfrCollectorTestSupport.Result;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.Action;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class CfrBranchCollectorTest {
  // この牌山シードでは、エピソード0の意思決定1で九種九牌を宣言できる。
  private static final long KYUSHU_SEED = 641;
  private static final BoundaryValue BOUNDARY = ignored -> new double[] {.2, -.2, .1, -.1};

  @DataProvider
  public Object[][] choices() {
    return new Object[][] {{false}, {true}};
  }

  @DataProvider
  public Object[][] rankProfiles() {
    return Arrays.stream(EpsilonUtilityProfile.values())
        .filter(EpsilonUtilityProfile::rankBased)
        .map(profile -> new Object[] {profile})
        .toArray(Object[][]::new);
  }

  @Test(dataProvider = "choices", timeOut = 20_000)
  public void terminalBranchesPreserveMainTrajectoryAndAverageTargets(boolean declare) {
    var originalPolicy = new AuditPolicy(declare);
    var branchPolicy = new AuditPolicy(declare);
    var original = collect(false, originalPolicy);
    var branched = collect(true, branchPolicy);
    assertMainCollectionEquals(original, branched);
    CfrEpisode before = original.episodes().getFirst();
    CfrEpisode after = branched.episodes().getFirst();
    assertMainEpisodeEquals(before, after);
    Assert.assertTrue(before.branchOutcomes().isEmpty());
    Assert.assertEquals(after.branchOutcomes().size(), 1);
    Assert.assertEquals(branched.stats().games(), 1L);
    Assert.assertTrue(branched.episodes().size() > 1);
    Assert.assertEquals(branched.stats().hands(), (long) branched.episodes().size());
    Assert.assertEquals(branched.stats().trajectories(), branched.stats().hands());
    Assert.assertEquals(branched.stats().decisions(), original.stats().decisions());
    Assert.assertEquals(branched.stats().branchEpisodes(), 1L);
    Assert.assertEquals(branched.stats().branchComparisons(), 1L);
    Assert.assertEquals(branched.observedRoots().size(), original.observedRoots().size());

    CfrDecision first = after.decisions().getFirst();
    var comparison = after.branchOutcomes().getFirst();
    Assert.assertEquals(comparison.decisionId(), first.decisionId());
    Assert.assertEquals(
        first.legalActionIds()[first.selectedSlot()] == Action.kyushuKyuhai().toIndex(), declare);
    Assert.assertEquals(comparison.alternativeUtility(), .2);
    if (declare) {
      Assert.assertEquals(after.decisions().size(), 1);
      Assert.assertTrue(
          branched.stats().branchInferenceRows() > after.decisions().size(),
          "An admitted continuation must drain after exceeding its initial inference budget");
      Assert.assertEquals(branchPolicy.firstBranchObservations.size(), 4);
      Assert.assertEquals(branchPolicy.maximumObservedHand.get(-1L), Integer.valueOf(1));
      Assert.assertTrue(branchPolicy.maximumObservedHand.get(0L) > 1);
      for (int seat = 0; seat < 4; seat++) {
        var input = CfrObservationCodec.decode(branchPolicy.firstBranchObservations.get(seat));
        Assert.assertTrue(input.host().legalActionCount(0) > 0);
      }
    } else {
      Assert.assertEquals(comparison.alternativeSlot(), slot(first, Action.kyushuKyuhai()));
      Assert.assertEquals(branched.stats().branchInferenceRows(), 0L);
    }

    var beforeRows = LcfrTargets.generate(before, 3, 1);
    var afterRows = LcfrTargets.generate(after, 3, 1);
    Assert.assertEquals(afterRows.size(), beforeRows.size());
    for (int row = 0; row < beforeRows.size(); row++) {
      var expected = beforeRows.get(row);
      var actual = afterRows.get(row);
      Assert.assertEquals(actual.id(), expected.id());
      Assert.assertEquals(actual.targetPolicy(), expected.targetPolicy());
      Assert.assertEquals(actual.observation(), expected.observation());
      Assert.assertEquals(actual.logAverageWeight(), expected.logAverageWeight());
    }
    Assert.assertNotNull(afterRows.getFirst().explicitRegretTargets());
  }

  @Test(timeOut = 20_000)
  public void acceptedRonHasUnitInclusionEvenWhenNoAlternativeRolloutIsNeeded() {
    // 観測可能な情報だけを使うこの向聴数方策は、エピソード1の意思決定73でロンを選ぶ。
    var config = configuration(CfrCollector.Mode.HAND_GRP, .01, 16, 1, 17, false, true);
    var enabled = new DecisionBranchComparisonSettings(true, true, false, 1, 1, 1);
    var original = new Collector(config).collect(List.of(1L), winningPolicy(), BOUNDARY);
    var branched = new Collector(config, enabled).collect(List.of(1L), winningPolicy(), BOUNDARY);
    var before = original.episodes().getFirst();
    var after = branched.episodes().getFirst();
    assertMainCollectionEquals(original, branched);
    assertMainEpisodeEquals(before, after);
    Assert.assertEquals(after.branchOutcomes().size(), 1);
    var comparison = after.branchOutcomes().getFirst();
    var ron =
        after.decisions().stream()
            .filter(decision -> decision.decisionId() == comparison.decisionId())
            .findFirst()
            .orElseThrow();
    Assert.assertEquals(comparison.decisionId(), 73L);
    Assert.assertEquals(ron.legalActionIds()[ron.selectedSlot()], Action.ronAgari().toIndex());
    Assert.assertEquals(comparison.mainInclusionProbability(), 1d);
    Assert.assertTrue(ron.behaviorProbability() < 1);
    Assert.assertEquals(comparison.alternativeSlot(), -1);
    Assert.assertEquals(branched.stats().branchEpisodes(), 0L);
    Assert.assertEquals(branched.stats().branchInferenceRows(), 0L);
    Assert.assertEquals(branched.stats().branchComparisons(), 1L);
    var beforeRow = LcfrTargets.generate(before, 1, 1).getLast();
    var afterRow = LcfrTargets.generate(after, 1, 1).getLast();
    Assert.assertEquals(afterRow.id().decisionId(), comparison.decisionId());
    Assert.assertEquals(
        afterRow.logRegretWeight(),
        beforeRow.logRegretWeight() + Math.log(ron.behaviorProbability()),
        1e-12);
    Assert.assertEquals(afterRow.logAverageWeight(), beforeRow.logAverageWeight());
  }

  @Test(timeOut = 20_000)
  public void declinedRonAddsForcedAcceptanceWithoutChangingTheMainTrajectory() {
    var config = configuration(CfrCollector.Mode.HAND_GRP, .01, 16, 1, 17, false, true);
    var settings = new DecisionBranchComparisonSettings(true, true, false, 1, 1, 1);
    var original = new Collector(config).collect(List.of(1L), winningPolicy(true), BOUNDARY);
    var branched =
        new Collector(config, settings).collect(List.of(1L), winningPolicy(true), BOUNDARY);
    var before = original.episodes().getFirst();
    var after = branched.episodes().getFirst();
    assertMainCollectionEquals(original, branched);
    assertMainEpisodeEquals(before, after);
    Assert.assertEquals(after.branchOutcomes().size(), 1);
    var comparison = after.branchOutcomes().getFirst();
    var declined =
        after.decisions().stream()
            .filter(decision -> decision.decisionId() == comparison.decisionId())
            .findFirst()
            .orElseThrow();
    Assert.assertEquals(comparison.decisionId(), 73L);
    Assert.assertEquals(
        declined.legalActionIds()[declined.selectedSlot()], Action.pass().toIndex());
    Assert.assertEquals(comparison.alternativeSlot(), slot(declined, Action.ronAgari()));
    Assert.assertEquals(comparison.mainInclusionProbability(), declined.behaviorProbability());
    Assert.assertEquals(comparison.alternativeInclusionProbability(), 1d);
    Assert.assertEquals(
        comparison.alternativeUtility(), BOUNDARY.evaluate(new float[0])[declined.seat()]);
    Assert.assertEquals(branched.stats().games(), 1L);
    Assert.assertEquals(branched.stats().branchEpisodes(), 1L);
    Assert.assertEquals(branched.stats().branchComparisons(), 1L);
    Assert.assertEquals(branched.stats().branchInferenceRows(), 0L);
  }

  @Test(dataProvider = "rankProfiles", timeOut = 30_000)
  public void matchBranchesContinueToActualTerminalRanksWithoutCallingGrp(
      EpsilonUtilityProfile profile) {
    var config =
        new CfrCollector.Config(
            CfrCollector.Mode.MATCH_MC,
            .1,
            32,
            1,
            KYUSHU_SEED,
            false,
            true,
            32,
            new DecisionDuelArenaSettings(1, 2, 64),
            new InferenceBatchingSettings(0),
            profile);
    var settings = new DecisionBranchComparisonSettings(true, false, true, 1, 1, 1);
    BoundaryValue forbiddenGrp =
        ignored -> {
          throw new AssertionError("MATCH_MC must not call GRP for either trajectory");
        };
    var policy = new AuditPolicy(true, false);
    var original =
        new Collector(config).collect(List.of(0L), new AuditPolicy(true, false), forbiddenGrp);
    var branched = new Collector(config, settings).collect(List.of(0L), policy, forbiddenGrp);
    var before = original.episodes().getFirst();
    var after = branched.episodes().getFirst();
    assertMainEpisodeEquals(before, after);
    Assert.assertEquals(branched.stats().games(), 1L);
    Assert.assertEquals(branched.stats().branchEpisodes(), 1L);
    Assert.assertEquals(branched.stats().branchComparisons(), 1L);
    Assert.assertTrue(
        policy.maximumObservedRound.get(-1L) >= 1, "Branch must cross the initial round");
    double[] sortedUtilities = after.utilities();
    Arrays.sort(sortedUtilities);
    double[] expectedUtilities = profile.rankUtilityDouble();
    Arrays.sort(expectedUtilities);
    Assert.assertEquals(sortedUtilities, expectedUtilities);
    Assert.assertTrue(
        Arrays.stream(expectedUtilities)
            .anyMatch(value -> value == after.branchOutcomes().getFirst().alternativeUtility()));
    var beforeRows = LcfrTargets.generate(before, 3, 1);
    var afterRows = LcfrTargets.generate(after, 3, 1);
    Assert.assertEquals(afterRows.size(), beforeRows.size());
    for (int row = 0; row < beforeRows.size(); row++)
      Assert.assertEquals(
          afterRows.get(row).logAverageWeight(), beforeRows.get(row).logAverageWeight());
  }

  private static Result collect(boolean enabled, CfrPolicy policy) {
    var config = configuration(CfrCollector.Mode.HAND_GRP, .1, 16, 1, KYUSHU_SEED, false, true);
    var branches = new DecisionBranchComparisonSettings(enabled, false, true, 1, 1, 1);
    return new Collector(config, branches).collect(List.of(0L), policy, BOUNDARY);
  }

  private static CfrPolicy winningPolicy() {
    return winningPolicy(false);
  }

  private static CfrPolicy winningPolicy(boolean declineFirstRon) {
    HandShapeAnalyzer analyzer = new HandShapeAnalyzer();
    return batch ->
        batch.stream()
            .map(
                context -> {
                  List<Action> actions = context.legalActions();
                  int best = 0, score = Integer.MAX_VALUE;
                  for (int slot = 0; slot < actions.size(); slot++) {
                    Action action = actions.get(slot);
                    int candidate =
                        switch (action.type()) {
                          case RON_AGARI, TSUMO_AGARI -> -100;
                          case RIICHI_DAHAI -> -50;
                          case PASS -> declineFirstRon && context.decisionId() == 73 ? -200 : -40;
                          case DAHAI ->
                              analyzer.calculateMinimumAfterRemoving(
                                  context.observation().hand(context.seat()), action.tileType());
                          default -> 100;
                        };
                    if (candidate < score) {
                      score = candidate;
                      best = slot;
                    }
                  }
                  double[] sigma = new double[actions.size()];
                  sigma[best] = 1;
                  return new Prediction(sigma, ObservationPayload.copyOf(new byte[0]));
                })
            .toList();
  }

  private static void assertMainEpisodeEquals(CfrEpisode expected, CfrEpisode actual) {
    Assert.assertEquals(actual.episodeId(), expected.episodeId());
    Assert.assertEquals(actual.utilities(), expected.utilities());
    Assert.assertEquals(actual.decisions().size(), expected.decisions().size());
    for (int row = 0; row < expected.decisions().size(); row++) {
      CfrDecision before = expected.decisions().get(row), after = actual.decisions().get(row);
      Assert.assertEquals(after.decisionId(), before.decisionId());
      Assert.assertEquals(after.seat(), before.seat());
      Assert.assertEquals(after.legalActionIds(), before.legalActionIds());
      Assert.assertEquals(after.selectedSlot(), before.selectedSlot());
      Assert.assertEquals(after.targetPolicy(), before.targetPolicy());
      Assert.assertEquals(after.behaviorProbability(), before.behaviorProbability());
      Assert.assertEquals(after.observation(), before.observation());
    }
  }

  private static void assertMainCollectionEquals(Result expected, Result actual) {
    Assert.assertEquals(actual.episodes().size(), expected.episodes().size());
    for (int index = 0; index < actual.episodes().size(); index++)
      assertMainEpisodeEquals(expected.episodes().get(index), actual.episodes().get(index));
  }

  private static int slot(CfrDecision decision, Action action) {
    int[] ids = decision.legalActionIds();
    for (int slot = 0; slot < ids.length; slot++) if (ids[slot] == action.toIndex()) return slot;
    throw new AssertionError("Missing action " + action);
  }

  private static final class AuditPolicy implements CfrPolicy {
    private final boolean declare;
    private final boolean encodeObservation;
    private final Map<Integer, ObservationPayload> firstBranchObservations = new HashMap<>();
    private final Map<Long, Integer> maximumObservedRound = new HashMap<>();
    private final Map<Long, Integer> maximumObservedHand = new HashMap<>();

    private AuditPolicy(boolean declare) {
      this(declare, true);
    }

    private AuditPolicy(boolean declare, boolean encodeObservation) {
      this.declare = declare;
      this.encodeObservation = encodeObservation;
    }

    @Override
    public List<Prediction> evaluate(List<DecisionContext> batch) {
      List<Prediction> predictions = new ArrayList<>(batch.size());
      for (DecisionContext context : batch) {
        Assert.assertEquals(context.observation().observer(), context.seat());
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> context.observation().hand((context.seat() + 1) % 4).count(0));
        maximumObservedRound.merge(
            context.episodeId(), context.observation().roundIndex(), Math::max);
        maximumObservedHand.merge(context.episodeId(), context.roundOrdinal(), Math::max);
        double[] sigma = new double[context.legalActions().size()];
        int kyushu = context.legalActions().indexOf(Action.kyushuKyuhai());
        if (kyushu >= 0) {
          double probability = declare ? .99 : .01;
          Arrays.fill(sigma, (1 - probability) / (sigma.length - 1));
          sigma[kyushu] = probability;
        } else {
          Arrays.fill(sigma, 1d / sigma.length);
        }
        ObservationPayload observation =
            encodeObservation
                ? CfrObservationCodec.encodePayload(context)
                : ObservationPayload.copyOf(new byte[0]);
        if (context.episodeId() < 0)
          firstBranchObservations.putIfAbsent(context.seat(), observation);
        predictions.add(new Prediction(sigma, observation));
      }
      return predictions;
    }
  }
}

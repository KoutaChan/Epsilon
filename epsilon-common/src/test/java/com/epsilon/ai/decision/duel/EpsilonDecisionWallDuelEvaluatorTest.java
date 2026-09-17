package com.epsilon.ai.decision.duel;

import com.epsilon.ai.decision.EpsilonDecisionSeeds;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator.Decision;
import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator.EvaluationPlan;
import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator.Mode;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.runtime.InferenceAdmission;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 同期入力元の区間実行、判定時点の停止条件、観測と記述統計の集約を検証する。 */
public class EpsilonDecisionWallDuelEvaluatorTest {
  private static final Path CANDIDATE = Path.of("candidate");
  private static final Path OPPONENT = Path.of("opponent");
  private static final long SEED_BASE = 1234L;
  private static final long DUEL_SEQUENCE = 7L;
  private static final DecisionEvalVsSettings SETTINGS =
      new DecisionEvalVsSettings(12, 16, 3, 999L);

  @Test
  public void fixedHorizonConsumesAllWallsDespitePromisingPrefix() {
    FakeSource source = new FakeSource(segment(1, 1, 1, -1, -1, -1));
    var result = evaluate(source, EvaluationPlan.fixed(6, 0.05));

    Assert.assertEquals(source.calls, List.of(new Call(6, duelSeed(), 0, 12)));
    Assert.assertEquals(result.games(), 24);
    Assert.assertEquals(result.wallSeeds(), 6);
    Assert.assertEquals(result.completedLooks(), 1);
    Assert.assertEquals(result.progressUpdates(), 2);
    Assert.assertEquals(result.mode(), Mode.FIXED_HORIZON);
    Assert.assertEquals(result.decision(), Decision.UNRESOLVED);
    Assert.assertEquals(result.pairedUtilityDeltaMean(), 0.0, 1.0e-12);
  }

  @DataProvider
  public Object[][] earlyDecisions() {
    return new Object[][] {
      {0.5, Decision.PROMOTED},
      {-0.5, Decision.HARMFUL}
    };
  }

  @Test(dataProvider = "earlyDecisions")
  public void decisiveFirstLookDoesNotRequestRemainingWalls(double utility, Decision expected) {
    FakeSource source = new FakeSource(segment(utility, utility));
    var result = evaluate(source, EvaluationPlan.twoLook(2, 7, 0.05, 0.25));

    Assert.assertEquals(source.calls, List.of(new Call(2, duelSeed(), 0, 12)));
    Assert.assertEquals(result.games(), 8);
    Assert.assertEquals(result.wallSeeds(), 2);
    Assert.assertEquals(result.plannedWallSeeds(), 7);
    Assert.assertEquals(result.completedLooks(), 1);
    Assert.assertEquals(result.mode(), Mode.TWO_LOOK_ALPHA_SPENDING);
    Assert.assertEquals(result.decision(), expected);
    Assert.assertEquals(result.alpha(), 0.05);
    Assert.assertEquals(result.decisionAlpha(), 0.0125);
  }

  @Test
  public void unresolvedLookContinuesSameSeedAndWeightsUnequalSegmentsByGames() {
    Segment first =
        new Segment(
            new double[] {0, 0},
            -1.0,
            new DuelEvaluation.Result(
                CANDIDATE,
                OPPONENT,
                8,
                2,
                0,
                0,
                0,
                100,
                1,
                4,
                1,
                0,
                0,
                1,
                new double[] {1, 0.5, 0, 0, 0, 0}));
    Segment second =
        new Segment(
            new double[] {0.5, 0.5, 0.5},
            1.0,
            new DuelEvaluation.Result(
                CANDIDATE,
                OPPONENT,
                12,
                3,
                0,
                0,
                0,
                -50,
                3,
                2,
                0,
                1,
                1,
                0,
                new double[] {-1, 0, 0, 0, 0, 0}));
    FakeSource source = new FakeSource(first, second);
    var result = evaluate(source, EvaluationPlan.twoLook(2, 5, 0.05, 0.25));

    Assert.assertEquals(
        source.calls, List.of(new Call(2, duelSeed(), 0, 12), new Call(3, duelSeed(), 2, 12)));
    Assert.assertEquals(result.candidateCheckpoint(), CANDIDATE);
    Assert.assertEquals(result.parentCheckpoint(), OPPONENT);
    Assert.assertEquals(result.games(), 20);
    Assert.assertEquals(result.wallSeeds(), 5);
    Assert.assertEquals(result.completedLooks(), 2);
    Assert.assertEquals(result.progressUpdates(), 3);
    Assert.assertEquals(result.decisionAlpha(), 0.0375, 1.0e-15);
    Assert.assertEquals(result.decision(), Decision.PROMOTED);
    Assert.assertEquals(result.pairedUtilityDeltaMean(), 0.3, 1.0e-12);
    Assert.assertEquals(result.pairedRankDeltaMean(), 0.2, 1.0e-12);

    var descriptive = result.descriptiveResult();
    Assert.assertEquals(descriptive.scoreAdvantage(), 10.0, 1.0e-12);
    Assert.assertEquals(descriptive.candidateAverageRank(), 2.2, 1.0e-12);
    Assert.assertEquals(descriptive.opponentAverageRank(), 2.8, 1.0e-12);
    Assert.assertEquals(descriptive.candidateTopRate(), 0.4, 1.0e-12);
    Assert.assertEquals(descriptive.opponentTopRate(), 0.6, 1.0e-12);
    Assert.assertEquals(descriptive.candidateLastRate(), 0.6, 1.0e-12);
    Assert.assertEquals(descriptive.opponentLastRate(), 0.4, 1.0e-12);
    Assert.assertEquals(descriptive.utilityProfileAdvantages()[0], -0.2, 1.0e-12);
    Assert.assertEquals(descriptive.utilityProfileAdvantages()[1], 0.2, 1.0e-12);
    // [-1, -1, 1, 1, 1]の牌山標本から再計算し、区間別SEの平均にはしない。
    Assert.assertEquals(descriptive.pairedRankDeltaSe(), 0.4898979485566356, 1.0e-12);
    Assert.assertEquals(descriptive.pairedRankDeltaLcb(), -0.7601823352710619, 1.0e-12);
  }

  @DataProvider
  public Object[][] decisionBoundaries() {
    return new Object[][] {
      {0.25, 0.75, 0.25, 0.0, Decision.UNRESOLVED},
      {-0.75, -0.25, 0.0, -0.25, Decision.UNRESOLVED},
      {Math.nextUp(0.25), 0.75, 0.25, 0.0, Decision.UNRESOLVED},
      {-0.75, Math.nextDown(-0.25), 0.0, -0.25, Decision.UNRESOLVED},
      {0.25000001, 0.75, 0.25, 0.0, Decision.PROMOTED},
      {-0.75, -0.25000001, 0.0, -0.25, Decision.HARMFUL}
    };
  }

  @Test(dataProvider = "decisionBoundaries")
  public void onlySeparationBeyondBoundaryRoundingDecides(
      double lower, double upper, double promotionMargin, double harmfulMargin, Decision expected) {
    var estimate =
        new EpsilonDecisionFixedSampleConfidenceInterval.Estimate(
            10, (lower + upper) / 2.0, 0.1, lower, upper);
    Assert.assertEquals(
        EpsilonDecisionWallDuelEvaluator.decide(estimate, promotionMargin, harmfulMargin),
        expected);
  }

  @Test
  public void sourceFailureAfterAnObservationPropagatesWithoutReturningPartialResult() {
    IllegalStateException failure = new IllegalStateException("source failed after first wall");
    DuelEvaluationSource source =
        new DuelEvaluationSource() {
          @Override
          public Path candidateCheckpoint() {
            return CANDIDATE;
          }

          @Override
          public Path opponentCheckpoint() {
            return OPPONENT;
          }

          @Override
          public DuelEvaluation evaluateWalls(
              int wallSeeds,
              long seedBase,
              long firstWallFamilyId,
              int gamesInFlight,
              Consumer<DuelEvaluation.WallOutcome> sink) {
            sink.accept(new DuelEvaluation.WallOutcome(firstWallFamilyId, seedBase, 0, 0));
            throw failure;
          }
        };
    IllegalStateException actual =
        Assert.expectThrows(
            IllegalStateException.class, () -> evaluate(source, EvaluationPlan.fixed(3, 0.05)));
    Assert.assertSame(actual, failure);
  }

  private static EpsilonDecisionWallDuelEvaluator.Result evaluate(
      DuelEvaluationSource source, EvaluationPlan plan) {
    return EpsilonDecisionWallDuelEvaluator.evaluate(
        source,
        plan,
        SEED_BASE,
        DUEL_SEQUENCE,
        0.0,
        0.0,
        EpsilonUtilityProfile.PLACEMENT,
        SETTINGS);
  }

  private static long duelSeed() {
    return EpsilonDecisionSeeds.promotionAttempt(SEED_BASE, DUEL_SEQUENCE);
  }

  private static Segment segment(double... utilities) {
    return new Segment(
        utilities,
        0.0,
        new DuelEvaluation.Result(
            CANDIDATE,
            OPPONENT,
            utilities.length * 4,
            utilities.length,
            0,
            0,
            0,
            0,
            2.5,
            2.5,
            0.25,
            0.25,
            0.25,
            0.25,
            new double[EpsilonUtilityProfile.values().length]));
  }

  private record Segment(double[] utilities, double rankDelta, DuelEvaluation.Result result) {}

  private record Call(int walls, long seed, long firstWall, int gamesInFlight) {}

  private static final class FakeSource implements DuelEvaluationSource {
    private final Segment[] segments;
    private final List<Call> calls = new ArrayList<>();

    private FakeSource(Segment... segments) {
      this.segments = segments;
    }

    @Override
    public Path candidateCheckpoint() {
      return CANDIDATE;
    }

    @Override
    public Path opponentCheckpoint() {
      return OPPONENT;
    }

    @Override
    public DuelEvaluation evaluateWalls(
        int wallSeeds,
        long seedBase,
        long firstWallFamilyId,
        int gamesInFlight,
        Consumer<DuelEvaluation.WallOutcome> sink) {
      Segment segment = segments[calls.size()];
      calls.add(new Call(wallSeeds, seedBase, firstWallFamilyId, gamesInFlight));
      for (int i = 0; i < segment.utilities().length; i++) {
        long wall = firstWallFamilyId + i;
        sink.accept(
            new DuelEvaluation.WallOutcome(
                wall,
                EpsilonDecisionSeeds.evalVsGame(seedBase, wall),
                segment.rankDelta(),
                segment.utilities()[i]));
      }
      int games = segment.result().games();
      return new DuelEvaluation(
          segment.result(),
          new DuelEvaluation.Metrics(
              games, gamesInFlight, games, 0, 0, 0, 0, InferenceAdmission.Metrics.empty()));
    }
  }
}

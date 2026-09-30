package com.epsilon.major.ai.decision.arena;

import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.ai.decision.EpsilonDecisionSeeds;
import com.epsilon.ai.grp.EpsilonGrpRankPredictor;
import com.epsilon.core.Action;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.major.ai.decision.data.BranchFragmentAssertions;
import com.epsilon.major.ai.decision.data.EpsilonDecisionCompletedGame;
import com.epsilon.major.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionInferenceServer.Prediction;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 実エンジンとCPU評価器で分岐収集を通し、元対局とValue教師を変えないことを確認する。 */
public class DecisionBranchArenaTest {
  @Test(timeOut = 120_000)
  public void queuedMainRowsDoNotStarveBranchInference() throws Exception {
    long seed = kyushuSeed();
    var evaluator = new SingleSlotEvaluator();
    var disabled = collect(seed, false, new SingleSlotEvaluator(), 16);
    var games = collect(seed, true, evaluator, 16);
    Assert.assertTrue(evaluator.blockedAttempts > 0, "Inference ingress must be congested");
    for (int index = 0; index < games.size(); index++) {
      assertMainTrajectoryUnchanged(disabled.get(index), games.get(index));
    }
    Assert.assertTrue(
        games.getFirst().samples().stream().anyMatch(sample -> sample.branchTarget().complete()),
        "The first game's KYUSHU continuation must finish while other games keep inference busy");
  }

  private static EpsilonDecisionEvaluator evaluator(boolean declare) {
    return batch -> {
      List<Prediction> results = new ArrayList<>();
      for (int row = 0; row < batch.size(); row++) {
        float[] logits = new float[batch.legalActionCount(row)];
        for (int slot = 0; slot < logits.length; slot++) {
          Action action = Action.fromIndex(batch.legalActionId(row, slot));
          logits[slot] = action.type().group() == Action.Group.KYUSHU ? (declare ? 8 : -8) : 0;
        }
        results.add(new Prediction(logits, 0));
      }
      return results;
    };
  }

  private static long kyushuSeed() {
    for (long seed = 1; seed < 50_000; seed++) {
      GameEngine engine = new GameEngine(EpsilonDecisionSeeds.trainGame(seed, 0));
      var decisions = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
      for (var decision : decisions.decisions()) {
        if (decision.player() == 0 && decision.legalActions().contains(Action.kyushuKyuhai()))
          return seed;
      }
    }
    throw new AssertionError("No KYUSHU fixture found");
  }

  @DataProvider
  public Object[][] choices() {
    return new Object[][] {{false}, {true}};
  }

  @Test(timeOut = 120_000, dataProvider = "choices")
  public void enabledComparisonPreservesMainTrajectoryAndDisabledModeHasNoTargets(boolean declare)
      throws Exception {
    long seed = kyushuSeed();
    var disabled = collect(seed, false, declare);
    var enabled = collect(seed, true, declare);
    Assert.assertEquals(
        assertMainTrajectoryUnchanged(disabled, enabled),
        1,
        "Exactly one admitted, completed KYUSHU pair is required");
  }

  private static int assertMainTrajectoryUnchanged(
      EpsilonDecisionCompletedGame disabled, EpsilonDecisionCompletedGame enabled) {
    Assert.assertEquals(enabled.finalRanksCode(), disabled.finalRanksCode());
    Assert.assertEquals(enabled.samples().size(), disabled.samples().size());
    int complete = 0;
    for (int i = 0; i < enabled.samples().size(); i++) {
      var original = disabled.samples().get(i);
      var compared = enabled.samples().get(i);
      Assert.assertEquals(original.branchTarget(), DecisionBranchTarget.NONE);
      Assert.assertEquals(compared.chosenActionId(), original.chosenActionId());
      Assert.assertEquals(compared.behaviorLogProb(), original.behaviorLogProb());
      Assert.assertEquals(compared.valueTarget(), original.valueTarget());
      Assert.assertEquals(compared.advantage(), original.advantage());
      if (compared.branchTarget().complete()) {
        complete++;
        Assert.assertEquals(compared.branchTarget().code(), 3);
        Assert.assertEquals(compared.branchTarget().acceptedUtility(), 0f, 1e-6f);
        Assert.assertEquals(compared.branchTarget().declinedUtility(), 0f, 1e-6f);
      }
    }
    return complete;
  }

  private static EpsilonDecisionCompletedGame collect(long seed, boolean enabled, boolean declare)
      throws Exception {
    return collect(seed, enabled, evaluator(declare), 1).getFirst();
  }

  private static List<EpsilonDecisionCompletedGame> collect(
      long seed, boolean enabled, EpsilonDecisionEvaluator evaluator, int games) throws Exception {
    var config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.decision.rollout.branchComparison.enabled", Boolean.toString(enabled),
                "epsilon.decision.rollout.branchComparison.maxExtraInferenceRatio", "1",
                "epsilon.decision.rollout.branchComparison.maxInFlightBranches", "1",
                "epsilon.decision.inference.maxBatchWaitMicros", "0",
                "epsilon.decision.train.arena.advanceWorkers", "1"));
    var dir = Files.createTempDirectory("epsilon-branch-test-");
    var results = new ArrayList<EpsilonDecisionCompletedGame>();
    for (int index = 0; index < games; index++) results.add(null);
    EpsilonGrpRankPredictor grp =
        (sequence, seat) -> CompletableFuture.completedFuture(new float[] {.25f, .25f, .25f, .25f});
    try (var store = EpsilonDecisionTrajectoryPayloadStore.asyncFileBacked(dir, 32, 1)) {
      var metrics =
          EpsilonDecisionArena.collectPopulationGamesWithRankPredictor(
              games,
              seed,
              games,
              evaluator,
              1,
              new long[][] {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}},
              ignored -> evaluator,
              EpsilonDecisionPlayer.RolloutConfig.fromSettings(config),
              EpsilonDecisionPlayer.RolloutConfig.opponentFromSettings(config),
              (index, gameSeed, game) -> results.set(index, game),
              store,
              grp,
              null,
              EpsilonDecisionArena.GameIndexPolicy.standard(),
              config);
      Assert.assertEquals(metrics.games(), games, "Additional branches must not count as games");
      BranchFragmentAssertions.verify(results.getFirst());
      return results;
    } finally {
      try (var paths = Files.list(dir)) {
        for (var file : paths.toList()) {
          Files.delete(file);
        }
      }
      Files.delete(dir);
    }
  }

  /** 常に一行・一枠だけを受け付け、残りの通常対局を推論キューへ滞留させる。 */
  private static final class SingleSlotEvaluator implements EpsilonDecisionEvaluator {
    private final EpsilonDecisionEvaluator delegate = evaluator(true);
    private boolean occupied;
    private CompletableFuture<Void> available = new CompletableFuture<>();
    private int blockedAttempts;

    @Override
    public int preferredStreamingBatchSize() {
      return 1;
    }

    @Override
    public synchronized DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
      if (occupied) {
        blockedAttempts++;
        return DecisionInferenceIngress.Attempt.blocked(available);
      }
      occupied = true;
      return DecisionInferenceIngress.Attempt.acquired(
          new DecisionInferenceIngress(this, null, this::release));
    }

    private synchronized void release() {
      occupied = false;
      var signal = available;
      available = new CompletableFuture<>();
      signal.complete(null);
    }

    @Override
    public List<Prediction> evaluateBatch(DecisionHostBatch batch) {
      return delegate.evaluateBatch(batch);
    }
  }
}

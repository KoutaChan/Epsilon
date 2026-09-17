package com.epsilon.pico.ai.decision.arena;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRankPredictor;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.engine.EngineDecisionSelection;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.pico.config.settings.EpsilonSettings;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 局の切り替わりで、行動選択専用のプレイヤーは GRP を計算せず、学習側だけが必要な予測を保持・再利用することを検証する。 */
public class DecisionPlayerBoundaryTest {
  private static final EpsilonDecisionEvaluator UNUSED_EVALUATOR =
      batch -> {
        throw new AssertionError("Boundary preparation does not need Decision inference");
      };

  @Test
  public void actionOnlyPlayerDoesNotPrepareGrpAfterRoundSettlement() {
    EpsilonGrpRankPredictor unusedGrp =
        (sequence, seat) -> {
          throw new AssertionError("Action-only player does not need GRP inference");
        };
    EpsilonDecisionPlayer opponent =
        EpsilonDecisionPlayer.builder(UNUSED_EVALUATOR, settings()).grpInference(unusedGrp).build();
    GameEngine engine = new GameEngine(193);
    opponent.onRoundSettled(nextSettlement(engine));
    Assert.assertSame(
        opponent.prepareBoundaryContextAsync(engine.getState()).join(),
        DecisionBoundaryContext.uniform());
  }

  @Test
  public void actorKeepsBoundaryPrefixesAndReusesPreparedContextAcrossRounds() {
    List<float[]> prefixes = new ArrayList<>();
    EpsilonGrpRankPredictor grp =
        (sequence, seat) -> {
          if (seat == 0) prefixes.add(sequence);
          return CompletableFuture.completedFuture(new float[] {0.25f, 0.25f, 0.25f, 0.25f});
        };
    SettingsLoader config = settings();
    EpsilonDecisionPlayer actor =
        EpsilonDecisionPlayer.trainingRollout(
            UNUSED_EVALUATOR,
            1,
            271,
            EpsilonDecisionPlayer.RolloutConfig.fromSettings(config),
            null,
            grp,
            null,
            config);
    GameEngine engine = new GameEngine(193);
    actor.prepareBoundaryContextAsync(engine.getState()).join();
    Assert.assertEquals(prefixes.size(), 1);
    Assert.assertEquals(prefixes.getFirst(), EpsilonGrpFeature.fromState(engine.getState()));
    for (int boundary = 1; boundary <= 2; boundary++) {
      RoundSettlement settlement = nextSettlement(engine);
      var next = (RoundTransition.NextRound) settlement.transition();
      actor.onRoundSettled(settlement);
      Assert.assertEquals(prefixes.size(), boundary + 1);
      float[] prefix = prefixes.getLast();
      float[] previous = prefixes.get(boundary - 1);
      Assert.assertEquals(prefix.length, previous.length + EpsilonGrpFeature.FEATURE_SIZE);
      Assert.assertEquals(Arrays.copyOf(prefix, previous.length), previous);
      float[] nextFeatures =
          EpsilonGrpFeature.fromProgress(
              next.kyokuIndex(),
              next.honba(),
              next.kyotakuCount(),
              settlement.snapshotFinalScores());
      Assert.assertEquals(Arrays.copyOfRange(prefix, previous.length, prefix.length), nextFeatures);
      engine.stepHanchan();
      CompletableFuture<DecisionBoundaryContext> prepared =
          actor.prepareBoundaryContextAsync(engine.getState());
      Assert.assertSame(actor.prepareBoundaryContextAsync(engine.getState()), prepared);
      Assert.assertEquals(prefixes.size(), boundary + 1);
      Assert.assertEquals(prepared.join().publicFeatures(), nextFeatures);
      Assert.assertTrue(prepared.join().present());
    }
  }

  private static SettingsLoader settings() {
    return EpsilonSettings.of(Map.of("epsilon.decision.utilityProfile", "TOP"));
  }

  private static RoundSettlement nextSettlement(GameEngine engine) {
    GameStepResult step = engine.stepHanchan();
    while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
      List<EngineDecisionSelection> selections = new ArrayList<>();
      for (var decision : awaiting.decisions()) {
        selections.add(
            new EngineDecisionSelection(decision.id(), decision.legalActions().getFirst()));
      }
      step = engine.commitDecisions(selections);
    }
    return ((GameStepResult.RoundSettled) step).settlement();
  }
}

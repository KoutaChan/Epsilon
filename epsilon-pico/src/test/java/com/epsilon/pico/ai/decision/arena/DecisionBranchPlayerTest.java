package com.epsilon.pico.ai.decision.arena;

import com.epsilon.ai.decision.DecisionBranchGate;
import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionInferenceServer.Prediction;
import com.epsilon.pico.config.settings.EpsilonSettings;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DecisionBranchPlayerTest {
  @Test
  public void winComparisonRequiresExplorationAttributionNotMerelyDeclining() throws Exception {
    var noExploration = selections(0, false);
    var exploration = selections(.5f, false);
    Assert.assertEquals(
        exploration.actions(),
        noExploration.actions(),
        "Attribution must not consume the main RNG");
    Assert.assertEquals(noExploration.comparisons(), 0);
    Assert.assertTrue(exploration.comparisons() > 0);
    Assert.assertTrue(exploration.comparisons() < 128);
  }

  @Test
  public void kyushuTakesPriorityWhenBothDeclarationsAreLegal() throws Exception {
    Assert.assertTrue(selections(.5f, true).comparisons() > 0);
  }

  private record Selection(int[] actions, int comparisons) {}

  private static Selection selections(float mass, boolean withKyushu) throws Exception {
    var config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.decision.rollout.branchComparison.enabled", "true",
                "epsilon.decision.rollout.fullSupport.adaptiveExploration.schedule", "NORMAL",
                "epsilon.decision.rollout.branchComparison.winDecline", "true",
                "epsilon.decision.rollout.fullSupport.terminalGateExplorationMass",
                    Float.toString(mass)));
    GameState state = new GameState(7);
    state.startRound(0, 0, 0, 0);
    state.dealInitialHands();
    state.hand(0).clear();
    for (int tile :
        new int[] {
          Tile.M1, Tile.M2, Tile.M3, Tile.P1, Tile.P2, Tile.P3, Tile.S1, Tile.S2, Tile.S3, Tile.TON,
          Tile.TON, Tile.TON, Tile.NAN, Tile.NAN
        }) {
      state.hand(0).add(tile);
    }
    state.recordDraw(0, Tile.NAN * 4, TurnEvent.DrawSource.WALL, false);
    var actions =
        withKyushu
            ? List.of(Action.tsumoAgari(), Action.kyushuKyuhai(), Action.dahai(Tile.NAN))
            : List.of(Action.tsumoAgari(), Action.dahai(Tile.NAN));
    EpsilonDecisionEvaluator evaluator =
        batch -> List.of(new Prediction(new float[batch.legalActionCount(0)], 0));
    var dir = Files.createTempDirectory("epsilon-branch-selection-");
    try (var store = EpsilonDecisionTrajectoryPayloadStore.asyncFileBacked(dir, 32, 1)) {
      var player =
          EpsilonDecisionPlayer.trainingRollout(
              evaluator,
              1,
              271,
              EpsilonDecisionPlayer.RolloutConfig.fromSettings(config),
              store,
              null,
              null,
              config);
      int[] chosen = new int[128];
      int comparisons = 0;
      for (int i = 0; i < chosen.length; i++) {
        Action selected = player.selectAction(state, 0, actions);
        chosen[i] = selected.toIndex();
        var candidate = player.takeBranchCandidate();
        if (candidate != null) {
          comparisons++;
          if (withKyushu) {
            Assert.assertEquals(candidate.gate(), DecisionBranchGate.KYUSHU);
            Assert.assertNotEquals(selected, Action.tsumoAgari());
            Assert.assertNotEquals(candidate.alternative(), Action.tsumoAgari());
          } else {
            Assert.assertEquals(selected.type(), Action.Type.DAHAI);
            Assert.assertEquals(candidate.alternative(), Action.tsumoAgari());
          }
        }
      }
      if (withKyushu) {
        var game = player.flushTrajectoryDeferred(1, new int[] {25000, 25000, 25000, 25000});
        for (var sample : game.samples())
          Assert.assertEquals(sample.branchTarget(), DecisionBranchTarget.KYUSHU_ONLY);
      }
      return new Selection(chosen, comparisons);
    } finally {
      try (var paths = Files.list(dir)) {
        for (var file : paths.toList()) {
          Files.delete(file);
        }
      }
      Files.delete(dir);
    }
  }
}

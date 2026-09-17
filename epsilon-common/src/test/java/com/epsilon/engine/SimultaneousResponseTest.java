package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 同時ロンによる未成立の立直宣言の取消しと、三家和による途中流局を検証する。 */
public class SimultaneousResponseTest {
  @DataProvider(name = "winners")
  public Object[][] winners() {
    return new Object[][] {{1}, {2}, {3}};
  }

  @Test(dataProvider = "winners")
  public void simultaneousRonCancelsPendingRiichiAndThreeWinnersAbort(int winners) {
    GameEngine engine = new GameEngine(19);
    GameState state = engine.getState();
    state.startRound(0, 0, 0, 0);
    MahjongFixtures.hand(state, 0, "123m123p123s111z5z");
    for (int seat = 1; seat < 4; seat++) MahjongFixtures.hand(state, seat, "123m123p123s11z55z");
    int[] wall = java.util.stream.IntStream.range(0, 136).toArray();
    wall[52] = 0;
    wall[0] = 52;
    state.initializeWall(wall);
    var turn = (GameStepResult.AwaitingDecisions) engine.stepRound();
    var decision = turn.decisions().getFirst();
    Action declaration =
        decision.legalActions().stream()
            .filter(a -> a.type() == Action.Type.RIICHI_DAHAI && a.tileType() == Tile.HAKU)
            .findFirst()
            .orElseThrow();
    var responses =
        (GameStepResult.AwaitingDecisions) engine.commitDecision(decision.id(), declaration);
    Assert.assertEquals(responses.decisions().size(), 3);
    List<EngineDecisionSelection> choices = new ArrayList<>();
    for (var response : responses.decisions()) {
      Action.Type type = response.player() <= winners ? Action.Type.RON_AGARI : Action.Type.PASS;
      var action =
          response.legalActions().stream().filter(a -> a.type() == type).findFirst().orElseThrow();
      choices.add(new EngineDecisionSelection(response.id(), action));
    }
    // 同時応答のリスト順で優先順位や精算を変えない。
    java.util.Collections.reverse(choices);
    var ended = (GameStepResult.RoundEnded) engine.commitDecisions(choices);
    Assert.assertFalse(state.isRiichi(0));
    Assert.assertEquals(state.getKyotakuCount(), 0);
    int[] deltas = ended.result().pointDelta().toArray();
    Assert.assertEquals(java.util.Arrays.stream(deltas).sum(), 0);
    if (winners == 3) {
      Assert.assertEquals(
          ((RoundResult.AbortiveDraw) ended.result()).reason(),
          RoundResult.AbortiveDrawReason.TRIPLE_RON);
      Assert.assertEquals(deltas, new int[4]);
    } else {
      Assert.assertTrue(deltas[0] < 0);
      for (int seat = 1; seat < 4; seat++) {
        if (seat <= winners) Assert.assertTrue(deltas[seat] > 0);
        else Assert.assertEquals(deltas[seat], 0);
      }
    }
  }
}

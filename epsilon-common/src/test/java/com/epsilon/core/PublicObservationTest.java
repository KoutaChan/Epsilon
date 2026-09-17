package com.epsilon.core;

import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 席ごとの観測に自家の手牌と他家の公開情報だけが含まれ、他家の自摸牌が漏れないことを検証する。 */
public class PublicObservationTest {
  @Test
  public void seatViewExposesOwnTilesAndOnlyPublicOpponentInformation() {
    GameEngine engine = new GameEngine(4L);
    GameStepResult.AwaitingDecisions awaiting =
        (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var point = awaiting.decisions().getFirst();
    PublicObservation view = engine.observation(point.player());
    Assert.assertEquals(view.hand(point.player()).concealedTileCount(), 14);
    Assert.assertTrue(view.hand(point.player()).concealedTileTypeMask() != 0);
    int opponent = (point.player() + 1) % 4;
    Assert.assertEquals(view.hand(opponent).concealedTileCount(), 13);
    Assert.assertEquals(view.hand(opponent).meldCount(), 0);
    Assert.expectThrows(IllegalArgumentException.class, () -> view.hand(opponent).count(0));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> view.hand(opponent).concealedTileTypeMask());
    Assert.expectThrows(IllegalArgumentException.class, () -> view.isTemporaryFuriten(opponent));
    EngineDecisionBuffer analysis = new EngineDecisionBuffer();
    view.analyze(point.legalActions(), analysis);
    Assert.assertSame(analysis.state(), view);
    Assert.assertEquals(analysis.actionCount(), point.legalActions().size());
    Assert.expectThrows(
        IllegalArgumentException.class, () -> analysis.state().hand(opponent).count(0));
  }

  @Test
  public void opponentDrawDoesNotRevealPhysicalTile() {
    GameEngine engine = new GameEngine(7L);
    var awaiting = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    int current = awaiting.decisions().getFirst().player();
    var ownDraw = (TurnEvent.Draw) engine.observation(current).turnEvent();
    var opponentDraw = (TurnEvent.Draw) engine.observation((current + 1) % 4).turnEvent();
    Assert.assertTrue(ownDraw.physicalTileId() >= 0);
    Assert.assertEquals(opponentDraw.physicalTileId(), -1);
    Assert.assertEquals(opponentDraw.player(), current);
  }
}

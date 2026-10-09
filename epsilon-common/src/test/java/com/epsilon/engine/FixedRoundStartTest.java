package com.epsilon.engine;

import org.testng.Assert;
import org.testng.annotations.Test;

public class FixedRoundStartTest {
  @Test
  public void freshSeededRoundPreservesPublicConditionsAndDealsHiddenHands() {
    int[] scores = {31000, 24000, 21000, 22000};
    RoundStartPosition root = new RoundStartPosition(5, 1, 2, 2, scores);
    scores[0] = 0;
    GameEngine a = new GameEngine(123L, null, root);
    GameEngine b = new GameEngine(123L, null, root);
    var pa = (GameStepResult.AwaitingDecisions) a.stepHanchan();
    b.stepHanchan();
    Assert.assertEquals(pa.decisions().getFirst().player(), 1);
    var view = a.observation(1);
    Assert.assertEquals(view.roundIndex(), 5);
    Assert.assertEquals(view.honba(), 2);
    Assert.assertEquals(view.riichiSticks(), 2);
    Assert.assertEquals(view.score(0), 31000);
    Assert.assertEquals(view.hand(1).concealedTileCount(), 14);
    for (int seat = 0; seat < 4; seat++) {
      for (int tile = 0; tile < 34; tile++) {
        Assert.assertEquals(
            a.observation(seat).hand(seat).count(tile), b.observation(seat).hand(seat).count(tile));
      }
    }
  }

  @Test
  public void rejectsInvalidRootMoneyAndDealer() {
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new RoundStartPosition(0, 0, 0, 1, new int[] {25000, 25000, 25000, 25000}));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new RoundStartPosition(1, 0, 0, 0, new int[] {25000, 25000, 25000, 25000}));
  }
}

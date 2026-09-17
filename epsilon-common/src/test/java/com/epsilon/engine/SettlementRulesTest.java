package com.epsilon.engine;

import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 和了点の丸めと満貫以上の点数、流局時の点棒移動、四風連打と四家立直の成立時点を検証する。 */
public class SettlementRulesTest {
  @DataProvider(name = "payments")
  public Object[][] payments() {
    return new Object[][] {
      {1, 30, 1000, 1500}, {2, 25, 1600, 2400}, {3, 30, 3900, 5800},
      {4, 30, 7700, 11600}, {4, 40, 8000, 12000}, {6, 30, 12000, 18000},
      {8, 30, 16000, 24000}, {11, 30, 24000, 36000}, {13, 30, 32000, 48000}
    };
  }

  @Test(dataProvider = "payments")
  public void roundsRonPaymentsAndAppliesLimitHands(int han, int fu, int child, int dealer) {
    int base = ScoreMath.normalBasePoints(han, fu);
    Assert.assertEquals(ScorePayments.ron(base, WinnerRole.CHILD), new WinPayment.Ron(child));
    Assert.assertEquals(ScorePayments.ron(base, WinnerRole.DEALER), new WinPayment.Ron(dealer));
  }

  @Test
  public void exhaustiveDrawTransfersExactlyThreeThousandPoints() {
    GameState state = MahjongFixtures.state();
    MahjongFixtures.hand(state, 0, "123456789m123p4p");
    for (int seat = 1; seat < 4; seat++) MahjongFixtures.hand(state, seat, "13579m2468p1359s");
    var result = (RoundResult.ExhaustiveDraw) new RoundDrawResolver(state).resolveExhaustiveDraw();
    Assert.assertTrue(result.isTenpai(0));
    Assert.assertEquals(result.pointDelta().toArray(), new int[] {3000, -1000, -1000, -1000});
  }

  @Test
  public void fourthIdenticalWindDiscardAbortsTheRound() {
    GameState state = MahjongFixtures.state();
    for (int seat = 0; seat < 4; seat++) state.commitDiscard(seat, Tile.TON, false);
    var result = (RoundResult.AbortiveDraw) new RoundDrawResolver(state).afterDiscard(false);
    Assert.assertEquals(result.reason(), RoundResult.AbortiveDrawReason.FOUR_WINDS);
  }

  @Test
  public void fourRiichiWaitUntilTheDeclarationDiscardIsCommitted() {
    GameState state = MahjongFixtures.state();
    for (int seat = 0; seat < 4; seat++) state.setRiichi(seat, true);
    RoundDrawResolver resolver = new RoundDrawResolver(state);
    Assert.assertNull(resolver.afterDiscard(false));
    Assert.assertEquals(
        ((RoundResult.AbortiveDraw) resolver.afterDiscard(true)).reason(),
        RoundResult.AbortiveDrawReason.FOUR_RIICHI);
  }
}

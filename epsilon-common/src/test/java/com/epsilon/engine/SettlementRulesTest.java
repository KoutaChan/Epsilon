package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.calculate.scoring.YakuBits;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
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
    Assert.assertEquals(ScorePayments.ronPoints(base, false), child);
    Assert.assertEquals(ScorePayments.ronPoints(base, true), dealer);
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

  @DataProvider(name = "fuCodes")
  public Object[][] fuCodes() {
    return new Object[][] {{0, 0}, {20, 1}, {25, 2}, {30, 3}, {40, 4}, {110, 11}};
  }

  @Test(dataProvider = "fuCodes")
  public void fuCodesRoundTrip(int fu, int code) {
    Assert.assertEquals(ScoreMath.encodeFuCode(fu), code);
    Assert.assertEquals(ScoreMath.decodeFuCode(code), fu);
  }

  @Test
  public void rejectsUnsupportedFuCodes() {
    Assert.expectThrows(IllegalArgumentException.class, () -> ScoreMath.encodeFuCode(22));
    Assert.expectThrows(IllegalArgumentException.class, () -> ScoreMath.decodeFuCode(12));
  }

  @DataProvider(name = "singleWinHanchanEnd")
  public Object[][] singleWinHanchanEnd() {
    return new Object[][] {
      {6, 0, 1, 40000, 20000, 20000, 20000, false},
      {7, 0, 1, 30000, 30000, 20000, 20000, true},
      {7, 0, 1, 29900, 29900, 20100, 20100, false},
      {7, 0, 0, 30000, 25000, 25000, 20000, true},
      {7, 1, 1, 30000, 30000, 20000, 20000, false},
      {7, 0, 1, 35000, -100, 30100, 35000, true},
      {11, 0, 1, 25000, 25000, 25000, 25000, true}
    };
  }

  @Test(dataProvider = "singleWinHanchanEnd")
  public void singleWinHanchanEndUsesScoresBeforeKyotaku(
      int kyoku,
      int oya,
      int winner,
      int score0,
      int score1,
      int score2,
      int score3,
      boolean expected) {
    Assert.assertEquals(
        HanchanProgression.shouldFinishAfterSingleWin(
            kyoku, oya, winner, score0, score1, score2, score3),
        expected);
  }

  @Test
  public void paoFactOnlyMarksOpenThirdDragonSet() {
    GameState state = MahjongFixtures.state();
    HandScoreBuffer daisangen = HandScoreBuffer.of(false, ScoringYaku.DAISANGEN.bit(), 0, 0, 0, 0);
    Assert.assertFalse(PaoRules.applies(state.hand(1), daisangen));
    state.addMeld(1, Meld.pon(Tile.HAKU, Meld.RelativeSource.SHIMOCHA), 0);
    state.addMeld(1, Meld.pon(Tile.HATSU, Meld.RelativeSource.TOIMEN), 0);
    state.addMeld(1, Meld.pon(Tile.CHUN, Meld.RelativeSource.KAMICHA), 0);
    boolean paoApplies = PaoRules.applies(state.hand(1), daisangen);
    Assert.assertTrue(paoApplies);

    HandAnalysisBuffer facts = new HandAnalysisBuffer();
    facts.clear();
    facts.appendWait(Tile.M1, daisangen, daisangen, false, paoApplies, paoApplies);
    Assert.assertNotEquals(facts.ronYakuBits(0), 0L);
    Assert.assertEquals(facts.ronHanWithoutUra(0), 0);
    Assert.assertEquals(facts.ronFu(0), 0);
    Assert.assertEquals(YakuBits.yakumanCount(facts.ronYakuBits(0)), 1);
    Assert.assertTrue(facts.ronPaoApplies(0));
  }

  @DataProvider(name = "paoSettlement")
  public Object[][] paoSettlement() {
    return new Object[][] {{false}, {true}};
  }

  @Test(dataProvider = "paoSettlement")
  public void claimAndBufferedSettlementAgreeWithAndWithoutPao(boolean pao) {
    GameState state = MahjongFixtures.state();
    HandScoreBuffer score;
    if (pao) {
      state.addMeld(1, Meld.pon(Tile.HAKU, Meld.RelativeSource.SHIMOCHA), 0);
      state.addMeld(1, Meld.pon(Tile.HATSU, Meld.RelativeSource.TOIMEN), 0);
      state.addMeld(1, Meld.pon(Tile.CHUN, Meld.RelativeSource.KAMICHA), 0);
      score = HandScoreBuffer.of(false, ScoringYaku.DAISANGEN.bit(), 0, 0, 0, 0);
    } else {
      score = HandScoreBuffer.of(true, ScoringYaku.RIICHI.bit(), 0, 0, 0, 30);
    }
    Assert.assertEquals(PaoRules.applies(state.hand(1), score), pao);

    PointDeltaBuffer tsumoDelta = new PointDeltaBuffer();
    WinSettlementCalculator.settleTsumoInto(1, 0, 2, score, state.hand(1), tsumoDelta);
    WinClaim.Tsumo tsumoClaim =
        WinSettlementCalculator.createTsumoClaim(1, 0, 2, score, state.hand(1), false);
    assertDelta(tsumoDelta, tsumoClaim.pointDelta());

    PointDeltaBuffer ronDelta = new PointDeltaBuffer();
    WinSettlementCalculator.settleRonInto(1, 2, 0, 2, score, state.hand(1), ronDelta);
    WinClaim.Ron ronClaim =
        WinSettlementCalculator.createRonClaim(1, 2, 0, 2, score, state.hand(1), false);
    assertDelta(ronDelta, ronClaim.pointDelta());
  }

  private static void assertDelta(PointDeltaBuffer actual, PointDelta expected) {
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      Assert.assertEquals(actual.get(seat), expected.get(seat));
    }
  }
}

package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** {@link WinPointDeltas} は、包のない和了の精算（{@link WinSettlementCalculator}）の点の移動に、和了者の供託を足した値を返す。 */
public class WinPointDeltasTest {

  @DataProvider(name = "wins")
  public Object[][] wins() {
    return new Object[][] {
      // 和了者、放銃者（和了者と同じならツモ）、親、本場、供託、翻、符
      {0, 0, 0, 0, 0, 1, 30},
      {1, 1, 0, 2, 1, 3, 40},
      {2, 2, 1, 0, 3, 13, 30},
      {0, 2, 0, 1, 0, 4, 30},
      {3, 1, 0, 3, 2, 5, 30},
      {1, 0, 0, 0, 1, 2, 25}
    };
  }

  @Test(dataProvider = "wins")
  public void matchesSettlementPlusKyotaku(
      int winner, int from, int dealer, int honba, int kyotaku, int han, int fu) {
    HandScoreBuffer score = HandScoreBuffer.of(true, ScoringYaku.RIICHI.bit(), han - 1, 0, 0, fu);
    Hand hand = new Hand();
    PointDeltaBuffer settled = new PointDeltaBuffer();
    if (from == winner) {
      WinSettlementCalculator.settleTsumoInto(winner, dealer, honba, score, hand, settled);
    } else {
      WinSettlementCalculator.settleRonInto(winner, from, dealer, honba, score, hand, settled);
    }
    int[] deltas = new int[GameState.NUM_PLAYERS];
    WinPointDeltas.settleInto(score.basePoints(), winner, from, dealer, honba, kyotaku, deltas);
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      int sticks = seat == winner ? kyotaku * HanchanProgression.RIICHI_COST : 0;
      Assert.assertEquals(deltas[seat], settled.get(seat) + sticks, "seat " + seat);
    }
  }

  /** 基本点は 1 から全ての役満が重なった和了（8000 × 12）まで。外は点を溢れさせずに例外にする。 */
  @Test
  public void rejectsBasePointsOutsideTheRules() {
    int[] deltas = new int[GameState.NUM_PLAYERS];
    Assert.assertEquals(WinPointDeltas.MAX_BASE_POINTS, 8000 * 12);
    WinPointDeltas.settleInto(WinPointDeltas.MAX_BASE_POINTS, 0, 1, 0, 0, 0, deltas);
    Assert.assertEquals(deltas[1], -6 * WinPointDeltas.MAX_BASE_POINTS);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> WinPointDeltas.settleInto(WinPointDeltas.MAX_BASE_POINTS + 1, 0, 1, 0, 0, 0, deltas));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> WinPointDeltas.settleInto(Integer.MAX_VALUE, 0, 0, 0, 0, 0, deltas));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> WinPointDeltas.settleInto(0, 0, 1, 0, 0, 0, deltas));
  }
}

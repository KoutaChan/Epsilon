package com.epsilon.prism;

import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 上限の値と、上限が使う観測（O）の範囲が、これまでの四人打ちの値から変わらないことを確かめる。この範囲は、補助のヘッドのフィンガープリントと
 * バッチの大きさに入る。
 */
public class LimitsTest {

  @Test
  public void constantsKeepTheFourPlayerValues() {
    Assert.assertEquals(Limits.MAX_INDICATORS, 5);
    Assert.assertEquals(Limits.MAX_EVENT_TOKENS, 110);
    Assert.assertEquals(Limits.MAX_LEGAL_ACTIONS, 32);
    Assert.assertEquals(Limits.MAX_TRANSITIONS, 192);
    // 観測（O）の範囲。入力と補助のヘッドは、観測からこの範囲の値を読む。
    Assert.assertEquals(ObservationSchema.MAX_CALL_DISCARDS, 11);
    Assert.assertEquals(ObservationSchema.MAX_FU_CODE, 11);
    Assert.assertEquals(ObservationSchema.MAX_YAKUMAN, 12);
    Assert.assertEquals(Meld.IDS, 8160);
    Assert.assertEquals(Limits.MAX_WIN_POINTS, 576_000);
    Assert.assertEquals(Limits.MAX_DEAL_IN_POINTS, 960_000);
    Assert.assertEquals(Limits.MAX_WAIT_KINDS, 13);
    Assert.assertEquals(Limits.MAX_SHANTEN, 6);
    Assert.assertEquals(Limits.MAX_STANDARD_SHANTEN, 8);
    Assert.assertEquals(Limits.MAX_CHIITOI_SHANTEN, 6);
    Assert.assertEquals(Limits.MAX_KOKUSHI_SHANTEN, 13);
    Assert.assertEquals(Limits.MAX_OPEN_YAKUHAI_HAN, 5);
  }

  @Test
  public void variantValuesAtTenhouFourPlayer() {
    Variant four = Variant.TENHOU_4P;
    Assert.assertEquals(Limits.akaTiles(four), 3);
    Assert.assertEquals(Limits.maxKyotaku(four), 100);
    Assert.assertEquals(Limits.maxDora(four), 23);
    Assert.assertEquals(Limits.maxWinHan(four), 75);
    Assert.assertEquals(four.maxDrawsPerSeat(), 18);
  }
}

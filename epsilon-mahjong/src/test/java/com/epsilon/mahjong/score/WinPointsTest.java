package com.epsilon.mahjong.score;

import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 採点で得た基本点からの払いを、天鳳の点数表の値で確かめる（四人打ちと三人打ち、親と子、ロンとツモ）。払いは {@code ScoreMath} と
 * {@code Settlement} が求め、三人打ちのツモ損も {@code Settlement} が受け持つ。和了者は、親なら席 0、子なら席 1。
 */
public class WinPointsTest {

  /** 立直して和了する 40 符の手（么九の暗刻 8 ＋ 嵌張 2）。和了牌は 5s。表示牌 4p と 3p のドラは、それぞれ 5p と 4p。 */
  private static final String FORTY_FU = "111m234p567p99s46s";

  private static final String PINFU = "123p456p789s55s23s";

  /** 立直 ＋ ドラ 2 の 3 翻 40 符のロンは、子が 5200 点、親が 7700 点（四人打ちも三人打ちも同じ）。 */
  @Test
  public void aThreeHanFortyFuRonPaysTheSameInBothVariants() {
    for (Variant variant : Variant.values()) {
      WinScore child = WinCase.of(variant, FORTY_FU).riichi().ron("5s").dora("4p3p").score();
      Assert.assertEquals(child.han(), 3, variant + " " + child);
      Assert.assertEquals(child.fu(), 40, variant + " " + child);
      Assert.assertEquals(ScoreMath.ron(child.base(), false), 5_200, variant.name());
      WinScore dealer = WinCase.of(variant, FORTY_FU).dealer().riichi().ron("5s").dora("4p3p").score();
      Assert.assertEquals(ScoreMath.ron(dealer.base(), true), 7_700, variant.name());
    }
  }

  /** 立直・ツモ・ドラ 1 の 3 翻 40 符のツモ。四人打ちでは子が 1300・2600、親が 2600 オール。 */
  @Test
  public void aFourPlayerTsumoTakesFromEverySeat() {
    WinScore child = WinCase.four(FORTY_FU).riichi().tsumo("5s").dora("4p").score();
    Assert.assertEquals(child.han(), 3, child.toString());
    Assert.assertEquals(child.fu(), 40, child.toString());
    assertTsumo(Variant.TENHOU_4P, child, 1, -2_600, 5_200, -1_300, -1_300);
    WinScore dealer = WinCase.four(FORTY_FU).dealer().riichi().tsumo("5s").dora("4p").score();
    assertTsumo(Variant.TENHOU_4P, dealer, 0, 7_800, -2_600, -2_600, -2_600);
  }

  /** 三人打ちで同じ手をツモると、いない北家の分は誰も払わない（子は 1300・2600 で 3900 点、親は 2600 点ずつ 2 席から）。 */
  @Test
  public void aThreePlayerTsumoLosesTheMissingSeatsShare() {
    WinScore child = WinCase.three(FORTY_FU).riichi().tsumo("5s").dora("4p").score();
    assertTsumo(Variant.TENHOU_3P, child, 1, -2_600, 3_900, -1_300);
    WinScore dealer = WinCase.three(FORTY_FU).dealer().riichi().tsumo("5s").dora("4p").score();
    assertTsumo(Variant.TENHOU_3P, dealer, 0, 5_200, -2_600, -2_600);
  }

  /** 4 翻 40 符は満貫になる（切り上げ満貫は無いが、40 符 4 翻は基本点 2560 で満貫に届く）。 */
  @Test
  public void manganTsumoInBothVariants() {
    WinScore four = WinCase.four(FORTY_FU).riichi().tsumo("5s").dora("4p3p").score();
    Assert.assertEquals(four.base(), 2_000, four.toString());
    assertTsumo(Variant.TENHOU_4P, four, 1, -4_000, 8_000, -2_000, -2_000);
    WinScore three = WinCase.three(FORTY_FU).dealer().riichi().tsumo("5s").dora("4p3p").score();
    assertTsumo(Variant.TENHOU_3P, three, 0, 8_000, -4_000, -4_000);
  }

  /** 平和ツモの 2 翻 20 符。四人打ちの子は 400・700 で 1500 点、三人打ちの子は 1100 点。 */
  @Test
  public void pinfuTsumoInBothVariants() {
    WinScore four = WinCase.four(PINFU).tsumo("1s").score();
    Assert.assertEquals(four.fu(), 20, four.toString());
    assertTsumo(Variant.TENHOU_4P, four, 1, -700, 1_500, -400, -400);
    WinScore three = WinCase.three(PINFU).tsumo("1s").score();
    assertTsumo(Variant.TENHOU_3P, three, 1, -700, 1_100, -400);
  }

  @Test
  public void yakumanPaysEightThousandBasePerMultiple() {
    WinScore child = WinCase.four("19m19p19s1234567z").ron("1m").score();
    Assert.assertEquals(ScoreMath.ron(child.base(), false), 32_000);
    WinScore dealer = WinCase.four("19m19p19s1234567z").dealer().ron("1m").score();
    Assert.assertEquals(ScoreMath.ron(dealer.base(), true), 48_000);
    WinScore three = WinCase.three("19m19p19s1234567z").dealer().tsumo("1m").score();
    assertTsumo(Variant.TENHOU_3P, three, 0, 32_000, -16_000, -16_000);
    WinScore heavenly = WinCase.four("19m19p19s1234567z").dealer().firstDraw().tsumo("1m").score();
    assertTsumo(Variant.TENHOU_4P, heavenly, 0, 96_000, -32_000, -32_000, -32_000);
  }

  private static void assertTsumo(Variant variant, WinScore score, int winner, int... expected) {
    int[] delta = new int[variant.seats()];
    Settlement.tsumo(variant, score.base(), winner, 0, 0, RoundOutcome.NO_SEAT, delta);
    Assert.assertEquals(delta, expected, variant + " " + score);
  }
}

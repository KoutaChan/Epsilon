package com.epsilon.mahjong.rules;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 基本点と、ロン・ツモの払いを、天鳳の点数表の値で確かめる。 */
public class ScoreMathTest {

  private static final int[] FUS = {20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110};

  @Test
  public void belowManganTheBaseIsFuTimesTwoToTheTwoPlusHan() {
    Assert.assertEquals(ScoreMath.base(1, 30, 0), 240);
    Assert.assertEquals(ScoreMath.base(1, 110, 0), 880);
    Assert.assertEquals(ScoreMath.base(2, 20, 0), 320);
    Assert.assertEquals(ScoreMath.base(2, 25, 0), 400);
    Assert.assertEquals(ScoreMath.base(3, 30, 0), 960);
    Assert.assertEquals(ScoreMath.base(3, 60, 0), 1920);
    Assert.assertEquals(ScoreMath.base(4, 20, 0), 1280);
    Assert.assertEquals(ScoreMath.base(4, 30, 0), 1920, "no rounding up to mangan");
  }

  @Test
  public void fromManganTheBaseFollowsTheLimits() {
    Assert.assertEquals(ScoreMath.base(4, 40, 0), 2000);
    Assert.assertEquals(ScoreMath.base(3, 70, 0), 2000);
    int[] byHan = {0, 0, 0, 0, 0, 2000, 3000, 3000, 4000, 4000, 4000, 6000, 6000, 8000, 8000, 8000};
    for (int han = 5; han < byHan.length; han++) {
      for (int fu : FUS) {
        Assert.assertEquals(ScoreMath.base(han, fu, 0), byHan[han], han + " han " + fu + " fu");
      }
    }
  }

  @Test
  public void yakumanIsEightThousandPerMultipleWhateverTheHanAndFu() {
    for (int multiple = 1; multiple <= 6; multiple++) {
      Assert.assertEquals(ScoreMath.base(0, 0, multiple), 8_000 * multiple);
      Assert.assertEquals(ScoreMath.base(13, 40, multiple), 8_000 * multiple);
    }
  }

  /** どの翻と符の組でも、基本点は表にある値になる。20 符（平和のツモ）と 25 符（七対子）は 2 翻から数える。 */
  @Test
  public void everyHandLandsInTheBaseTable() {
    for (int han = 1; han <= 20; han++) {
      for (int fu : FUS) {
        if (han >= 2 || fu > 25) {
          BaseClass.of(ScoreMath.base(han, fu, 0));
        }
      }
    }
    for (int multiple = 1; multiple <= 6; multiple++) {
      BaseClass.of(ScoreMath.base(0, 0, multiple));
    }
  }

  @Test
  public void inputsWithoutBasePointsAreRejected() {
    Assert.assertThrows(IllegalArgumentException.class, () -> ScoreMath.base(1, 30, -1));
    Assert.assertThrows(IllegalArgumentException.class, () -> ScoreMath.base(0, 30, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> ScoreMath.base(2, 10, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> ScoreMath.ron(0, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> ScoreMath.tsumo(-240, false, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> ScoreMath.tsumo(2_000, true, true));
  }

  @Test
  public void ronIsFourBaseFromAChildWinAndSixBaseFromADealerWin() {
    int[] bases = {240, 320, 400, 960, 1_920, 2_000, 3_000, 8_000};
    int[] child = {1_000, 1_300, 1_600, 3_900, 7_700, 8_000, 12_000, 32_000};
    int[] dealer = {1_500, 2_000, 2_400, 5_800, 11_600, 12_000, 18_000, 48_000};
    for (int i = 0; i < bases.length; i++) {
      Assert.assertEquals(ScoreMath.ron(bases[i], false), child[i], "child " + bases[i]);
      Assert.assertEquals(ScoreMath.ron(bases[i], true), dealer[i], "dealer " + bases[i]);
    }
  }

  @Test
  public void tsumoIsTwoBaseFromTheDealerOrForADealerWinAndOneBaseOtherwise() {
    int[] bases = {240, 320, 960, 1_280, 2_000, 3_000};
    int[] childFromChild = {300, 400, 1_000, 1_300, 2_000, 3_000};
    int[] childFromDealer = {500, 700, 2_000, 2_600, 4_000, 6_000};
    int[] dealerFromChild = {500, 700, 2_000, 2_600, 4_000, 6_000};
    for (int i = 0; i < bases.length; i++) {
      Assert.assertEquals(ScoreMath.tsumo(bases[i], false, false), childFromChild[i], "child win " + bases[i]);
      Assert.assertEquals(ScoreMath.tsumo(bases[i], false, true), childFromDealer[i], "child win " + bases[i]);
      Assert.assertEquals(ScoreMath.tsumo(bases[i], true, false), dealerFromChild[i], "dealer win " + bases[i]);
    }
  }

  /** どの基本点でも、払いは式の値を 100 点単位に切り上げた額になる。 */
  @Test
  public void paymentsRoundUpToTheNextHundred() {
    for (BaseClass base : BaseClass.values()) {
      int b = base.base();
      assertRoundedUp(ScoreMath.ron(b, false), 4 * b);
      assertRoundedUp(ScoreMath.ron(b, true), 6 * b);
      assertRoundedUp(ScoreMath.tsumo(b, false, false), b);
      assertRoundedUp(ScoreMath.tsumo(b, false, true), 2 * b);
      assertRoundedUp(ScoreMath.tsumo(b, true, false), 2 * b);
    }
  }

  private static void assertRoundedUp(int paid, int exact) {
    Assert.assertEquals(paid % 100, 0, paid + " for " + exact);
    Assert.assertTrue(paid >= exact && paid - exact < 100, paid + " for " + exact);
  }
}

package com.epsilon.mahjong.rules;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 順位の効用の値と、フィンガープリントや config.json に書く名前を確かめる。 */
public class RankUtilityTest {

  @Test
  public void tenhouKeepsItsNameAndFourPlayerValues() {
    Assert.assertEquals(RankUtility.TENHOU.name(), "TENHOU");
    Assert.assertSame(RankUtility.valueOf("TENHOU"), RankUtility.TENHOU);
    assertValues(RankUtility.TENHOU, Variant.TENHOU_4P, 2.0 / 3, 1.0 / 3, 0, -1);
    assertValues(RankUtility.TENHOU, Variant.TENHOU_3P, 1, 0, -1);
  }

  @Test
  public void averageRankIsLinearInTheRank() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      for (int rank = 0; rank < seats; rank++) {
        double expected = 1 - 2.0 * rank / (seats - 1);
        Assert.assertEquals(RankUtility.AVERAGE_RANK.value(variant, rank), expected, 1e-15);
      }
    }
    assertValues(RankUtility.AVERAGE_RANK, Variant.TENHOU_4P, 1, 1.0 / 3, -1.0 / 3, -1);
  }

  @Test
  public void everyUtilityFallsWithTheRankInEveryVariant() {
    for (RankUtility utility : RankUtility.values()) {
      for (Variant variant : Variant.values()) {
        for (int rank = 1; rank < variant.seats(); rank++) {
          Assert.assertTrue(
              utility.value(variant, rank) < utility.value(variant, rank - 1), utility + " " + variant + " " + rank);
        }
        int outside = variant.seats();
        Assert.assertThrows(ArrayIndexOutOfBoundsException.class, () -> utility.value(variant, outside));
      }
    }
  }

  private static void assertValues(RankUtility utility, Variant variant, double... expected) {
    for (int rank = 0; rank < expected.length; rank++) {
      Assert.assertEquals(utility.value(variant, rank), expected[rank], 0.0, utility + " " + variant + " " + rank);
    }
  }
}

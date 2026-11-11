package com.epsilon.mahjong.rules;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 局の始まりの不変条件と、点の配列をコピーして持つことを確かめる。 */
public class RoundStartTest {

  @Test
  public void firstRoundGivesEverySeatTheStartingPoints() {
    for (Variant variant : Variant.values()) {
      RoundStart start = RoundStart.first(variant);
      Assert.assertEquals(start.round(), 0);
      Assert.assertEquals(start.honba(), 0);
      Assert.assertEquals(start.kyotaku(), 0);
      Assert.assertEquals(start.scores().length, variant.seats());
      for (int seat = 0; seat < variant.seats(); seat++) {
        Assert.assertEquals(start.score(seat), variant.startPoints());
      }
    }
  }

  @Test
  public void pointsAndDepositsMustAddUp() {
    Variant four = Variant.TENHOU_4P;
    int[] oneShort = {25_000, 25_000, 25_000, 24_000};
    int[] even = {25_000, 25_000, 25_000, 25_000};
    Assert.assertEquals(new RoundStart(four, 5, 2, 1, oneShort).kyotaku(), 1);
    Assert.assertThrows(IllegalArgumentException.class, () -> new RoundStart(four, 5, 2, 0, oneShort));
    Assert.assertThrows(IllegalArgumentException.class, () -> new RoundStart(four, 0, -1, 0, even));
    Assert.assertThrows(IllegalArgumentException.class, () -> new RoundStart(four, 0, 0, -1, even));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> new RoundStart(four, 0, 0, 0, new int[] {25_000, 25_000, 50_000}));
  }

  @Test
  public void roundMustExistInTheVariant() {
    int[] four = {25_000, 25_000, 25_000, 25_000};
    int[] three = {35_000, 35_000, 35_000};
    Assert.assertEquals(new RoundStart(Variant.TENHOU_4P, 11, 0, 0, four).round(), 11);
    Assert.assertEquals(new RoundStart(Variant.TENHOU_3P, 8, 0, 0, three).round(), 8);
    Assert.assertThrows(IllegalArgumentException.class, () -> new RoundStart(Variant.TENHOU_4P, 12, 0, 0, four));
    Assert.assertThrows(IllegalArgumentException.class, () -> new RoundStart(Variant.TENHOU_3P, 9, 0, 0, three));
    Assert.assertThrows(IllegalArgumentException.class, () -> new RoundStart(Variant.TENHOU_3P, -1, 0, 0, three));
  }

  @Test
  public void scoresAreCopied() {
    int[] scores = {40_000, 30_000, 35_000};
    RoundStart start = new RoundStart(Variant.TENHOU_3P, 1, 0, 0, scores);
    scores[0] = 0;
    start.scores()[1] = 0;
    Assert.assertEquals(start.score(0), 40_000);
    Assert.assertEquals(start.score(1), 30_000);
  }

  @Test
  public void equalityComparesTheScores() {
    RoundStart one = new RoundStart(Variant.TENHOU_3P, 2, 1, 0, new int[] {40_000, 30_000, 35_000});
    RoundStart same = new RoundStart(Variant.TENHOU_3P, 2, 1, 0, new int[] {40_000, 30_000, 35_000});
    RoundStart other = new RoundStart(Variant.TENHOU_3P, 2, 1, 0, new int[] {30_000, 40_000, 35_000});
    Assert.assertEquals(one, same);
    Assert.assertEquals(one.hashCode(), same.hashCode());
    Assert.assertNotEquals(one, other);
  }
}

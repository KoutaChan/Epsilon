package com.epsilon.mahjong.rules;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 卓の値の読み出し、コピー、終局で残った供託の渡し方を確かめる。 */
public class TableStateTest {

  @Test
  public void startsAtTheFirstRound() {
    for (Variant variant : Variant.values()) {
      TableState table = new TableState(variant);
      Assert.assertSame(table.variant(), variant);
      Assert.assertEquals(table.round(), 0);
      Assert.assertEquals(table.dealer(), 0);
      Assert.assertEquals(table.honba(), 0);
      Assert.assertEquals(table.kyotaku(), 0);
      Assert.assertFalse(table.gameOver());
      for (int seat = 0; seat < variant.seats(); seat++) {
        Assert.assertEquals(table.score(seat), variant.startPoints());
        Assert.assertEquals(table.rank(seat), seat);
      }
      Assert.assertEquals(table.top(), 0);
    }
  }

  @Test
  public void setReadsTheRoundStart() {
    TableState table = new TableState(Variant.TENHOU_3P);
    table.set(new RoundStart(Variant.TENHOU_3P, 4, 2, 1, new int[] {40_000, 30_000, 34_000}));
    Assert.assertEquals(table.round(), 4);
    Assert.assertEquals(table.dealer(), 1);
    Assert.assertEquals(table.honba(), 2);
    Assert.assertEquals(table.kyotaku(), 1);
    Assert.assertEquals(table.score(2), 34_000);
    Assert.assertEquals(table.rank(0), 0);
    Assert.assertEquals(table.rank(1), 2);
    Assert.assertEquals(table.rank(2), 1);
    Assert.assertFalse(table.gameOver());
  }

  /**
   * 値を渡して卓を書き換える {@code set} は、メモリを確保せずに {@code RoundStart} と同じ決まりで値を確かめ、点をコピーして
   * 終局の印を消す。拒んだときは卓を変えない。
   */
  @Test
  public void setFromValuesChecksLikeTheRoundStart() {
    TableState table = new TableState(Variant.TENHOU_4P);
    table.finish();
    int[] scores = {31_000, 18_000, 26_500, 23_500};
    table.set(5, 2, 1, scores);
    scores[0] = 0;
    Assert.assertEquals(table.round(), 5);
    Assert.assertEquals(table.honba(), 2);
    Assert.assertEquals(table.kyotaku(), 1);
    Assert.assertEquals(table.score(0), 31_000);
    Assert.assertFalse(table.gameOver());
    int[] even = {25_000, 25_000, 25_000, 25_000};
    Assert.assertThrows(IllegalArgumentException.class, () -> table.set(12, 0, 0, even));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.set(0, -1, 0, even));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.set(0, 0, 1, even));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.set(0, 0, 0, new int[] {35_000, 35_000, 35_000}));
    Assert.assertEquals(table.score(0), 31_000, "a rejected set changes nothing");
  }

  /** もしもの卓は点と供託の合計を問わない。局の番号・本場・供託・点の個数だけを {@code RoundStart} と同じ決まりで確かめる。 */
  @Test
  public void aScenarioSkipsOnlyTheTotal() {
    TableState table = new TableState(Variant.TENHOU_4P);
    table.finish();
    int[] scores = {3_270_000, -3_270_000, 0, 100};
    table.setScenario(11, 255, 100, scores);
    scores[0] = 0;
    Assert.assertEquals(table.round(), 11);
    Assert.assertEquals(table.honba(), 255);
    Assert.assertEquals(table.kyotaku(), 100);
    Assert.assertEquals(table.score(0), 3_270_000);
    Assert.assertEquals(table.score(1), -3_270_000);
    Assert.assertFalse(table.gameOver());
    int[] even = {25_000, 25_000, 25_000, 25_000};
    Assert.assertThrows(IllegalArgumentException.class, () -> table.setScenario(12, 0, 0, even));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.setScenario(-1, 0, 0, even));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.setScenario(0, -1, 0, even));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.setScenario(0, 0, -1, even));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.setScenario(0, 0, 0, new int[3]));
    Assert.assertEquals(table.score(0), 3_270_000, "a rejected scenario changes nothing");
    Assert.assertEquals(table.honba(), 255, "a rejected scenario changes nothing");
  }

  @Test
  public void startsAtComparesEveryFieldOfTheRoundStart() {
    RoundStart start = new RoundStart(Variant.TENHOU_4P, 5, 1, 2, new int[] {30_000, 20_000, 24_000, 24_000});
    TableState table = new TableState(Variant.TENHOU_4P);
    table.set(start);
    Assert.assertTrue(table.startsAt(start));
    Assert.assertFalse(table.startsAt(new RoundStart(Variant.TENHOU_4P, 6, 1, 2, start.scores())));
    Assert.assertFalse(table.startsAt(new RoundStart(Variant.TENHOU_4P, 5, 0, 2, start.scores())));
    Assert.assertFalse(
        table.startsAt(new RoundStart(Variant.TENHOU_4P, 5, 1, 1, new int[] {31_000, 20_000, 24_000, 24_000})));
    Assert.assertFalse(
        table.startsAt(new RoundStart(Variant.TENHOU_4P, 5, 1, 2, new int[] {20_000, 30_000, 24_000, 24_000})));
    Assert.assertFalse(table.startsAt(RoundStart.first(Variant.TENHOU_3P)));
    table.finish();
    Assert.assertFalse(table.startsAt(start));
  }

  @Test
  public void endsWithNeedsTheGameOverAndTheSameScores() {
    TableState table = new TableState(Variant.TENHOU_4P);
    table.set(new RoundStart(Variant.TENHOU_4P, 7, 0, 1, new int[] {30_000, 20_000, 24_000, 25_000}));
    Assert.assertFalse(table.endsWith(new int[] {30_000, 20_000, 24_000, 25_000}));
    table.finish();
    Assert.assertTrue(table.endsWith(new int[] {31_000, 20_000, 24_000, 25_000}));
    Assert.assertFalse(table.endsWith(new int[] {30_000, 21_000, 24_000, 25_000}));
  }

  @Test
  public void variantsMustMatch() {
    TableState four = new TableState(Variant.TENHOU_4P);
    Assert.assertThrows(IllegalArgumentException.class, () -> four.set(RoundStart.first(Variant.TENHOU_3P)));
    Assert.assertThrows(IllegalArgumentException.class, () -> four.copyFrom(new TableState(Variant.TENHOU_3P)));
  }

  @Test
  public void copyFromCopiesEveryFieldAndStaysIndependent() {
    TableState source = new TableState(Variant.TENHOU_4P);
    source.set(new RoundStart(Variant.TENHOU_4P, 7, 3, 2, new int[] {40_000, 18_000, 20_000, 20_000}));
    source.finish();
    TableState copy = new TableState(Variant.TENHOU_4P);
    copy.copyFrom(source);
    Assert.assertEquals(copy.round(), 7);
    Assert.assertEquals(copy.honba(), 3);
    Assert.assertEquals(copy.kyotaku(), 0);
    Assert.assertTrue(copy.gameOver());
    for (int seat = 0; seat < 4; seat++) {
      Assert.assertEquals(copy.score(seat), source.score(seat));
    }
    source.setScore(0, 0);
    Assert.assertEquals(copy.score(0), 42_000);
  }

  @Test
  public void finishGivesTheRemainingDepositsToTheTop() {
    TableState table = new TableState(Variant.TENHOU_4P);
    table.set(new RoundStart(Variant.TENHOU_4P, 7, 0, 3, new int[] {20_000, 31_000, 31_000, 15_000}));
    table.finish();
    Assert.assertTrue(table.gameOver());
    Assert.assertEquals(table.kyotaku(), 0);
    Assert.assertEquals(table.score(1), 34_000);
    Assert.assertEquals(table.score(2), 31_000);
    Assert.assertEquals(table.rank(1), 0);
    int total = 0;
    for (int seat = 0; seat < 4; seat++) {
      total += table.score(seat);
    }
    Assert.assertEquals(total, 4 * Variant.TENHOU_4P.startPoints());
  }

  @Test
  public void setRoundContinuesTheGame() {
    TableState table = new TableState(Variant.TENHOU_4P);
    table.finish();
    table.setRound(5, 1, 2);
    table.setScore(0, 23_000);
    Assert.assertFalse(table.gameOver());
    Assert.assertEquals(table.round(), 5);
    Assert.assertEquals(table.dealer(), 1);
    Assert.assertEquals(table.honba(), 1);
    Assert.assertEquals(table.kyotaku(), 2);
    Assert.assertEquals(table.score(0), 23_000);
  }
}

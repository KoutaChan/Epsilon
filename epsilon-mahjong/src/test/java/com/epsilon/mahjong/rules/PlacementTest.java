package com.epsilon.mahjong.rules;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 順位は点の高い順に付け、同点なら東 1 局の席順で上を決める。終局で残った供託の渡し先も確かめる。 */
public class PlacementTest {

  @Test
  public void tiesGoToTheSeatNearerTheFirstDealer() {
    int[] ranks = new int[4];
    Placement.ranks(new int[] {25_000, 30_000, 25_000, 20_000}, ranks);
    Assert.assertEquals(ranks, new int[] {1, 0, 2, 3});
    Placement.ranks(new int[] {25_000, 25_000, 25_000, 25_000}, ranks);
    Assert.assertEquals(ranks, new int[] {0, 1, 2, 3});
    Assert.assertEquals(Placement.top(new int[] {30_000, 30_000, 20_000, 20_000}), 0);
    Assert.assertEquals(Placement.top(new int[] {10_000, 40_000, 40_000}), 1);
  }

  /** 乱数の点（同点が多い）でも、順位は点の高い順、同点なら席の番号の小さい順に並べた順と一致する。 */
  @Test
  public void ranksSortByScoreThenSeat() {
    Random random = new Random(7);
    for (int trial = 0; trial < 10_000; trial++) {
      int seats = 3 + random.nextInt(2);
      int[] scores = new int[seats];
      for (int seat = 0; seat < seats; seat++) {
        scores[seat] = (random.nextInt(7) - 2) * 10_000;
      }
      int[] ranks = new int[seats];
      Placement.ranks(scores, ranks);
      Integer[] order = new Integer[seats];
      for (int seat = 0; seat < seats; seat++) {
        order[seat] = seat;
      }
      Arrays.sort(order, Comparator.comparingInt((Integer seat) -> -scores[seat]).thenComparingInt(seat -> seat));
      for (int rank = 0; rank < seats; rank++) {
        Assert.assertEquals(ranks[order[rank]], rank, Arrays.toString(scores));
        Assert.assertEquals(Placement.rank(scores, order[rank]), rank, Arrays.toString(scores));
      }
      Assert.assertEquals(Placement.top(scores), (int) order[0], Arrays.toString(scores));
    }
  }

  @Test
  public void remainingDepositsGoToTheTop() {
    int[] scores = {30_000, 30_000, 25_000, 13_000};
    Placement.awardDeposits(scores, 2);
    Assert.assertEquals(scores, new int[] {32_000, 30_000, 25_000, 13_000});
    int[] three = {20_000, 45_000, 39_000};
    Placement.awardDeposits(three, 1);
    Assert.assertEquals(three, new int[] {20_000, 46_000, 39_000});
  }
}

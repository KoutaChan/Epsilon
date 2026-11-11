package com.epsilon.prism.luck;

import com.epsilon.mahjong.tile.Tiles;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 裏ドラの数の分布を、1 点に集まる分布・総当たり・2 人の同時分布の例で確かめる。 */
public class UraDistributionTest {

  private static final double EPSILON = 1e-12;

  /** 表示牌の候補が 1 種類なら、裏ドラの数は（その種類で数える裏ドラの数）×（表示牌の枚数）の 1 点に集まる。 */
  @Test
  public void oneCandidateKindGivesAPointMass() {
    UraDistribution distribution = new UraDistribution();
    int[] pool = new int[Tiles.KINDS];
    int[] count = new int[Tiles.KINDS];
    pool[Tiles.M1] = 4;
    count[Tiles.M1] = 2;
    distribution.single(pool, count, 3);
    assertPoint(distribution, 6);
  }

  /** プールのどの種類でも数える裏ドラの数が同じなら、これも 1 点に集まる分布になる。 */
  @Test
  public void kindsWithTheSameCountGiveAPointMass() {
    UraDistribution distribution = new UraDistribution();
    int[] pool = new int[Tiles.KINDS];
    int[] count = new int[Tiles.KINDS];
    for (int kind = Tiles.P1; kind < Tiles.P1 + 9; kind++) {
      pool[kind] = 3;
      count[kind] = 1;
    }
    distribution.single(pool, count, 5);
    assertPoint(distribution, 5);
    distribution.single(pool, new int[Tiles.KINDS], 2);
    assertPoint(distribution, 0);
  }

  /** 小さなプールで、1 人の分布も 2 人の同時分布も、引く組の総当たりと一致する。1 人の分布は、同時分布の周辺分布と一致する。 */
  @Test
  public void smallPoolsMatchBruteForce() {
    UraDistribution distribution = new UraDistribution();
    Random random = new Random(20261109L);
    for (int trial = 0; trial < 300; trial++) {
      int[] pool = new int[Tiles.KINDS];
      int[] first = new int[Tiles.KINDS];
      int[] second = new int[Tiles.KINDS];
      int total = 0;
      int kinds = 1 + random.nextInt(7);
      for (int k = 0; k < kinds; k++) {
        int kind = random.nextInt(Tiles.KINDS);
        int add = 1 + random.nextInt(3);
        if (pool[kind] + add > Tiles.COPIES) {
          continue;
        }
        pool[kind] += add;
        total += add;
        first[kind] = random.nextInt(Tiles.COPIES + 1);
        second[kind] = random.nextInt(Tiles.COPIES + 1);
      }
      if (total == 0) {
        continue;
      }
      int draws = 1 + random.nextInt(Math.min(UraDistribution.MAX_INDICATORS, total));
      double[][] brute = bruteForce(pool, first, second, draws);
      distribution.joint(pool, first, second, draws);
      double sum = 0;
      for (int k1 = 0; k1 <= UraDistribution.MAX_URA; k1++) {
        for (int k2 = 0; k2 <= UraDistribution.MAX_URA; k2++) {
          Assert.assertEquals(distribution.probability(k1, k2), brute[k1][k2], EPSILON, "trial " + trial);
          sum += distribution.probability(k1, k2);
        }
      }
      Assert.assertEquals(sum, 1, EPSILON);
      distribution.single(pool, first, draws);
      Assert.assertEquals(distribution.maxSecond(), 0);
      for (int k1 = 0; k1 <= UraDistribution.MAX_URA; k1++) {
        double marginal = 0;
        for (int k2 = 0; k2 <= UraDistribution.MAX_URA; k2++) {
          marginal += brute[k1][k2];
        }
        Assert.assertEquals(distribution.probability(k1), marginal, EPSILON, "trial " + trial);
        Assert.assertEquals(distribution.probability(k1, 1), 0.0);
      }
    }
  }

  /**
   * 2 人が同じ表示牌で数えるので、同時分布は周辺分布の積と違う。2 種類から表示牌を 1 枚引き、1 人目は片方、2 人目はもう片方だけを
   * 数えると、2 人とも裏が乗ることは無い（積なら 1/4）。2 人が同じ種類を数えると、乗るときは 2 人とも乗る。
   */
  @Test
  public void theJointDistributionDiffersFromTheProduct() {
    UraDistribution distribution = new UraDistribution();
    int[] pool = new int[Tiles.KINDS];
    pool[Tiles.M1] = 1;
    pool[Tiles.S9] = 1;
    int[] first = new int[Tiles.KINDS];
    int[] second = new int[Tiles.KINDS];
    first[Tiles.M1] = 1;
    second[Tiles.S9] = 1;
    distribution.joint(pool, first, second, 1);
    Assert.assertEquals(distribution.probability(1, 1), 0.0);
    Assert.assertEquals(distribution.probability(1, 0), 0.5, EPSILON);
    Assert.assertEquals(distribution.probability(0, 1), 0.5, EPSILON);
    Assert.assertEquals(distribution.probability(0, 0), 0.0);
    double firstMarginal = distribution.probability(1, 0) + distribution.probability(1, 1);
    double secondMarginal = distribution.probability(0, 1) + distribution.probability(1, 1);
    Assert.assertEquals(firstMarginal * secondMarginal, 0.25, EPSILON);

    distribution.joint(pool, first, first, 1);
    Assert.assertEquals(distribution.probability(1, 1), 0.5, EPSILON);
    Assert.assertEquals(distribution.probability(0, 0), 0.5, EPSILON);
    Assert.assertEquals(distribution.probability(1, 0), 0.0);
    Assert.assertEquals(distribution.probability(0, 1), 0.0);
  }

  /** 裏ドラの数が範囲の外なら、その確率は 0。 */
  @Test
  public void outOfRangeCountsHaveNoProbability() {
    UraDistribution distribution = new UraDistribution();
    int[] pool = new int[Tiles.KINDS];
    int[] count = new int[Tiles.KINDS];
    pool[Tiles.TON] = 2;
    pool[Tiles.NAN] = 2;
    count[Tiles.TON] = 3;
    distribution.single(pool, count, 1);
    Assert.assertEquals(distribution.maxFirst(), 3);
    Assert.assertEquals(distribution.probability(-1), 0.0);
    Assert.assertEquals(distribution.probability(4), 0.0);
    Assert.assertEquals(distribution.probability(0, -1), 0.0);
    Assert.assertEquals(distribution.probability(3), 0.5, EPSILON);
    Assert.assertEquals(distribution.probability(0), 0.5, EPSILON);
  }

  /** 表示牌の枚数・プール・数える裏ドラの数が範囲の外なら IllegalArgumentException を投げる。 */
  @Test
  public void invalidArgumentsAreRejected() {
    UraDistribution distribution = new UraDistribution();
    int[] pool = new int[Tiles.KINDS];
    int[] count = new int[Tiles.KINDS];
    pool[Tiles.M5] = 2;
    Assert.assertThrows(IllegalArgumentException.class, () -> distribution.single(pool, count, 0));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> distribution.single(pool, count, UraDistribution.MAX_INDICATORS + 1));
    Assert.assertThrows(IllegalArgumentException.class, () -> distribution.single(pool, count, 3));
    int[] badCount = new int[Tiles.KINDS];
    badCount[Tiles.M5] = Tiles.COPIES + 1;
    Assert.assertThrows(IllegalArgumentException.class, () -> distribution.single(pool, badCount, 1));
    badCount[Tiles.M5] = -1;
    Assert.assertThrows(IllegalArgumentException.class, () -> distribution.joint(pool, count, badCount, 1));
    int[] badPool = pool.clone();
    badPool[Tiles.M6] = -1;
    Assert.assertThrows(IllegalArgumentException.class, () -> distribution.single(badPool, count, 1));
  }

  private static void assertPoint(UraDistribution distribution, int point) {
    for (int k = 0; k <= UraDistribution.MAX_URA; k++) {
      Assert.assertEquals(distribution.probability(k), k == point ? 1.0 : 0.0, EPSILON, "k = " + k);
    }
  }

  /** 裏ドラの数の組ごとの確率を {@code [1 人目の裏][2 人目の裏]} で返す。プールの牌を 1 枚ずつ区別して、引く組を全て数える。 */
  private static double[][] bruteForce(int[] pool, int[] first, int[] second, int draws) {
    int total = 0;
    for (int tiles : pool) {
      total += tiles;
    }
    int[] tiles = new int[total];
    int at = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      for (int copy = 0; copy < pool[kind]; copy++) {
        tiles[at++] = kind;
      }
    }
    double[][] counts = new double[UraDistribution.MAX_URA + 1][UraDistribution.MAX_URA + 1];
    long subsets = enumerate(tiles, 0, draws, 0, 0, first, second, counts);
    for (double[] row : counts) {
      for (int k2 = 0; k2 < row.length; k2++) {
        row[k2] /= subsets;
      }
    }
    return counts;
  }

  private static long enumerate(
      int[] tiles, int from, int left, int k1, int k2, int[] first, int[] second, double[][] out) {
    if (left == 0) {
      out[k1][k2]++;
      return 1;
    }
    long subsets = 0;
    for (int index = from; index <= tiles.length - left; index++) {
      int kind = tiles[index];
      subsets += enumerate(tiles, index + 1, left - 1, k1 + first[kind], k2 + second[kind], first, second, out);
    }
    return subsets;
  }
}

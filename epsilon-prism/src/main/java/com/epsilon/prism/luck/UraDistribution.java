package com.epsilon.prism.luck;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 立直の和了での裏ドラの数の分布を求める。n 枚の裏ドラ表示牌は、誰にも見えていない牌（プール）から重複なしに引いた n 枚と同じ分布に従う
 * （多変量超幾何分布）。和了者 w の裏ドラの数は、引いた表示牌の種類 t ごとに、その表示牌で増える裏ドラの数 g_w(t) を足したもの。
 *
 * <p>表示牌 1 枚で増える裏ドラの数が同じ種類は、枚数を足して 1 つの組にまとめても分布は変わらない。和了者が 1 人ならその数（0〜4）ごとに、
 * 2 人なら 2 人分の数の組ごとにプールをまとめる。組ごとに何枚引くかの動的計画法で、引いた枚数と裏ドラの数の同時分布を求める。
 *
 * <p>使い回す作業領域で、求めた分布は次の呼び出しまで有効。メモリを確保しない。同時に使えるのは 1 つのスレッドだけ。
 */
public final class UraDistribution {

  /** 裏ドラ表示牌の最大の枚数。 */
  public static final int MAX_INDICATORS = WinContext.MAX_INDICATORS;

  /** 表示牌 1 枚で増える裏ドラの数の最大（指す種類の、純手牌・面子・抜いた北の枚数）。 */
  public static final int MAX_COUNT = Tiles.COPIES;

  /** 和了者 1 人の裏ドラの数の最大。 */
  public static final int MAX_URA = MAX_INDICATORS * MAX_COUNT;

  private static final int LEVELS = MAX_COUNT + 1;
  private static final int SIDE = MAX_URA + 1;
  private static final int PLANE = SIDE * SIDE;

  /** プールの枚数の上限（全てのルールの牌の枚数の最大）。 */
  private static final int MAX_POOL =
      Arrays.stream(Variant.values()).mapToInt(variant -> variant.tiles().size()).max().orElseThrow();

  /** [c][j] → c 枚から j 枚を選ぶ組み合わせの数（j は表示牌の枚数まで）。 */
  private static final double[][] BINOMIAL = binomials();

  /** [組] → プールの枚数。組の番号は、1 人目の増える数 × LEVELS ＋ 2 人目の増える数。 */
  private final int[] grouped = new int[LEVELS * LEVELS];

  /** [引いた枚数][1 人目の裏ドラの数][2 人目の裏ドラの数] → 組み合わせの数。求め終えると、引いた枚数が表示牌の枚数の所だけを確率に直す。 */
  private final double[] ways = new double[(MAX_INDICATORS + 1) * PLANE];

  private int draws;
  private int maxFirst;
  private int maxSecond;

  /**
   * 和了者 1 人の分布を求める。
   *
   * @param pool [種類] → 誰にも見えていない枚数
   * @param count [表示牌の種類] → その表示牌で増える裏ドラの数（0〜4）
   * @param draws 裏ドラ表示牌の枚数（1〜5、プールの枚数以下）
   * @throws IllegalArgumentException 枚数か裏ドラの数が範囲の外のとき
   */
  public void single(int[] pool, int[] count, int draws) {
    compute(pool, count, null, draws);
  }

  /**
   * 和了者 2 人（ダブロン）の同時分布を求める。2 人は同じ裏ドラ表示牌で数える。
   *
   * @param pool [種類] → 誰にも見えていない枚数
   * @param first [表示牌の種類] → 1 人目が数える裏ドラの数（0〜4）
   * @param second [表示牌の種類] → 2 人目が数える裏ドラの数（0〜4）
   * @param draws 裏ドラ表示牌の枚数（1〜5、プールの枚数以下）
   * @throws IllegalArgumentException 枚数か裏ドラの数が範囲の外のとき
   */
  public void joint(int[] pool, int[] first, int[] second, int draws) {
    compute(pool, first, second, draws);
  }

  /** 1 人目の裏ドラの数の上限。これより大きい数の確率は 0。 */
  public int maxFirst() {
    return maxFirst;
  }

  /** 2 人目の裏ドラの数の上限。これより大きい数の確率は 0。1 人の分布では 0。 */
  public int maxSecond() {
    return maxSecond;
  }

  /** 1 人の分布で、裏ドラの数が k になる確率。 */
  public double probability(int k) {
    return probability(k, 0);
  }

  /**
   * 1 人目の裏ドラの数が k1、2 人目が k2 になる確率。1 人の分布では、k2 = 0 のときの値が 1 人目の周辺確率になる。
   *
   * @return 確率。範囲の外の数なら 0
   */
  public double probability(int k1, int k2) {
    if (k1 < 0 || k1 > maxFirst || k2 < 0 || k2 > maxSecond) {
      return 0;
    }
    return ways[draws * PLANE + k1 * SIDE + k2];
  }

  private void compute(int[] pool, int[] first, int[] second, int draws) {
    if (draws < 1 || draws > MAX_INDICATORS) {
      throw new IllegalArgumentException("Ura indicators must number 1 to " + MAX_INDICATORS + ": " + draws);
    }
    Arrays.fill(grouped, 0);
    int total = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      int tiles = pool[kind];
      if (tiles == 0) {
        continue;
      }
      if (tiles < 0) {
        throw new IllegalArgumentException("Negative pool " + tiles + " for kind " + kind);
      }
      grouped[level(first, kind) * LEVELS + (second == null ? 0 : level(second, kind))] += tiles;
      total += tiles;
    }
    if (total < draws || total > MAX_POOL) {
      throw new IllegalArgumentException("A pool of " + total + " tiles cannot yield " + draws + " ura indicators");
    }
    this.draws = draws;
    Arrays.fill(ways, 0, (draws + 1) * PLANE, 0);
    ways[0] = 1;
    int reachFirst = 0;
    int reachSecond = 0;
    for (int group = 0; group < grouped.length; group++) {
      int tiles = grouped[group];
      if (tiles == 0) {
        continue;
      }
      int v1 = group / LEVELS;
      int v2 = group % LEVELS;
      // 引いた枚数の多い方から書くので、同じ配列の中で、前の組までの値を上書きされる前に読める
      for (int drawn = draws; drawn >= 1; drawn--) {
        int most = Math.min(drawn, tiles);
        for (int k1 = Math.min(reachFirst + v1 * draws, drawn * MAX_COUNT); k1 >= 0; k1--) {
          for (int k2 = Math.min(reachSecond + v2 * draws, drawn * MAX_COUNT); k2 >= 0; k2--) {
            double sum = ways[drawn * PLANE + k1 * SIDE + k2];
            for (int j = 1; j <= most; j++) {
              int r1 = k1 - j * v1;
              int r2 = k2 - j * v2;
              if (r1 < 0 || r2 < 0) {
                break;
              }
              sum += ways[(drawn - j) * PLANE + r1 * SIDE + r2] * BINOMIAL[tiles][j];
            }
            ways[drawn * PLANE + k1 * SIDE + k2] = sum;
          }
        }
      }
      reachFirst += v1 * draws;
      reachSecond += v2 * draws;
    }
    maxFirst = Math.min(reachFirst, draws * MAX_COUNT);
    maxSecond = Math.min(reachSecond, draws * MAX_COUNT);
    double norm = 1 / BINOMIAL[total][draws];
    int base = draws * PLANE;
    for (int k1 = 0; k1 <= maxFirst; k1++) {
      for (int k2 = 0; k2 <= maxSecond; k2++) {
        ways[base + k1 * SIDE + k2] *= norm;
      }
    }
  }

  private static int level(int[] count, int kind) {
    int value = count[kind];
    if (value < 0 || value > MAX_COUNT) {
      throw new IllegalArgumentException("Ura count " + value + " for kind " + kind + " lies outside [0, " + MAX_COUNT + "]");
    }
    return value;
  }

  private static double[][] binomials() {
    double[][] table = new double[MAX_POOL + 1][MAX_INDICATORS + 1];
    for (int c = 0; c <= MAX_POOL; c++) {
      table[c][0] = 1;
      for (int j = 1; j <= Math.min(c, MAX_INDICATORS); j++) {
        table[c][j] = c == j ? 1 : table[c - 1][j - 1] + table[c - 1][j];
      }
    }
    return table;
  }
}

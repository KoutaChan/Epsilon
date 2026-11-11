package com.epsilon.agent.eval;

import java.util.Arrays;

/**
 * 王者との決闘で使う牌山の数の上限と互角の幅を、牌山 1 つの値の標準偏差 σ から選び直す。上限 N 牌山の決闘が真の差 d を「良い」と判定する確率は、
 * 最後の判定の時点の区間だけを見て Φ(d√N / σ − c) とする（c は最後の時点の臨界値。途中の時点で止まる分は数えない）。幅は、真の差
 * {@value #REACH} × 幅をこの確率 {@value #POWER} 以上で見つけられるように決める。
 */
public final class DuelSizing {

  /** 見つける確率の目標。 */
  public static final double POWER = 0.95;

  /** 見つけたい真の差が、幅の何倍か。 */
  public static final double REACH = 1.5;

  private DuelSizing() {}

  /**
   * @param result 終わった決闘の結果
   * @return 判定に使った量の、牌山 1 つの値の標準偏差（平均の標準誤差 × √牌山の数）
   */
  public static double sigma(DuelResult result) {
    Estimate estimate = result.score() == Score.RANK ? result.rank() : result.utility();
    return estimate.error() * Math.sqrt(result.walls());
  }

  /**
   * @param values 値（1 つ以上）
   * @return 中央値。数が偶数なら、真ん中の 2 つの平均
   */
  public static double median(double[] values) {
    if (values.length == 0) {
      throw new IllegalArgumentException("A median needs a value");
    }
    double[] sorted = values.clone();
    Arrays.sort(sorted);
    int middle = sorted.length / 2;
    return sorted.length % 2 == 1 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
  }

  /**
   * @param sequential 判定の時点の間隔と有意水準の消費関数
   * @param walls 上限の牌山の数（判定の間隔の正の倍数）
   * @param sigma 牌山 1 つの値の標準偏差（正）
   * @param difference 真の差（候補が良い向き）
   * @return 最後の判定の時点で区間の下端が 0 を超える確率 Φ(d√N / σ − c)
   */
  public static double detection(Sequential sequential, int walls, double sigma, double difference) {
    requireSigma(sigma);
    double[] critical = sequential.boundaries(walls);
    return Normal.upper(critical[critical.length - 1] - difference * Math.sqrt(walls) / sigma);
  }

  /**
   * 決闘 1 回で打つ半荘の上限に収まる、最も多い牌山の数を上限にする。その上限で、真の差 {@value #REACH} × 幅を確率 {@value #POWER} で
   * 見つけられる幅を求める。幅は (c + z) σ / ({@value #REACH} √N)（z は標準正規分布の上側 1 − {@value #POWER} の点）。
   *
   * @param sequential 判定の時点の間隔と有意水準の消費関数
   * @param sigma 牌山 1 つの値の標準偏差（正）
   * @param gamesPerWall 牌山 1 つで打つ半荘の数
   * @param games 決闘 1 回で打つ半荘の上限
   * @return 上限（判定の間隔の倍数で、判定の時点は {@value TableSpec#MAX_LOOKS} 回まで）と幅
   * @throws IllegalArgumentException 判定の間隔 1 回分の半荘だけで上限を超えるとき
   */
  public static Choice choose(Sequential sequential, double sigma, int gamesPerWall, long games) {
    requireSigma(sigma);
    if (gamesPerWall < 1) {
      throw new IllegalArgumentException("A wall plays at least one game, not " + gamesPerWall);
    }
    long perLook = (long) sequential.checkEvery() * gamesPerWall;
    long looks = Math.min(TableSpec.MAX_LOOKS, games / perLook);
    if (looks < 1) {
      throw new IllegalArgumentException(
          "One look of " + sequential.checkEvery() + " walls plays " + perLook + " games, more than " + games);
    }
    int walls = Math.toIntExact(looks * sequential.checkEvery());
    double[] critical = sequential.boundaries(walls);
    double margin = (critical[critical.length - 1] + Normal.upperPoint(1 - POWER)) * sigma / (REACH * Math.sqrt(walls));
    return new Choice(walls, margin);
  }

  private static void requireSigma(double sigma) {
    if (!(sigma > 0 && Double.isFinite(sigma))) {
      throw new IllegalArgumentException("Sigma must be positive and finite: " + sigma);
    }
  }

  /**
   * @param walls 上限の牌山の数
   * @param margin 互角の幅（判定に使う量の単位）
   */
  public record Choice(int walls, double margin) {}
}

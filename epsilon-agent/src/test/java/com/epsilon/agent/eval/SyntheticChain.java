package com.epsilon.agent.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * 答えの分かる合成の系譜を作るテスト用のヘルパー。点 v0, v1, … の真の強さ θ は −0.002 × v で、v0 が基準。辺は次の 3 種類。
 *
 * <ul>
 *   <li>1 つ前の点とのラダー（v − 1）。牌山は辺ごとに別で、牌山の値は「真の差 ＋ 0.2 × 標準正規乱数」。
 *   <li>8 つ前の点とのラダー（v ≥ 8 のとき）。作り方は 1 つ前とのラダーと同じ。
 *   <li>基準との辺（v ≥ 2 のとき）。牌山の値は「真の差 ＋ 牌山の運 ＋ 0.1 × 標準正規乱数」で、牌山の運は基準との辺の全てで共通。
 * </ul>
 */
final class SyntheticChain {

  static final String ANCHOR = "v0";

  private static final double LADDER_NOISE = 0.2;
  private static final double BASELINE_NOISE = 0.1;

  private SyntheticChain() {}

  /** @return 点 v の真の θ */
  static double theta(int v) {
    return -0.002 * v;
  }

  /** @return 点の名前（v12 など）から取り出した番号 */
  static int index(String label) {
    return Integer.parseInt(label.substring(1));
  }

  /**
   * 合成の系譜の辺を作る。
   *
   * @param nodes 点の数（基準を含む）
   * @param walls 辺ごとの牌山の数
   * @param sharedLuck 基準との辺が共有する牌山の運の標準偏差。0 なら基準との辺どうしは独立
   * @param labelShared 基準との辺に同じ牌山の種を付けるか。付けなければ、当てはめでは独立な辺として扱われる
   * @param seed 乱数の種
   * @return 作った辺
   */
  static List<Ratings.Edge> edges(int nodes, int walls, double sharedLuck, boolean labelShared, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    double[] luck = new double[walls];
    for (int i = 0; i < walls; i++) {
      luck[i] = sharedLuck * random.nextGaussian();
    }
    long baselineSeed = seed * 1_000_003L;
    long nextSeed = baselineSeed + 1;
    List<Ratings.Edge> edges = new ArrayList<>();
    for (int v = 1; v < nodes; v++) {
      edges.add(ladder(v, v - 1, walls, random, nextSeed++));
      if (v >= 8) {
        edges.add(ladder(v, v - 8, walls, random, nextSeed++));
      }
      if (v >= 2) {
        double[] values = new double[walls];
        for (int i = 0; i < walls; i++) {
          values[i] = theta(v) - theta(0) + luck[i] + BASELINE_NOISE * random.nextGaussian();
        }
        edges.add(new Ratings.Edge("v" + v, ANCHOR, labelShared ? baselineSeed : nextSeed++, values));
      }
    }
    return edges;
  }

  private static Ratings.Edge ladder(int a, int b, int walls, SplittableRandom random, long wallSeed) {
    double[] values = new double[walls];
    for (int i = 0; i < walls; i++) {
      values[i] = theta(a) - theta(b) + LADDER_NOISE * random.nextGaussian();
    }
    return new Ratings.Edge("v" + a, "v" + b, wallSeed, values);
  }

  /**
   * 乱数の種を変えて何度も当て直し、点ごとに θ のずれと標準誤差を集める。基準との辺の牌山の運は標準偏差 0.2 にする。
   *
   * @param nodes 点の数（基準を含む）
   * @param walls 辺ごとの牌山の数
   * @param repeats 当て直す回数
   * @param labelShared 基準との辺に同じ牌山の種を付けるか
   * @param seed 1 回目の乱数の種。回ごとに 1 ずつ足す
   * @return 点の番号ごとの集計
   */
  static Spread[] spread(int nodes, int walls, int repeats, boolean labelShared, long seed) {
    Spread[] spread = new Spread[nodes];
    for (int v = 0; v < nodes; v++) {
      spread[v] = new Spread();
    }
    for (int repeat = 0; repeat < repeats; repeat++) {
      Ratings ratings = Ratings.fit(edges(nodes, walls, 0.2, labelShared, seed + repeat), ANCHOR);
      for (Ratings.Rating rating : ratings.ratings()) {
        int v = index(rating.label());
        spread[v].add(rating.value() - theta(v), rating.error());
      }
    }
    return spread;
  }

  /** 1 つの点について、当て直しごとの θ のずれと標準誤差を集計する。 */
  static final class Spread {
    private final WallMean deviation = new WallMean();
    private double errors;

    void add(double value, double error) {
      deviation.add(value);
      errors += error;
    }

    /** @return ずれの平均 */
    double bias() {
      return deviation.mean();
    }

    /** @return ずれの標準偏差（シミュレーションで測った標準誤差） */
    double simulated() {
      return deviation.error() * Math.sqrt(deviation.walls());
    }

    /** @return 頑健な標準誤差の平均 */
    double reported() {
      return errors / deviation.walls();
    }

    /** @return ずれの平均の標準誤差 */
    double biasError() {
      return deviation.error();
    }
  }
}

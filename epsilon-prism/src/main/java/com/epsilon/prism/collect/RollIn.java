package com.epsilon.prism.collect;

import com.epsilon.prism.core.Greedy;

/**
 * 中立ロールインの逸脱の分布 q(·|o)。台は候補集合 C(o)（{@link CandidateSet#support}）で、C の中の s の順位 r（0 が最大。比べ方は
 * {@link Greedy#before}: 同点は p0 の大きい手、それも同じなら番号の小さい手を上）に沿った重みに一様な床 ε_q を混ぜる。
 *
 * <pre>
 * q(c) = (1 − ε_q)·0.5^r(c) / Σ_{k=0}^{|C|−1} 0.5^k + ε_q / |C|      （c ∈ C）
 * </pre>
 *
 * d は硬貨 u（{@link CollectContext#deviationCoin}）で、C を順位の順に並べた累積分布の逆関数で引く（u が累積を初めて下回る手）。d は打った手
 * played と同じでもよい。傾向は p = β_c(o)·q(d|o)。
 */
final class RollIn {
  private RollIn() {}

  /**
   * 逸脱。
   *
   * @param action d の合法手の位置
   * @param probability q(d|o)
   */
  record Deviation(int action, float probability) {}

  /**
   * q を台の順（support の順）に返す。
   *
   * @param support 候補集合 C(o)（昇順）
   * @param scores 合法手ごとの s
   * @param prior 合法手ごとの p0（s の同点を割る。null なら番号で割る）
   */
  static double[] distribution(int[] support, float[] scores, float[] prior, double epsilonQ) {
    int m = support.length;
    if (m == 0) throw new IllegalArgumentException("Roll-in support is empty");
    if (!(epsilonQ >= 0 && epsilonQ <= 1))
      throw new IllegalArgumentException("epsilonQ must be in [0, 1]: " + epsilonQ);
    int[] order = CandidateSet.ranking(support, scores, prior);
    double normalizer = 0;
    for (int rank = 0; rank < m; rank++) normalizer += Math.scalb(1.0, -rank);
    double[] q = new double[m];
    for (int rank = 0; rank < m; rank++)
      q[order[rank]] = (1 - epsilonQ) * Math.scalb(1.0, -rank) / normalizer + epsilonQ / m;
    return q;
  }

  /** d ~ q(·|o) を硬貨 coin（[0, 1)）で引く。 */
  static Deviation draw(
      int[] support, float[] scores, float[] prior, double epsilonQ, double coin) {
    double[] q = distribution(support, scores, prior, epsilonQ);
    int[] order = CandidateSet.ranking(support, scores, prior);
    double cumulative = 0;
    for (int rank = 0; rank < order.length; rank++) {
      int at = order[rank];
      cumulative += q[at];
      if (coin < cumulative) return new Deviation(support[at], (float) q[at]);
    }
    // 丸めで累積が 1 に届かなかったときは順位の最も低い手
    int last = order[order.length - 1];
    return new Deviation(support[last], (float) q[last]);
  }
}

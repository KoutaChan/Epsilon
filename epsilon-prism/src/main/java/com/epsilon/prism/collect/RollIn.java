package com.epsilon.prism.collect;

import com.epsilon.prism.core.DeviationTarget;
import com.epsilon.prism.core.Greedy;
import java.util.Arrays;

/**
 * 中立ロールインで逸脱する手の分布 q(·|o)。台 S は候補集合 C(o)（{@link CandidateSet#support}）か、そこから打った手を除いた集合
 * （{@link #support}）。S の中での s の順位 r に沿った重みに、一様分布を ε_q だけ混ぜる。順位は 0 が最大で、比べ方は {@link Greedy#before}
 * に従う（同点なら p0 の大きい手、それも同じなら番号の小さい手を上にする）。
 *
 * <pre>
 * q(c) = (1 − ε_q)·0.5^r(c) / Σ_{k=0}^{|S|−1} 0.5^k + ε_q / |S|      （c ∈ S）
 * </pre>
 *
 * d は乱数 u（{@link CollectContext#deviationCoin}）から、S を順位の順に並べた累積分布の逆関数で引く。u が累積を初めて下回る手が d に
 * なる。台が C(o) なら、d は打った手 played と同じでもよい。傾向スコアは p = β_c(o)·q(d|o)。
 */
final class RollIn {
  private RollIn() {}

  /**
   * 引いた逸脱。
   *
   * @param action d が合法手の何番目か
   * @param probability q(d|o)
   */
  record Deviation(int action, float probability) {}

  /**
   * q の台を返す。順位の同点は番号で決めるので、同値な打牌は、組の中で番号の最も小さい手で渡す（どちらの手を打っても、引く組が
   * 変わらないように）。
   *
   * @param candidates 候補集合 C(o)（昇順）
   * @param played 打った手の位置
   * @param target 逸脱を引く手の集合
   * @return {@link DeviationTarget#ANY} なら {@code candidates} そのもの。{@link DeviationTarget#OTHER} なら、打った手を除いた昇順の
   *     新しい配列
   * @throws IllegalStateException 打った手のほかに候補が無いとき
   */
  static int[] support(int[] candidates, int played, DeviationTarget target) {
    if (target == DeviationTarget.ANY) return candidates;
    int[] others = new int[candidates.length];
    int size = 0;
    for (int a : candidates) if (a != played) others[size++] = a;
    if (size == 0) throw new IllegalStateException("No candidate other than the played action " + played);
    return size == candidates.length ? others : Arrays.copyOf(others, size);
  }

  /**
   * q を台 S と同じ順に返す。
   *
   * @param support 台 S（昇順）
   * @param scores 合法手ごとの s
   * @param prior 合法手ごとの p0（s が同点のときに順位を決める。null なら番号で決める）
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

  /** d ~ q(·|o) を、[0, 1) の一様乱数 {@code coin} で引く。 */
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
    // 丸めで累積が 1 に届かなかったときは、順位が最も低い手にする
    int last = order[order.length - 1];
    return new Deviation(support[last], (float) q[last]);
  }
}

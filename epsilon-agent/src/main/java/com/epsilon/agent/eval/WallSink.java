package com.epsilon.agent.eval;

/** 評価の卓（{@link Arena#play}）から、牌山ごとの結果を牌山の番号の順に受け取る。{@link #accept} を同時に呼ぶスレッドは 1 つだけ。 */
@FunctionalInterface
public interface WallSink {

  /**
   * 1 つの牌山の半荘が全て終わったときに呼ばれる。
   *
   * @param wall 牌山の番号
   * @param scores 半荘ごと・席ごとの最終の点（[牌山の中の半荘][席]）。半荘は {@link ArenaPlan#placement} の順。呼び出しの後は保持しない
   * @param stats 半荘ごとの打ち方の内訳（スロットごとに数える。半荘の順）。呼び出しの後は保持しない
   * @return false を返すと、卓はまだ始めていない牌山を始めず、これより後の牌山も渡さない
   */
  boolean accept(long wall, int[][] scores, PlayStats[] stats);

  /**
   * 卓がまだ始めない最初の牌山の番号。卓はこれより前の番号の牌山だけを始める。卓のドライバのスレッドから呼ばれる。返す値は前より下げない。
   * 走っている半荘が全て終わるまでに値が上がらないと、卓は止まったとみなして失敗する。
   */
  default long startBefore() {
    return Long.MAX_VALUE;
  }

  /**
   * 卓の外から打ち切りを求めたかを返す。true を返した後、卓はまだ始めていない牌山を始めず、牌山も渡さず、走っている半荘が終われば終わる。
   * どのスレッドからも呼ばれる。
   */
  default boolean stopped() {
    return false;
  }
}

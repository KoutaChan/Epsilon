package com.epsilon.prism.core;

/**
 * 貪欲の比べ方。方策の argmax（{@code GreedyAnswer}）、候補集合と逸脱の分布の s の順位（{@code CandidateSet}・{@code RollIn}）、体制の代表、
 * 評価の合議の席と天鳳の打ち手の一位で共通に使う。
 *
 * <p>s の大きい手を上にし、s が同じなら規則事前 p0 の大きい手、それも同じなら番号の小さい手を上にする。白紙の網（s ≡ 0）の貪欲は argmax p0 になる。
 * s が全て違う網では p0 は効かない。
 *
 * <p>NaN は比べると偽なので、NaN の手は上にならない（挿入ソートでも順列を返す）。
 */
public final class Greedy {
  private Greedy() {}

  /**
   * a が b より上の順位か。
   *
   * @param scores 合法手ごとの主の値（ふつうは s）
   * @param prior 同点を割る合法手ごとの p0（null なら比べず、番号で割る）
   */
  public static boolean before(int a, int b, float[] scores, float[] prior) {
    if (scores[a] != scores[b]) return scores[a] > scores[b];
    if (prior != null && prior[a] != prior[b]) return prior[a] > prior[b];
    return a < b;
  }

  /** 先頭 n 手の一位（{@link #before} の比べ方）。 */
  public static int argmax(float[] scores, float[] prior, int n) {
    int best = 0;
    for (int a = 1; a < n; a++) if (before(a, best, scores, prior)) best = a;
    return best;
  }
}

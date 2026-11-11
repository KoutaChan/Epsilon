package com.epsilon.prism.core;

/**
 * 貪欲に手を選ぶときの比べ方。方策の argmax（{@code GreedyAnswer}）、候補集合と逸脱の分布での s の順位（{@code CandidateSet}・
 * {@code RollIn}）、体制の代表（{@code RegimeRepresentatives}）、評価の合議の席、天鳳の打ち手がいちばん上の手を選ぶときに、共通で使う。
 *
 * <p>s の大きい手を上にする。s が同じならルールベースの事前分布 p0 の大きい手を、それも同じなら番号の小さい手を上にする。初期化したままの
 * ネットワーク（s ≡ 0）では、貪欲に選ぶ手は argmax p0 になる。s が全て違うネットワークでは p0 は効かない。同値な打牌（{@link Twins}）は
 * s も p0 も同じなので、合法手の並びで先に来る手出しが上になる。実際に打つ方は {@link TwinDiscard} で替える。
 *
 * <p>s が NaN の手は、NaN でないどの手よりも下にする。NaN どうしは s が同じ手として扱う。この比べ方で挿入ソートをしても、結果は元の手の
 * 並べ替えになる。
 */
public final class Greedy {
  private Greedy() {}

  /**
   * 1 つ目の手が 2 つ目の手より上の順位なら true を返す。
   *
   * @param a 比べる手の番号
   * @param b もう一方の手の番号
   * @param scores 合法手ごとに、まず比べる値（ふつうは s）
   * @param prior 同点のときに比べる、合法手ごとの p0。null なら p0 は比べず、番号で決める
   * @return 1 つ目の手が上なら true
   */
  public static boolean before(int a, int b, float[] scores, float[] prior) {
    if (scores[a] > scores[b]) return true;
    if (scores[a] < scores[b]) return false;
    boolean missing = Float.isNaN(scores[a]);
    if (missing != Float.isNaN(scores[b])) return !missing;
    if (prior != null && prior[a] != prior[b]) return prior[a] > prior[b];
    return a < b;
  }

  /** 先頭の {@code n} 手のうち、いちばん上の手の番号を返す。比べ方は {@link #before} と同じ。 */
  public static int argmax(float[] scores, float[] prior, int n) {
    int best = 0;
    for (int a = 1; a < n; a++) if (before(a, best, scores, prior)) best = a;
    return best;
  }
}

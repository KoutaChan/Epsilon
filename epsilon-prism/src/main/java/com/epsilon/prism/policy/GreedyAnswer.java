package com.epsilon.prism.policy;

import com.epsilon.prism.core.Greedy;

/**
 * 網の方策の答え。s を書き、retain の問い合わせと s の最大が同点の問い合わせ（貪欲の同点を p0 で割る。白紙の網では全て）
 * にだけ規則事前 p0（{@link Prior}）を書き、σ を貪欲の one-hot（{@link Greedy#argmax}: s の最大、同点は p0 の大きい手、それも同じなら番号の
 * 小さい手）にする。同点のない問い合わせの貪欲は p0 に依らないので、p0 を作らない。打つ方策は常に貪欲で、床 ε_b は行動方策が混ぜる。
 */
public final class GreedyAnswer {
  private GreedyAnswer() {}

  /**
   * p0 を作るか。
   *
   * @param request 問い合わせ
   * @param scores 合法手ごとの s
   * @return retain の問い合わせか、s の最大が 2 手以上で同じなら true
   */
  public static boolean needsPrior(PolicyRequest request, float[] scores) {
    return request.retain() || topTied(scores, request.legal().size());
  }

  /**
   * s・p0・σ を問い合わせに書く。
   *
   * @param request 問い合わせ
   * @param scores 合法手ごとの s（所有権を受け取る）
   * @param prior 合法手ごとの p0（{@link #needsPrior} が偽なら null）
   */
  public static void write(PolicyRequest request, float[] scores, float[] prior) {
    int n = request.legal().size();
    float[] sigma = new float[n];
    sigma[Greedy.argmax(scores, prior, n)] = 1;
    request.scores = scores;
    request.prior = prior;
    request.sigma = sigma;
  }

  /** 先頭 n 手の s の最大が 2 手以上で同じか。 */
  private static boolean topTied(float[] scores, int n) {
    int best = 0;
    boolean tied = false;
    for (int a = 1; a < n; a++) {
      if (scores[a] > scores[best]) {
        best = a;
        tied = false;
      } else if (scores[a] == scores[best]) {
        tied = true;
      }
    }
    return tied;
  }
}

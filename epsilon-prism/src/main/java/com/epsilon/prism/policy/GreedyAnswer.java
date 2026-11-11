package com.epsilon.prism.policy;

import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.core.Twins;

/**
 * ネットワークの方策の答えを要求に書く。s は常に書く。ルールベースの事前分布 p0（{@link Prior}）は、{@code retain} の要求と、s の最大が
 * 同点の要求にだけ書く（貪欲の同点を p0 で決めるため。初期化したままのネットワークでは全ての要求が同点になる）。σ は貪欲の one-hot にする
 * （{@link Greedy#argmax}: s が最大の手。同点なら p0 の大きい手、それも同じなら番号の小さい手）。同点の無い要求では貪欲が p0 に依らないので、
 * p0 を作らない。打つ方策は常に貪欲で、一様分布を混ぜるのは行動方策の役目。
 *
 * <p>同値な打牌（{@link Twins}）は s も p0 も同じなので、σ は組の中で番号の小さい手に置く。切る方の手は、手を引いた後に {@link TwinDiscard}
 * で替える。σ の側で替えると、床を混ぜた b の上の指数競争で、同値な打牌でない手の当たり方まで変わる。
 */
public final class GreedyAnswer {
  private GreedyAnswer() {}

  /**
   * p0 を作るかを返す。
   *
   * @param request 方策の要求
   * @param scores 合法手ごとの s
   * @return {@code retain} の要求か、s が最大の手が 2 つ以上あるなら true
   */
  public static boolean needsPrior(PolicyRequest request, float[] scores) {
    return request.retain() || topTied(scores, request.legalCount());
  }

  /**
   * s・p0・σ を要求に書く。
   *
   * @param request 方策の要求
   * @param scores 合法手ごとの s（所有権を受け取る）
   * @param prior 合法手ごとの p0（{@link #needsPrior} が false なら null）
   */
  public static void write(PolicyRequest request, float[] scores, float[] prior) {
    int n = request.legalCount();
    float[] sigma = new float[n];
    sigma[Greedy.argmax(scores, prior, n)] = 1;
    request.scores = scores;
    request.prior = prior;
    request.sigma = sigma;
  }

  /** 先頭 n 手のうち、s が最大の手が 2 つ以上あるか。 */
  private static boolean topTied(float[] scores, int n) {
    int best = 0;
    boolean tied = false;
    for (int a = 1; a < n; a++) {
      if (scores[a] > scores[best] || Float.isNaN(scores[best])) {
        best = a;
        tied = false;
      } else if (scores[a] == scores[best]) {
        tied = true;
      }
    }
    return tied;
  }
}

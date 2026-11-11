package com.epsilon.agent.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import java.util.concurrent.CompletableFuture;

/**
 * まとめて推論する方策。1 つのバッチには区分が同じ記録だけを入れる。呼び出し側は、バッチを作る前に {@link #tryAcquire} で推論の
 * 容量を予約し、その予約と一緒にバッチを {@link #choose} へ渡す。
 */
public non-sealed interface BatchedPolicy extends Policy {

  /** 記録を入れるバッチの区分（0 以上）。区分で決まるのはバッチの形（入力の大きさ）だけで、選ぶ手は変わらない。 */
  int admission(Observation observation);

  /** 1 つのバッチに入れられる記録の数の上限。区分ごとに決まる。 */
  int maxBatch(int admission);

  /**
   * 1 つのバッチを推論する容量を、待たずに予約する。既定の実装は容量を数えずに予約を返す。呼び出し側のスレッドで推論する方策は、
   * この既定の実装のままでよい。
   *
   * @param admission 予約するバッチの区分
   * @return 取れた予約。取れなければ、容量が空いたときに完了する通知
   */
  default DecisionInferenceIngress.Attempt tryAcquire(int admission) {
    return DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(this));
  }

  /** 予約済みのバッチだけでは実行先に渡す分が足りず、上限に満たないバッチでも出してほしいときに true。既定は true。 */
  default boolean needsWork() {
    return true;
  }

  /**
   * バッチの全ての記録について手を選び、記録の番号の順に書いてから完了する。予約の所有権は、呼んだ時点で方策へ移る。方策は予約を
   * 実行先へ渡すか、返す。受け付けないバッチで例外を投げるときも、予約は返す。完了するまでは方策がバッチと書く先の配列を使うので、
   * 呼び出し側は触らない。
   *
   * @param batch 手を選ぶ記録のバッチ
   * @param chosen 選んだ手（合法手の何番目か）を書く先
   * @param reservation この方策の {@link #tryAcquire} で取った予約
   * @return 全ての手を書き終えたときに完了する
   */
  CompletableFuture<Void> choose(ObservationBatch batch, int[] chosen, DecisionInferenceIngress reservation);
}

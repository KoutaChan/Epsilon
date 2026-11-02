package com.epsilon.prism.policy;

import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * 1つの版の方策。inline の実装は進行ワーカーでその場で評価し、そうでない実装はドライバがまとめて非同期に評価する。
 *
 * <p>同じ版の評価は呼び出し順や同時実行に依存せず、同じ入力に同じ σ を返す。
 */
public interface PolicyService extends AutoCloseable {
  int version();

  /** 進行ワーカーで {@link #evaluateInline} を呼べるなら true。 */
  boolean inline();

  /**
   * 問い合わせの記録 O（{@link PolicyRequest#capture}。網の入力）だけを読むなら true。true の方策（inline でない）に出す問い合わせは、出す側が
   * 自分のスレッドで O を捕捉してから出す。評価器のスレッドは公開の観測にも局面にも触れない。
   */
  default boolean readsRecord() {
    return false;
  }

  /** σ をその場で書く。状態は呼び出しの間だけ読む。 */
  default void evaluateInline(GameState state, EngineDecisionPoint point, PolicyRequest request) {
    throw new UnsupportedOperationException(getClass().getSimpleName() + " is not inline");
  }

  /** 1バッチの行数の上限（受付の区分 {@link #admissionKey} ごと）。 */
  default int maxBatch(Capacity capacity) {
    return 4096;
  }

  /**
   * 受付の区分。同じ区分の行だけを 1 つのバッチにまとめ、行数の上限は区分で引く（{@link #maxBatch}）。既定は行の容量区分（{@link
   * PolicyRequest#capacity}）で、鳴きの行は別のバッチになる。入力の確保が区分に依らない方策は全ての行を 1 つの区分にまとめてよい。
   */
  default Capacity admissionKey(PolicyRequest request) {
    return request.capacity();
  }

  /** 実行先に仕事がなく、満たないバッチでも出すべきなら true。 */
  default boolean needsWork() {
    return true;
  }

  default DecisionInferenceIngress.Attempt tryAcquireIngress() {
    return DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(this));
  }

  /** 全行の σ を書き終えてから完了する。予約の所有権は呼び出しと同時に移る。 */
  default CompletableFuture<Void> submit(
      List<PolicyRequest> batch, DecisionInferenceIngress ingress) {
    throw new UnsupportedOperationException(getClass().getSimpleName() + " is inline");
  }

  @Override
  default void close() {}
}

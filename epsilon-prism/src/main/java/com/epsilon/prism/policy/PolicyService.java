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

  /** σ（と retain なら payload）をその場で書く。状態は呼び出しの間だけ読む。 */
  default void evaluateInline(GameState state, EngineDecisionPoint point, PolicyRequest request) {
    throw new UnsupportedOperationException(getClass().getSimpleName() + " is not inline");
  }

  /** 1バッチの行数の上限。 */
  default int maxBatch(Capacity capacity) {
    return 4096;
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

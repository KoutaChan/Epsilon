package com.epsilon.prism.model;

import com.epsilon.nano.ai.decision.input.DecisionHostBatch;

/**
 * 差分整合の 1 行（{@link PrismLoss}）。ラベル行（MAIN・ROLLIN）から作る。
 *
 * @param host 1 行の入力
 * @param values 合法手ごとの Y（測っていない手は NaN。測った手が C）
 * @param prior 合法手ごとの p0（全ての合法手。損失は C の上で正規化して使う）
 * @param lambda 行の類の温度 λ_c（ロジット z = s/λ_c、KL の係数も λ_c）
 */
public record LabelExample(DecisionHostBatch host, float[] values, float[] prior, float lambda)
    implements Example {
  public LabelExample {
    if (!(lambda > 0) || !Float.isFinite(lambda))
      throw new IllegalArgumentException("lambda must be positive: " + lambda);
  }
}

package com.epsilon.prism.model;

import com.epsilon.nano.ai.decision.input.DecisionHostBatch;

/**
 * 蒸留の 1 行（{@link Distill}）。
 *
 * @param host 1 行の入力
 * @param target 合法手ごとの教師の分布 t = (softmax(s_A/λ) + softmax(s_B/λ))/2（全ての合法手、和 1）
 * @param lambda 行の類の温度 λ_c（生徒のロジット z = s/λ_c）
 */
public record DistillExample(DecisionHostBatch host, float[] target, float lambda)
    implements Example {
  public DistillExample {
    if (!(lambda > 0) || !Float.isFinite(lambda))
      throw new IllegalArgumentException("lambda must be positive: " + lambda);
  }
}

package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup;
import java.util.List;
import java.util.function.Predicate;

/**
 * FP32 の連続バッファを使った AdamW の勾配準備。勾配のノルムをデバイス上で測って上限で切り、切った回数を見積もる（報告だけ）。連続バッファに
 * 入れる（更新する）重みは trainable で選ぶ（入れない重みは更新も複製への配布もされない）。
 */
final class OptimizerGroup<L extends DataParallelGroup.Lane> extends DataParallelGroup<L> {
  private final float gradientClip;

  /** 勾配のノルムを上限で切った歩数の見積もり（{@link PrismLearner#METRIC_EVERY} 歩に 1 回の標本から）。 */
  private int clipped;

  private int steps;

  OptimizerGroup(
      List<L> lanes,
      DecisionExecutionContext execution,
      float gradientClip,
      Predicate<String> trainable) {
    super(
        lanes,
        ParameterScope.ALL,
        null,
        DataType.FLOAT32,
        execution,
        (scope, name) -> trainable.test(name));
    this.gradientClip = gradientClip;
    recordParameterReadiness();
  }

  @Override
  protected boolean allowsZeroGradientUpdates() {
    return true;
  }

  @Override
  protected void prepareOptimizerGradient(ParameterScope scope, NDArray gradient) {
    steps++;
    if (gradientClip <= 0) return;
    try (NDManager temporary = gradient.getManager().newSubManager()) {
      // 倍率 min(1, 上限 / ノルム) をデバイス上で作って掛ける（ホストへ値を戻さない）
      NDArray squared = gradient.square();
      squared.attach(temporary);
      NDArray norm = squared.sum().sqrt();
      norm.attach(temporary);
      NDArray factor = norm.maximum(gradientClip).pow(-1).mul(gradientClip);
      factor.attach(temporary);
      if (steps % PrismLearner.METRIC_EVERY == 1) {
        float value = norm.getFloat();
        if (!Float.isFinite(value)) throw new IllegalStateException("Nonfinite gradient norm");
        if (value > gradientClip) clipped += PrismLearner.METRIC_EVERY;
      }
      gradient.muli(factor);
    }
  }

  int clipped() {
    return clipped;
  }
}

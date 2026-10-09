package com.epsilon.cfr.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup;
import java.util.List;

/** 方策とスカラー基準値の学習で共有する、FP32の連続バッファを使ったAdamWの勾配準備処理。 */
final class CfrOptimizerGroup<L extends DataParallelGroup.Lane> extends DataParallelGroup<L> {
  private final float gradientClip;
  private double gradientNorm;

  CfrOptimizerGroup(List<L> lanes, DecisionExecutionContext execution, float gradientClip) {
    super(lanes, ParameterScope.ALL, null, DataType.FLOAT32, execution, (scope, name) -> true);
    this.gradientClip = gradientClip;
    recordParameterReadiness();
  }

  double gradientNorm() {
    return gradientNorm;
  }

  @Override
  protected boolean allowsZeroGradientUpdates() {
    return true;
  }

  @Override
  protected void prepareOptimizerGradient(ParameterScope scope, NDArray gradient) {
    try (NDManager temporary = gradient.getManager().newSubManager()) {
      NDArray absolute = gradient.abs();
      absolute.attach(temporary);
      NDArray maximum = absolute.max();
      maximum.attach(temporary);
      double max = maximum.getFloat();
      if (max == 0) {
        gradientNorm = 0;
        return;
      }
      NDArray scaled = gradient.div(max);
      scaled.attach(temporary);
      NDArray squared = scaled.square();
      squared.attach(temporary);
      NDArray sum = squared.sum();
      sum.attach(temporary);
      gradientNorm = max * Math.sqrt(sum.getFloat());
      if (!Double.isFinite(gradientNorm))
        throw new IllegalStateException("Nonfinite CFR gradient norm");
      if (gradientClip > 0 && gradientNorm > gradientClip)
        gradient.muli(gradientClip / gradientNorm);
    }
  }
}

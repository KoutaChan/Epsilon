package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.List;

/**
 * 蒸留の損失（論文 3.4 の「2 本以上を学び、その平均を 1 本に CE で蒸留する」）。行ごとに λ = 行の λ_c、z = s/λ とし、全ての合法手の上で
 *
 * <pre>
 * ℓ = − Σ_{a∈合法手} t(a)·log softmax_合法手(z)(a)        （t は {@link DistillExample#target}）
 * </pre>
 *
 * 行の重みは 1、全ての行が分母に入る（{@link #contributes} は常に真）。全て FP32。合法でない手は where で外す。softmax の安定化に引く行の最大値は
 * 勾配を切る。t = softmax(s/λ) で勾配は 0（損失の値は KL より t のエントロピーだけ大きい）。
 */
public record Distill() implements Objective<DistillExample> {
  @Override
  public boolean contributes(DistillExample example) {
    return true;
  }

  @Override
  public NDArray total(
      NDManager scope, NDArray scores, NDArray legal, List<DistillExample> part, int capacity) {
    int rows = part.size();
    float[] targets = new float[rows * capacity], lambdas = new float[rows];
    for (int row = 0; row < rows; row++) {
      DistillExample example = part.get(row);
      float[] target = example.target();
      if (target.length > capacity)
        throw new IllegalArgumentException(
            "Row " + row + " has " + target.length + " targets for capacity " + capacity);
      for (int a = 0; a < target.length; a++) {
        if (!(target[a] >= 0) || !Float.isFinite(target[a]))
          throw new IllegalArgumentException(
              "Row " + row + " action " + a + " has target " + target[a]);
        targets[row * capacity + a] = target[a];
      }
      lambdas[row] = example.lambda();
    }
    NDArray z = scores.div(scope.create(lambdas, new Shape(rows, 1)));
    MaskedSoftmax all = MaskedSoftmax.of(scope, z, MaskedSoftmax.bool(legal));
    return all.logPi().mul(scope.create(targets, new Shape(rows, capacity))).sum().neg();
  }
}

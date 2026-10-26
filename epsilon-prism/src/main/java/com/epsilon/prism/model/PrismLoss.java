package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.List;

/**
 * 差分整合の損失（論文 3.4 M3、命題 6）。行ごとに C = 値を測った手（values が NaN でない手）、λ = 行の λ_c、z = s/λ とし、
 *
 * <pre>
 * π_C(a)  = softmax_C(z)(a)                         （a ∈ C）
 * p0_C(a) = p0(a) / Σ_{b∈C} p0(b)                    （a ∈ C）
 * ℓ = − Σ_{a∈C} π_C(a)·Y_a
 *     + λ · Σ_{a∈C} π_C(a)·(log π_C(a) − log p0_C(a))
 *     − κ · log Σ_{a∈C} softmax_合法手(z)(a)
 * </pre>
 *
 * <ul>
 *   <li>|C| &lt; 2 の行は損失に入れない（{@link #contributes} が偽、{@link #total} に何も足さない）。行の重みは 1。全て FP32。
 *   <li>C の外の手のロジットには結果（Y）からの勾配が流れない（C の外は掛け算でなく where で外す。κ の項だけが C の外に勾配を流し、その項は Y に依らない）。
 *   <li>第 1・2 項は全ての Y に共通の定数を足しても勾配が変わらない（Σ_a ∇π_C(a) = 0）。V は学ばない。
 *   <li>母集団の最適は π_C ∝ p0_C·exp(Q/λ)（命題 6 (a)）。
 *   <li>softmax の安定化に引く行の最大値は勾配を切る。
 * </ul>
 *
 * <p>p0 は log を取る前に {@link #PRIOR_FLOOR} で下から抑え、C の上の正規化も抑えた値で行う（p0 は正の softmax なので普通は効かない）。
 *
 * @param kappa C の外の質量を下げる項の係数 κ（既定 0.02。0 なら C の外に勾配が流れない）
 */
public record PrismLoss(float kappa) implements Objective<LabelExample> {
  /** log を取る前の p0 の下限。 */
  static final float PRIOR_FLOOR = 1e-30f;

  private static final int[] ACTIONS = {1};

  public PrismLoss {
    if (!(kappa >= 0) || !Float.isFinite(kappa))
      throw new IllegalArgumentException("kappa must be non-negative: " + kappa);
  }

  /** 値を測った手が 2 つ以上の行。 */
  @Override
  public boolean contributes(LabelExample example) {
    int count = 0;
    for (float value : example.values()) if (!Float.isNaN(value) && ++count >= 2) return true;
    return false;
  }

  @Override
  public NDArray total(
      NDManager scope, NDArray scores, NDArray legal, List<LabelExample> part, int capacity) {
    int rows = part.size();
    // ホストで [行, 容量] の教師を並べる: C の印、Y（C の外は 0）、log p0_C（C の外は 0）。λ と行の重みは [行, 1]
    boolean[] measured = new boolean[rows * capacity];
    float[] targets = new float[rows * capacity], logPrior = new float[rows * capacity];
    float[] lambdas = new float[rows], weights = new float[rows];
    for (int row = 0; row < rows; row++) {
      LabelExample example = part.get(row);
      float[] values = example.values(), prior = example.prior();
      if (values.length > capacity || prior.length != values.length)
        throw new IllegalArgumentException(
            "Row "
                + row
                + " has "
                + values.length
                + " values and "
                + prior.length
                + " priors for capacity "
                + capacity);
      int base = row * capacity, count = 0;
      double mass = 0;
      for (int a = 0; a < values.length; a++) {
        if (Float.isNaN(values[a])) continue;
        if (!Float.isFinite(values[a]) || !(prior[a] >= 0) || !Float.isFinite(prior[a]))
          throw new IllegalArgumentException(
              "Row " + row + " action " + a + " has value " + values[a] + " prior " + prior[a]);
        measured[base + a] = true;
        targets[base + a] = values[a];
        mass += Math.max(prior[a], PRIOR_FLOOR);
        count++;
      }
      if (count > 0) {
        double logMass = Math.log(mass);
        for (int a = 0; a < values.length; a++)
          if (measured[base + a])
            logPrior[base + a] = (float) (Math.log(Math.max(prior[a], PRIOR_FLOOR)) - logMass);
      }
      lambdas[row] = example.lambda();
      weights[row] = count >= 2 ? 1 : 0;
    }
    Shape shape = new Shape(rows, capacity), column = new Shape(rows, 1);
    NDArray lambda = scope.create(lambdas, column);
    NDArray z = scores.div(lambda);
    MaskedSoftmax c = MaskedSoftmax.of(scope, z, scope.create(measured, shape));
    // −Σ_C π·Y + λ·Σ_C π·(log π − log p0_C)
    NDArray rowLoss =
        c.pi()
            .mul(scope.create(targets, shape))
            .sum(ACTIONS, true)
            .neg()
            .add(
                c.pi()
                    .mul(c.logPi().sub(scope.create(logPrior, shape)))
                    .sum(ACTIONS, true)
                    .mul(lambda));
    if (kappa > 0) {
      // −κ·log Σ_{a∈C} softmax_合法手(z)(a) = −κ·(logsumexp_C z − logsumexp_合法手 z)
      MaskedSoftmax all = MaskedSoftmax.of(scope, z, MaskedSoftmax.bool(legal));
      rowLoss = rowLoss.sub(c.logSumExp().sub(all.logSumExp()).mul(kappa));
    }
    return rowLoss.mul(scope.create(weights, column)).sum();
  }
}

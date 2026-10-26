package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.prism.learn.DistillFit;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 差分整合（{@link PrismLoss}）と蒸留（{@link Distill}）の損失。小さな手作りのバッチで値と勾配をホストの式と比べ、設計の決め手の性質（Y の定数で
 * 勾配が変わらない・C の外に結果の勾配が流れない・最適は π_C ∝ p0_C·exp(Y/λ)・蒸留は t = softmax(s/λ) で勾配 0）を CPU の FP32 で確かめる。
 */
public class PrismLossTest {
  private static final int CAPACITY = 4;

  /** 行ごとの合法手の数。 */
  private static final int[] LEGAL = {4, 3, 2, 4};

  /** 行ごとの Y（NaN は測っていない手）。行 1 は |C| = 1。 */
  private static final float[][] VALUES = {
    {0.30f, Float.NaN, 0.10f, 0.50f},
    {Float.NaN, 0.20f, Float.NaN},
    {0.00f, 0.40f},
    {0.20f, 0.25f, Float.NaN, Float.NaN}
  };

  /** 行ごとの p0（全ての合法手、和 1、一様でない）。 */
  private static final float[][] PRIORS = {
    {0.4f, 0.3f, 0.2f, 0.1f}, {0.5f, 0.25f, 0.25f}, {0.9f, 0.1f}, {0.1f, 0.6f, 0.2f, 0.1f}
  };

  /** 行ごとの λ_c。 */
  private static final float[] LAMBDAS = {0.1f, 0.2f, 0.05f, 0.5f};

  /** 出力（合法でない枠にもでたらめな値を入れる）。 */
  private static final float[][] SCORES = {
    {0.25f, 0.90f, 0.05f, 0.42f},
    {0.10f, 0.30f, -0.20f, 0.70f},
    {0.12f, 0.07f, 5.0f, -7.0f},
    {0.01f, 0.02f, 3.0f, -1.0f}
  };

  private static List<LabelExample> labels(float[][] values) {
    List<LabelExample> rows = new ArrayList<>();
    for (int r = 0; r < values.length; r++)
      rows.add(new LabelExample(null, values[r], PRIORS[r], LAMBDAS[r]));
    return rows;
  }

  private static float[] flat(float[][] rows) {
    float[] flat = new float[rows.length * CAPACITY];
    for (int r = 0; r < rows.length; r++) System.arraycopy(rows[r], 0, flat, r * CAPACITY, CAPACITY);
    return flat;
  }

  private static NDArray legal(NDManager manager) {
    boolean[] mask = new boolean[LEGAL.length * CAPACITY];
    for (int r = 0; r < LEGAL.length; r++) for (int a = 0; a < LEGAL[r]; a++) mask[r * CAPACITY + a] = true;
    return manager.create(mask, new Shape(LEGAL.length, CAPACITY));
  }

  /** 損失の値と出力への勾配。 */
  private record Evaluated(float value, float[] gradient) {}

  private static <E extends Example> Evaluated evaluate(
      Objective<E> objective, List<E> rows, float[][] scores) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      NDArray s = manager.create(flat(scores), new Shape(scores.length, CAPACITY));
      s.setRequiresGradient(true);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray total = objective.total(manager, s, legal(manager), rows, CAPACITY);
        collector.backward(total);
        return new Evaluated(total.getFloat(), s.getGradient().toFloatArray());
      }
    }
  }

  /** ホストの double での行の損失と勾配（|C| &lt; 2 の行は 0）。 */
  private static double hostRow(int r, float[] s, float[] y, float kappa, double[] gradient) {
    int[] c = new int[y.length];
    int n = 0;
    for (int a = 0; a < y.length; a++) if (!Float.isNaN(y[a])) c[n++] = a;
    if (n < 2) return 0;
    double lambda = LAMBDAS[r], peak = Double.NEGATIVE_INFINITY, sum = 0, mass = 0;
    for (int i = 0; i < n; i++) peak = Math.max(peak, s[c[i]] / lambda);
    for (int i = 0; i < n; i++) {
      sum += Math.exp(s[c[i]] / lambda - peak);
      mass += PRIORS[r][c[i]];
    }
    double[] pi = new double[y.length], logPi = new double[y.length], logQ = new double[y.length];
    double expectedY = 0, expectedKl = 0;
    for (int i = 0; i < n; i++) {
      int a = c[i];
      logPi[a] = s[a] / lambda - peak - Math.log(sum);
      pi[a] = Math.exp(logPi[a]);
      logQ[a] = Math.log(PRIORS[r][a] / mass);
      expectedY += pi[a] * y[a];
      expectedKl += pi[a] * (logPi[a] - logQ[a]);
    }
    double value = -expectedY + lambda * expectedKl;
    for (int i = 0; i < n; i++) {
      int a = c[i];
      gradient[a] = pi[a] * ((logPi[a] - logQ[a] - expectedKl) - (y[a] - expectedY) / lambda);
    }
    if (kappa > 0) {
      int legal = LEGAL[r];
      double legalPeak = Double.NEGATIVE_INFINITY, legalSum = 0;
      for (int a = 0; a < legal; a++) legalPeak = Math.max(legalPeak, s[a] / lambda);
      for (int a = 0; a < legal; a++) legalSum += Math.exp(s[a] / lambda - legalPeak);
      value -= kappa * ((Math.log(sum) + peak) - (Math.log(legalSum) + legalPeak));
      for (int a = 0; a < legal; a++) {
        double all = Math.exp(s[a] / lambda - legalPeak) / legalSum;
        gradient[a] -= kappa / lambda * (pi[a] - all);
      }
    }
    return value;
  }

  /** 値と勾配がホストの式と合い、合法でない枠と |C| &lt; 2 の行の勾配はちょうど 0。κ の項は C の外の合法手に届く。 */
  @Test(groups = "native")
  public void lossMatchesTheAnalyticForm() {
    float kappa = 0.3f;
    Evaluated evaluated = evaluate(new PrismLoss(kappa), labels(VALUES), SCORES);
    double expected = 0;
    for (int r = 0; r < VALUES.length; r++) {
      double[] gradient = new double[CAPACITY];
      expected += hostRow(r, SCORES[r], VALUES[r], kappa, gradient);
      for (int a = 0; a < CAPACITY; a++) {
        float actual = evaluated.gradient()[r * CAPACITY + a];
        if (a >= LEGAL[r] || r == 1)
          Assert.assertEquals(actual, 0f, 0f, "row " + r + " action " + a);
        else
          Assert.assertEquals(
              actual, gradient[a], 1e-3 * Math.max(1, Math.abs(gradient[a])), "row " + r + " action " + a);
      }
    }
    Assert.assertNotEquals(evaluated.gradient()[1], 0f, "the mass term reaches the unmeasured legal action");
    Assert.assertEquals(evaluated.value(), expected, 1e-4 * Math.max(1, Math.abs(expected)));
  }

  /** κ = 0 なら C の外の手の勾配はちょうど 0。Y に行ごとの定数を足しても勾配は変わらない（κ &gt; 0 でも）。 */
  @Test(groups = "native")
  public void outcomeGradientStaysInsideTheCandidates() {
    Evaluated plain = evaluate(new PrismLoss(0f), labels(VALUES), SCORES);
    for (int r = 0; r < VALUES.length; r++)
      for (int a = 0; a < CAPACITY; a++)
        if (a >= VALUES[r].length || Float.isNaN(VALUES[r][a]))
          Assert.assertEquals(plain.gradient()[r * CAPACITY + a], 0f, 0f, "row " + r + " action " + a);
    float[] constants = {0.7f, -1.3f, 2.1f, 0.05f};
    float[][] shifted = new float[VALUES.length][];
    for (int r = 0; r < VALUES.length; r++) {
      shifted[r] = VALUES[r].clone();
      for (int a = 0; a < shifted[r].length; a++) shifted[r][a] += constants[r];
    }
    for (float kappa : new float[] {0f, 0.3f}) {
      float[] base = evaluate(new PrismLoss(kappa), labels(VALUES), SCORES).gradient();
      float[] moved = evaluate(new PrismLoss(kappa), labels(shifted), SCORES).gradient();
      for (int i = 0; i < base.length; i++)
        Assert.assertEquals(moved[i], base[i], 1e-4f, "kappa " + kappa + " entry " + i);
    }
  }

  /** κ = 0 の最適は C の上で s = λ·log p0_C + Y + 行の定数（π_C ∝ p0_C·exp(Y/λ)）で、勾配が 0。 */
  @Test(groups = "native")
  public void optimumIsTheTiltedPrior() {
    float[][] optimum = new float[SCORES.length][];
    for (int r = 0; r < SCORES.length; r++) {
      optimum[r] = SCORES[r].clone();
      double mass = 0;
      for (int a = 0; a < VALUES[r].length; a++) if (!Float.isNaN(VALUES[r][a])) mass += PRIORS[r][a];
      for (int a = 0; a < VALUES[r].length; a++)
        if (!Float.isNaN(VALUES[r][a]))
          optimum[r][a] = (float) (LAMBDAS[r] * Math.log(PRIORS[r][a] / mass) + VALUES[r][a] + 0.3 * r);
    }
    for (float g : evaluate(new PrismLoss(0f), labels(VALUES), optimum).gradient())
      Assert.assertEquals(g, 0f, 1e-4f);
  }

  /** 蒸留は t = softmax(s/λ)（{@link DistillFit#target} に同じ s を 2 つ渡したもの）で勾配が 0、合法でない枠の勾配は常に 0。 */
  @Test(groups = "native")
  public void distillationVanishesAtItsTarget() {
    List<DistillExample> rows = new ArrayList<>();
    List<DistillExample> uniform = new ArrayList<>();
    for (int r = 0; r < SCORES.length; r++) {
      float[] legal = new float[LEGAL[r]];
      System.arraycopy(SCORES[r], 0, legal, 0, LEGAL[r]);
      float[] target = DistillFit.target(legal, legal, LAMBDAS[r]);
      double sum = 0;
      for (float t : target) sum += t;
      Assert.assertEquals(sum, 1.0, 1e-6);
      rows.add(new DistillExample(null, target, LAMBDAS[r]));
      float[] flatTarget = new float[LEGAL[r]];
      java.util.Arrays.fill(flatTarget, 1f / LEGAL[r]);
      uniform.add(new DistillExample(null, flatTarget, LAMBDAS[r]));
    }
    for (float g : evaluate(new Distill(), rows, SCORES).gradient()) Assert.assertEquals(g, 0f, 1e-4f);
    float[] moved = evaluate(new Distill(), uniform, SCORES).gradient();
    for (int r = 0; r < SCORES.length; r++)
      for (int a = LEGAL[r]; a < CAPACITY; a++)
        Assert.assertEquals(moved[r * CAPACITY + a], 0f, 0f, "row " + r + " action " + a);
  }

  /**
   * 教師の分布は 2 本の softmax(s/λ) の平均（{@link DistillFit#target}）で、蒸留の損失は −Σ_合法手 t·log softmax(s/λ)、勾配は (softmax(s/λ) − t)/λ
   * （合法でない枠は 0）。ホストの double の式と比べる。
   */
  @Test(groups = "native")
  public void distillationIsTheCrossEntropyToTheTeacherMean() {
    List<DistillExample> rows = new ArrayList<>();
    double expected = 0;
    double[][] gradients = new double[SCORES.length][CAPACITY];
    for (int r = 0; r < SCORES.length; r++) {
      int n = LEGAL[r];
      double lambda = LAMBDAS[r];
      float[] a = new float[n], b = new float[n];
      for (int k = 0; k < n; k++) {
        a[k] = 0.3f * k - 0.1f * r;
        b[k] = -0.2f * k + 0.05f * r * k;
      }
      float[] target = DistillFit.target(a, b, LAMBDAS[r]);
      double[] pa = softmax(a, lambda), pb = softmax(b, lambda);
      float[] student = new float[n];
      System.arraycopy(SCORES[r], 0, student, 0, n);
      double[] pi = softmax(student, lambda);
      for (int k = 0; k < n; k++) {
        Assert.assertEquals(target[k], 0.5 * (pa[k] + pb[k]), 1e-6, "row " + r + " target " + k);
        expected -= target[k] * Math.log(pi[k]);
        gradients[r][k] = (pi[k] - target[k]) / lambda;
      }
      rows.add(new DistillExample(null, target, LAMBDAS[r]));
    }
    Evaluated evaluated = evaluate(new Distill(), rows, SCORES);
    Assert.assertEquals(evaluated.value(), expected, 1e-4 * Math.max(1, Math.abs(expected)));
    for (int r = 0; r < SCORES.length; r++)
      for (int k = 0; k < CAPACITY; k++)
        Assert.assertEquals(
            evaluated.gradient()[r * CAPACITY + k],
            gradients[r][k],
            1e-3 * Math.max(1, Math.abs(gradients[r][k])),
            "row " + r + " action " + k);
  }

  /** softmax(s/λ)（double、最大値を引く）。 */
  private static double[] softmax(float[] scores, double lambda) {
    double peak = Double.NEGATIVE_INFINITY, sum = 0;
    for (float score : scores) peak = Math.max(peak, score / lambda);
    double[] p = new double[scores.length];
    for (int k = 0; k < scores.length; k++) sum += p[k] = Math.exp(scores[k] / lambda - peak);
    for (int k = 0; k < scores.length; k++) p[k] /= sum;
    return p;
  }
}

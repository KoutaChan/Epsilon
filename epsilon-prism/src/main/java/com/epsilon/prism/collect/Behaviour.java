package com.epsilon.prism.collect;

import java.util.Arrays;

/**
 * 全席の行動方策 b（凍結した π_t の貪欲に一様の床 ε_b を混ぜたもの）。方策の σ（{@link com.epsilon.prism.policy.NetPolicyService} は s の argmax の
 * one-hot）に一様の ε_b を混ぜる。
 *
 * <pre>
 * b(a) = (1 − ε_b)·σ(a)/Σ_b σ(b) + ε_b / n       （Σσ = 0 なら一様）
 * </pre>
 *
 * @param epsilon 一様に混ぜる割合 ε_b（[0, 1)）
 */
public record Behaviour(double epsilon) {
  public Behaviour {
    if (!(epsilon >= 0 && epsilon < 1))
      throw new IllegalArgumentException("epsilon must be in [0, 1): " + epsilon);
  }

  /** σ から b を作る（長さ n）。 */
  public float[] policy(float[] sigma, int n) {
    double sum = 0;
    for (int a = 0; a < n; a++) sum += sigma[a];
    float[] b = new float[n];
    if (sum > 0) {
      double scale = (1 - epsilon) / sum, uniform = epsilon / n;
      for (int a = 0; a < n; a++) b[a] = (float) (scale * sigma[a] + uniform);
    } else {
      Arrays.fill(b, (float) (1.0 / n));
    }
    return b;
  }
}

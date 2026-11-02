package com.epsilon.prism.collect;

import com.epsilon.prism.core.CrnScheme;
import com.epsilon.prism.policy.NetworkPolicy;
import java.util.Arrays;

/**
 * 全席の行動方策 b（凍結した π_t の貪欲に一様の床 ε_b を混ぜたもの）。方策の σ（{@link NetworkPolicy} は s の argmax
 * の one-hot）に一様の ε_b を混ぜる。
 *
 * <pre>
 * b(a) = (1 − ε_b)·σ(a)/Σ_b σ(b) + ε_b / n       （Σσ = 0 なら一様）
 * </pre>
 *
 * <p>探索（{@link #temperature} > 0）: 主経路の判断だけは σ の代わりに s の Boltzmann 分布 softmax(s/τ) に床を混ぜて引く（{@link #explore}）。分岐と
 * 写しの続きは上の b のままなので、ラベル（分岐の値）の意味は変わらず、主経路が通る局面（ラベル行の状態の分布）だけが、網が僅差と見る代案の先へ
 * 広がる。貪欲の自己対戦は代案の先の局面をほとんど通らず、そこの続きの打ち方が上達しないので代案の値が低く出続ける（続きの罠）ことへの手当て。
 *
 * <p>b から引く乱数の使い方は {@link #scheme}（{@link CrnScheme#V1} は b の上の 1 回の競争、{@link CrnScheme#V2} は 2 段の硬貨と種別類 3 つ。
 * どちらも分布は b）。
 *
 * @param epsilon 一様に混ぜる割合 ε_b（[0, 1)）
 * @param temperature 主経路の探索の温度 τ（効用の単位、0 なら探索しない）
 * @param scheme 判断の共通乱数の使い方
 */
public record Behaviour(double epsilon, double temperature, CrnScheme scheme) {
  public Behaviour {
    if (!(epsilon >= 0 && epsilon < 1))
      throw new IllegalArgumentException("epsilon must be in [0, 1): " + epsilon);
    if (!(temperature >= 0)) throw new IllegalArgumentException("temperature must be >= 0: " + temperature);
    if (scheme == null) throw new IllegalArgumentException("scheme must be set");
  }

  /** 主経路の探索の分布: softmax(s/τ) に一様の ε_b を混ぜる（長さ n）。 */
  public float[] explore(float[] scores, int n) {
    double best = Double.NEGATIVE_INFINITY;
    for (int a = 0; a < n; a++) if (!Float.isNaN(scores[a])) best = Math.max(best, scores[a]);
    double[] weights = new double[n];
    double sum = 0;
    for (int a = 0; a < n; a++) {
      // NaN の手は重み 0（貪欲の比べ方 Greedy と同じく上にならない）。全て NaN なら一様
      weights[a] = Float.isNaN(scores[a]) || best == Double.NEGATIVE_INFINITY ? 0 : Math.exp((scores[a] - best) / temperature);
      sum += weights[a];
    }
    if (sum == 0) {
      Arrays.fill(weights, 1);
      sum = n;
    }
    float[] b = new float[n];
    double scale = (1 - epsilon) / sum, uniform = epsilon / n;
    for (int a = 0; a < n; a++) b[a] = (float) (scale * weights[a] + uniform);
    return b;
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

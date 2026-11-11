package com.epsilon.agent.play;

import java.util.Arrays;

/**
 * 卓の全ての席が使う行動方策 b。席の打ち手の分布 σ に、一様分布を割合 ε だけ混ぜる。
 *
 * <pre>
 * b(a) = (1 − ε)·σ(a)/Σ_b σ(b) + ε / n       （Σσ = 0 なら一様）
 * </pre>
 *
 * <p>探索（{@link #temperature} > 0）: 主経路の判断だけは、σ の代わりに手の価値 s の Boltzmann 分布 softmax(s/τ) に一様分布を混ぜて引く
 * （{@link #explore}）。写しと分岐の続きは上の b のままなので、続きの値の意味は変わらない。主経路だけが、打ち手が
 * 僅差と見る代案の先の局面へ広がる。貪欲な自己対戦は代案の先の局面をほとんど通らないので、そこでの続きの打ち方が上達せず、代案の
 * 値が低く出続ける（続きの罠）。探索はその手当て。
 *
 * <p>b から引く乱数の使い方は {@link #scheme} で決める。{@link CrnScheme#SINGLE_RACE} は b の上で 1 回の指数競争をする。
 * {@link CrnScheme#TWO_STAGE_FLOOR} は 2 段の乱数と 3 つの判断のクラスを使う。どちらも分布は b になる。
 *
 * @param epsilon 一様分布を混ぜる割合 ε（[0, 1)）
 * @param temperature 主経路の探索の温度 τ（効用の単位で、0 なら探索しない）
 * @param scheme 判断の共通乱数の使い方
 */
public record Behaviour(double epsilon, double temperature, CrnScheme scheme) {
  public Behaviour {
    if (!(epsilon >= 0 && epsilon < 1))
      throw new IllegalArgumentException("epsilon must be in [0, 1): " + epsilon);
    if (!(temperature >= 0)) throw new IllegalArgumentException("temperature must be >= 0: " + temperature);
    if (scheme == null) throw new IllegalArgumentException("scheme must be set");
  }

  /** 主経路の探索に使う分布（長さ n）。softmax(s/τ) に一様分布を割合 ε だけ混ぜる。 */
  public float[] explore(float[] scores, int n) {
    double best = Double.NEGATIVE_INFINITY;
    for (int a = 0; a < n; a++) if (!Float.isNaN(scores[a])) best = Math.max(best, scores[a]);
    double[] weights = new double[n];
    double sum = 0;
    for (int a = 0; a < n; a++) {
      // NaN の手は重みを 0 にし、最大の値にも数えない。全て NaN なら一様にする
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

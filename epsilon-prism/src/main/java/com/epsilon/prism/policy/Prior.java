package com.epsilon.prism.policy;


/**
 * ルールベースの事前分布 p0(a|o)。全ての合法手にわたる分布で、判断の時点に CPU で作る。ネットワークの方策は、{@code retain} の問い合わせと、
 * s の最大が同点の問い合わせにだけ p0 を書く（{@link GreedyAnswer}）。打牌には向聴と受け入れで重みを付け（{@link RulePrior}）、ほかの手は
 * 同じまとまりの中で一様にする（{@link ActionGroups}）。行は全ての合法手の p0 を持つ。
 *
 * <p>p0 は観測だけで決まる固定の関数で、バージョンにも学習にも依らない。合法手が 1 つなら {1}。使う特徴は、記録 O から求めた打牌の後の
 * 向聴と受け入れの枚数（{@link Encoding}）。
 */
public final class Prior {
  private Prior() {}

  /**
   * 打牌の後の向聴と受け入れの枚数から p0 を求める。
   *
   * @param legal 合法手の行動の番号
   * @param shanten 合法手ごとの、打牌した後の向聴（打牌のまとまりに入らない手の値は読まない）
   * @param ukeire 合法手ごとの、打牌した後の受け入れの枚数（打牌のまとまりに入らない手の値は読まない）
   * @param settings 係数
   * @return 合法手ごとの p0（FP32、和は 1、全て正）
   */
  public static float[] compute(
      int[] legal, float[] shanten, float[] ukeire, PriorSettings settings) {
    int n = legal.length;
    if (n == 1) return new float[] {1f};
    byte[] group = new byte[n];
    ActionGroups.fill(legal, group);
    double[] z = new double[n];
    RulePrior.logits(settings, n, shanten, ukeire, group, z);
    return softmax(z, n);
  }

  /**
   * softmax を求める。FP64 で最大値を引いてから指数を取る。FP32 に丸めて 0 になった手は {@link Float#MIN_NORMAL} にして、全ての確率を正に
   * 保つ。これで log p0 が有限になる（既定の係数では 0 にならない）。
   */
  static float[] softmax(double[] z, int n) {
    double max = Double.NEGATIVE_INFINITY;
    for (int a = 0; a < n; a++) {
      if (!Double.isFinite(z[a])) throw new IllegalStateException("Nonfinite prior logit");
      max = Math.max(max, z[a]);
    }
    double[] e = new double[n];
    double sum = 0;
    for (int a = 0; a < n; a++) sum += e[a] = Math.exp(z[a] - max);
    float[] p = new float[n];
    for (int a = 0; a < n; a++) {
      float value = (float) (e[a] / sum);
      p[a] = value > 0 ? value : Float.MIN_NORMAL;
    }
    return p;
  }
}

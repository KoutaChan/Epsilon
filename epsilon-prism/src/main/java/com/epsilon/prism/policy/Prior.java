package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import java.util.List;

/**
 * 規則事前 p0(a|o)。全ての合法手の上の分布で、CPU で判断の時点に作る（網の方策が retain と s の最大が同点の問い合わせに書く。
 * {@link GreedyAnswer}）。打牌は向聴と受け入れ（{@link RulePrior}）、ほかの手はまとまりの中で一様（{@link ActionGroups}）。行は全ての合法手の p0 を
 * 持つ。
 *
 * <p>p0 は観測だけの固定の関数で、版にも学習にも依らない。合法手が 1 つなら {1}。特徴は記録 O から作る打牌の後の向聴と受け入れの枚数
 * （{@link Encoding}）。
 */
public final class Prior {
  private Prior() {}

  /**
   * 打牌の後の向聴と受け入れの枚数から p0 を求める（FP32、和 1、全て正）。
   *
   * @param legal 合法手
   * @param shanten 手ごとの打牌の後の向聴（打牌のまとまりでない手は読まない）
   * @param ukeire 手ごとの打牌の後の受け入れの枚数（同じ）
   */
  public static float[] compute(
      List<Action> legal, float[] shanten, float[] ukeire, PriorSettings settings) {
    int n = legal.size();
    if (n == 1) return new float[] {1f};
    byte[] group = new byte[n];
    ActionGroups.fill(legal, group);
    double[] z = new double[n];
    RulePrior.logits(settings, n, shanten, ukeire, group, z);
    return softmax(z, n);
  }

  /**
   * softmax（FP64 で最大値を引いてから指数を取る）。FP32 へ丸めて 0 になる手は {@link Float#MIN_NORMAL} にして、全て正に保つ（log p0 を有限に
   * する。既定の係数では起きない）。
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

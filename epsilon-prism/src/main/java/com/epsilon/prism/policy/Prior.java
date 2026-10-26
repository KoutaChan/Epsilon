package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import java.util.List;

/**
 * 規則事前 p0(a|o)（論文 3.4 の錨）。全ての合法手の上の分布で、CPU で判断の時点に作る（{@link RequestEncoder#finish} が retain と s の最大が同点の問い合わせに書く）。
 * 打牌は向聴と受け入れ（{@link RulePrior}）、ほかの手はまとまりの中で一様（{@link ActionGroups}）。行は全ての合法手の p0 を持つ。
 *
 * <p>p0 は観測だけの固定の関数で、版にも学習にも依らない。合法手が 1 つなら {1}。
 */
public final class Prior {
  private Prior() {}

  /**
   * p0 を求める（FP32、和 1、全て正）。
   *
   * @param host 符号化済みの入力（遷移特徴から向聴と受け入れを読む。{@link R0Features}）
   * @param row host の中の行
   * @param legal 合法手（host の行の合法手と同じ並び）
   */
  public static float[] compute(
      DecisionHostBatch host, int row, List<Action> legal, PriorSettings settings) {
    int n = legal.size();
    if (n == 1) return new float[] {1f};
    byte[] group = new byte[n];
    ActionGroups.fill(legal, group);
    float[] shanten = new float[n], ukeire = new float[n];
    for (int a = 0; a < n; a++) {
      if (group[a] != ActionGroups.DISCARD) continue;
      shanten[a] = R0Features.shanten(host, row, a);
      ukeire[a] = R0Features.ukeire(host, row, a);
    }
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

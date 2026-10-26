package com.epsilon.prism.policy;

/**
 * 打牌の規則による事前分布のロジット（CRR の R0 を β = 0・c = 1 で移したもの。CRR の RulePrior と HedgePolicyHead の事前分布と基準測度の部分）。
 * 向聴を減らし受け入れを増やす打牌を優先する。
 *
 * <pre>
 * q(a) = −α·shanten(a) + γ·ukeire(a)                                  （a ∈ D: 打牌のまとまり {@link ActionGroups#DISCARD}）
 * prior(a) = q(a) − [logsumexp_{b∈D} q(b) − log|D|]                    （a ∈ D、|D| ≥ 2）、0（a ∉ D か |D| &lt; 2）
 * z(a) = prior(a) − log m(a)                                          （m(a) は a のまとまりの手の数。{@link ActionGroups#applyBaseMeasure}）
 * p0 = softmax(z)                                                     （{@link Prior}）
 * </pre>
 *
 * 正規化により Σ_{a∈D} e^{prior(a)} = |D| となり、事前分布は打牌のまとまりの中の配分だけを変える。まとまりの総確率は基準測度で決まる（打牌全体・鳴き全体・
 * ほかの 1 手ごとに等しい）。打牌でない手は自分のまとまりの中で一様。特徴は Nano の入力エンコーダーが計算した遷移特徴（{@link R0Features}）。
 */
public final class RulePrior {
  private RulePrior() {}

  /**
   * ロジット z を書く（内部は FP64）。
   *
   * @param n 合法手の数
   * @param shanten 打牌後の向聴（打牌でない位置は読まない）
   * @param ukeire 打牌後の受け入れ枚数（打牌でない位置は読まない）
   * @param group 手のまとまり（{@link ActionGroups#fill}）
   * @param z 長さ n 以上の出力
   */
  public static void logits(
      PriorSettings settings, int n, float[] shanten, float[] ukeire, byte[] group, double[] z) {
    double alpha = settings.alpha(), gamma = settings.gamma();
    int count = 0;
    double max = Double.NEGATIVE_INFINITY;
    for (int a = 0; a < n; a++) {
      z[a] = 0;
      if (group[a] != ActionGroups.DISCARD) continue;
      double q = -alpha * shanten[a] + gamma * ukeire[a];
      if (!Double.isFinite(q))
        throw new IllegalStateException(
            "Nonfinite prior feature shanten=" + shanten[a] + " ukeire=" + ukeire[a]);
      count++;
      max = Math.max(max, q);
    }
    if (count >= 2) {
      double sum = 0;
      for (int a = 0; a < n; a++)
        if (group[a] == ActionGroups.DISCARD)
          sum += Math.exp(-alpha * shanten[a] + gamma * ukeire[a] - max);
      // 打牌のまとまりの中で e^{prior} の和が |D| になるように引く（まとまりの総確率は基準測度だけで決まる）
      double normalizer = max + Math.log(sum) - Math.log(count);
      for (int a = 0; a < n; a++)
        if (group[a] == ActionGroups.DISCARD)
          z[a] = -alpha * shanten[a] + gamma * ukeire[a] - normalizer;
    }
    ActionGroups.applyBaseMeasure(group, n, z);
  }
}

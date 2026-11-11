package com.epsilon.prism.policy;

/**
 * ルールベースの事前分布のロジットを求める。打牌では、向聴を減らし受け入れを増やす手を優先する。
 *
 * <pre>
 * q(a) = −α·shanten(a) + γ·ukeire(a)                                  （a ∈ D: 打牌のまとまり {@link ActionGroups#DISCARD}）
 * prior(a) = q(a) − [logsumexp_{b∈D} q(b) − log|D|]                    （a ∈ D、|D| ≥ 2）、0（a ∉ D か |D| &lt; 2）
 * z(a) = prior(a) − log m(a)                                          （m(a) は a のまとまりの手の数。{@link ActionGroups#applyBaseMeasure}）
 * p0 = softmax(z)                                                     （{@link Prior}）
 * </pre>
 *
 * 正規化で Σ_{a∈D} e^{prior(a)} = |D| になるので、この事前分布は打牌のまとまりの中の配分だけを変える。各まとまりの確率の合計は基準測度で
 * 決まり、打牌全体、鳴き全体、ほかの手 1 つずつが同じ重みになる。打牌でない手は、自分のまとまりの中で一様。使う特徴は、記録 O から求めた
 * 打牌の後の向聴と受け入れの枚数（{@link Prior}）。
 */
public final class RulePrior {
  private RulePrior() {}

  /**
   * ロジット z を求めて書く。計算は FP64 で行う。
   *
   * @param settings 係数
   * @param n 合法手の数
   * @param shanten 合法手ごとの、打牌した後の向聴（打牌でない手の値は読まない）
   * @param ukeire 合法手ごとの、打牌した後の受け入れの枚数（打牌でない手の値は読まない）
   * @param group 合法手ごとのまとまり（{@link ActionGroups#fill}）
   * @param z 書く先（長さ n 以上）
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
      // 打牌のまとまりの中で e^{prior} の和が |D| になるように引く。これでまとまりの確率の合計は基準測度だけで決まる
      double normalizer = max + Math.log(sum) - Math.log(count);
      for (int a = 0; a < n; a++)
        if (group[a] == ActionGroups.DISCARD)
          z[a] = -alpha * shanten[a] + gamma * ukeire[a] - normalizer;
    }
    ActionGroups.applyBaseMeasure(group, n, z);
  }
}

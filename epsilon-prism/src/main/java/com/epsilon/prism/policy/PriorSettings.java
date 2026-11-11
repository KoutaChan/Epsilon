package com.epsilon.prism.policy;

/**
 * ルールベースの事前分布 p0 の係数（{@link RulePrior}）。方策は {@link #STANDARD} の係数で打つ。
 *
 * @param alpha 打牌した後の向聴 1 つごとに下げるロジット（nat）
 * @param gamma 打牌した後の受け入れ 1 枚ごとに上げるロジット（nat）
 */
public record PriorSettings(double alpha, double gamma) {

  /** PRISM の方策の係数。 */
  public static final PriorSettings STANDARD = new PriorSettings(4.0, 0.05);

  public PriorSettings {
    if (!(alpha >= 0) || !(gamma >= 0) || !Double.isFinite(alpha) || !Double.isFinite(gamma))
      throw new IllegalArgumentException("Invalid prior coefficients " + alpha + "/" + gamma);
  }
}

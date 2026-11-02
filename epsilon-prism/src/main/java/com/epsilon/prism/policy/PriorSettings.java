package com.epsilon.prism.policy;

/**
 * 規則事前 p0 の係数（{@link RulePrior}）。
 *
 * @param alpha 打牌後の向聴 1 つあたりの nat（既定 4.0）
 * @param gamma 打牌後の受け入れ 1 枚あたりの nat（既定 0.05）
 */
public record PriorSettings(double alpha, double gamma) {
  public PriorSettings {
    if (!(alpha >= 0) || !(gamma >= 0) || !Double.isFinite(alpha) || !Double.isFinite(gamma))
      throw new IllegalArgumentException("Invalid prior coefficients " + alpha + "/" + gamma);
  }
}

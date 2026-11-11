package com.epsilon.agent.eval;

/**
 * 牌山ごとの値の平均と区間。
 *
 * @param mean 平均
 * @param error 平均の標準誤差
 * @param low 区間の下端
 * @param high 区間の上端
 */
public record Estimate(double mean, double error, double low, double high) {

  /**
   * @param critical 区間の半幅が標準誤差の何倍か
   * @return 平均 ± 臨界値 × 標準誤差 の区間
   */
  public static Estimate of(double mean, double error, double critical) {
    return new Estimate(mean, error, mean - critical * error, mean + critical * error);
  }

  /** @return 符号を反転した値。順位（負が良い）を候補が良い向きの値に直すとき、またその逆に使う */
  public Estimate negate() {
    return new Estimate(-mean, error, -high, -low);
  }
}

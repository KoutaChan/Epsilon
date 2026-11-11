package com.epsilon.agent.eval;

/** 標準正規分布。上側確率は Hart の有理近似（West による倍精度の形）で求め、絶対誤差は 3e-16 以下。 */
final class Normal {

  private static final double INVERSE_SQRT_TWO_PI = 0.3989422804014327;

  private Normal() {}

  /** @return x での確率密度 */
  static double density(double x) {
    return INVERSE_SQRT_TWO_PI * Math.exp(-0.5 * x * x);
  }

  /** @return P(Z > x) */
  static double upper(double x) {
    double a = Math.abs(x);
    double tail;
    if (a > 37) {
      tail = 0;
    } else if (a < 7.07106781186547) {
      double numerator = 3.52624965998911e-02 * a + 0.700383064443688;
      numerator = numerator * a + 6.37396220353165;
      numerator = numerator * a + 33.912866078383;
      numerator = numerator * a + 112.079291497871;
      numerator = numerator * a + 221.213596169931;
      numerator = numerator * a + 220.206867912376;
      double denominator = 8.83883476483184e-02 * a + 1.75566716318264;
      denominator = denominator * a + 16.064177579207;
      denominator = denominator * a + 86.7807322029461;
      denominator = denominator * a + 296.564248779674;
      denominator = denominator * a + 637.333633378831;
      denominator = denominator * a + 793.826512519948;
      denominator = denominator * a + 440.413735824752;
      tail = Math.exp(-0.5 * a * a) * numerator / denominator;
    } else {
      double fraction = a + 0.65;
      fraction = a + 4 / fraction;
      fraction = a + 3 / fraction;
      fraction = a + 2 / fraction;
      fraction = a + 1 / fraction;
      tail = Math.exp(-0.5 * a * a) / fraction / 2.506628274631;
    }
    return x < 0 ? 1 - tail : tail;
  }

  /**
   * @param p 上側確率（0 より大きく 1 より小さい）
   * @return P(Z > z) = p となる z
   */
  static double upperPoint(double p) {
    if (!(p > 0 && p < 1)) {
      throw new IllegalArgumentException("Invalid probability: " + p);
    }
    double low = -40;
    double high = 40;
    while (high - low > 1e-13) {
      double middle = 0.5 * (low + high);
      if (upper(middle) > p) {
        low = middle;
      } else {
        high = middle;
      }
    }
    return 0.5 * (low + high);
  }
}

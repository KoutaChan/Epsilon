package com.epsilon.ai.decision.duel;

/**
 * 事前に固定した標本数について、正規近似を用いた平均値の両側信頼区間を求める。
 *
 * <p>独立な牌山ごとの結果から標本平均と不偏分散を計算し、{@code mean ± z(1-alpha/2) *
 * SE}を求める。区間は観測値の既知の範囲に収める。この区間だけを根拠に予定外の逐次停止をしてはならない。
 */
public final class EpsilonDecisionFixedSampleConfidenceInterval {

  private static final double LOWER_TAIL_APPROXIMATION_LIMIT = 0.02425;
  private static final double[] CENTRAL_NUMERATOR = {
    -39.69683028665376,
    220.9460984245205,
    -275.9285104469687,
    138.3577518672690,
    -30.66479806614716,
    2.506628277459239
  };
  private static final double[] CENTRAL_DENOMINATOR = {
    -54.47609879822406,
    161.5858368580409,
    -155.6989798598866,
    66.80131188771972,
    -13.28068155288572,
    1.0
  };
  private static final double[] TAIL_NUMERATOR = {
    -0.007784894002430293,
    -0.3223964580411365,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783
  };
  private static final double[] TAIL_DENOMINATOR = {
    0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416, 1.0
  };

  private final double observationLowerBound;
  private final double observationUpperBound;
  private int samples;
  private double mean;
  private double squaredDeviation;

  public EpsilonDecisionFixedSampleConfidenceInterval(
      double observationLowerBound, double observationUpperBound) {
    if (!Double.isFinite(observationLowerBound)
        || !Double.isFinite(observationUpperBound)
        || observationLowerBound >= observationUpperBound) {
      throw new IllegalArgumentException("observation bounds must be finite and increasing");
    }
    this.observationLowerBound = observationLowerBound;
    this.observationUpperBound = observationUpperBound;
  }

  public void add(double observation) {
    if (!Double.isFinite(observation)
        || observation < observationLowerBound
        || observation > observationUpperBound) {
      throw new IllegalArgumentException("observation is outside the configured bounds");
    }
    samples++;
    double delta = observation - mean;
    mean += delta / samples;
    squaredDeviation += delta * (observation - mean);
  }

  public int samples() {
    return samples;
  }

  public Estimate estimate(double alpha) {
    if (samples == 0) {
      throw new IllegalStateException("confidence interval has no observations");
    }
    if (!Double.isFinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
      throw new IllegalArgumentException("alpha must be finite and in (0, 1)");
    }
    if (samples == 1) {
      return new Estimate(samples, mean, 0.0, observationLowerBound, observationUpperBound);
    }
    double standardError = Math.sqrt((squaredDeviation / (samples - 1)) / samples);
    double radius = normalCriticalValue(alpha) * standardError;
    return new Estimate(
        samples,
        mean,
        standardError,
        Math.max(observationLowerBound, mean - radius),
        Math.min(observationUpperBound, mean + radius));
  }

  /** 両側誤差率 {@code alpha} に対応する標準正規分布の臨界値。 */
  static double normalCriticalValue(double alpha) {
    if (!Double.isFinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
      throw new IllegalArgumentException("alpha must be finite and in (0, 1)");
    }
    double lowerTailProbability = alpha * 0.5;
    if (lowerTailProbability < LOWER_TAIL_APPROXIMATION_LIMIT) {
      double q = Math.sqrt(-2.0 * Math.log(lowerTailProbability));
      double lowerQuantile = polynomial(q, TAIL_NUMERATOR) / polynomial(q, TAIL_DENOMINATOR);
      return -lowerQuantile;
    }

    double q = 0.5 - lowerTailProbability;
    double r = q * q;
    return polynomial(r, CENTRAL_NUMERATOR) * q / polynomial(r, CENTRAL_DENOMINATOR);
  }

  private static double polynomial(double value, double[] coefficients) {
    double result = coefficients[0];
    for (int index = 1; index < coefficients.length; index++) {
      result = result * value + coefficients[index];
    }
    return result;
  }

  public record Estimate(
      int samples, double mean, double standardError, double lower, double upper) {}
}

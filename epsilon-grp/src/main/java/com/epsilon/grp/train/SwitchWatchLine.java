package com.epsilon.grp.train;

import java.util.Objects;

/**
 * GRP を切り替えた後の監視の 1 行。確定したコホート h ごとに 1 行ある。M_used は h を集めたときのモデル、M_prev はその 1 つ前に使った
 * 別のモデルで、どちらも h では学習していない。値は全て有限で、標準誤差と揺れは 0 以上。
 *
 * @param mseDifference h での MSE(M_used) − MSE(M_prev)
 * @param standardError その差の、半荘をクラスタとした標準誤差
 * @param wobble 2 つの Φ を h の局末で比べた差の RMS（揺れ）
 */
public record SwitchWatchLine(
    int cohort, String usedId, String previousId, double mseDifference, double standardError, double wobble) {

  public SwitchWatchLine {
    Objects.requireNonNull(usedId, "usedId");
    Objects.requireNonNull(previousId, "previousId");
    if (!Double.isFinite(mseDifference) || !(standardError >= 0) || !(wobble >= 0)
        || !Double.isFinite(standardError) || !Double.isFinite(wobble)) {
      throw new IllegalArgumentException(
          "Switch watch values must be finite with non-negative SE and wobble: " + mseDifference + ", "
              + standardError + ", " + wobble);
    }
  }

  /** コホート h の局末での M_used と M_prev の値から、行を作る。 */
  public static SwitchWatchLine of(
      int cohort, String usedId, String previousId, HandEnds ends, HandEndValues used, HandEndValues previous) {
    Paired difference = GrpMetrics.pairedMse(ends, previous, used);
    double wobble = GrpMetrics.wobble(ends, used, previous);
    return new SwitchWatchLine(cohort, usedId, previousId, difference.mean(), difference.standardError(), wobble);
  }
}

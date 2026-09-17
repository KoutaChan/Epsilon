package com.epsilon.ai.grp;

/** 検証指標を比較し、新しく学習したGRPモデルを採用するかを判定する。 */
public final class EpsilonGrpValidationGate {

  private static final float MAX_MARGINAL_SUM_ERROR = 1.0e-5f;
  private static final float NLL_EQUIVALENCE_BAND = 1.0e-4f;
  private static final float BRIER_EQUIVALENCE_BAND = 1.0e-5f;
  private static final float MAX_ORDINAL_RPS_REGRESSION = 1.0e-4f;
  private static final float MAX_CALIBRATION_REGRESSION = 0.02f;

  private EpsilonGrpValidationGate() {}

  /**
   * 負の対数尤度（NLL）を主基準とし、順位付き確率スコア（RPS）や期待較正誤差（ECE）が許容値を超えて悪化した候補は採用しない。
   *
   * @param candidate 新しく学習した候補指標
   * @param incumbent 現在の採用モデルの指標
   * @return 候補を採用すべきなら {@code true}
   */
  public static boolean better(
      EpsilonGrpInference.EvalMetrics candidate, EpsilonGrpInference.EvalMetrics incumbent) {
    if (!structurallyValid(candidate)) {
      return false;
    }
    if (!structurallyValid(incumbent)) {
      return true;
    }
    if (candidate.ordinalRps() > incumbent.ordinalRps() + MAX_ORDINAL_RPS_REGRESSION
        || candidate.macroEce() > incumbent.macroEce() + MAX_CALIBRATION_REGRESSION
        || candidate.lastRankEce() > incumbent.lastRankEce() + MAX_CALIBRATION_REGRESSION) {
      return false;
    }
    float nllDelta = candidate.marginalNll() - incumbent.marginalNll();
    if (nllDelta < -NLL_EQUIVALENCE_BAND) {
      return true;
    }
    if (nllDelta > NLL_EQUIVALENCE_BAND) {
      return false;
    }
    float brierDelta = candidate.marginalBrier() - incumbent.marginalBrier();
    if (brierDelta < -BRIER_EQUIVALENCE_BAND) {
      return true;
    }
    if (brierDelta > BRIER_EQUIVALENCE_BAND) {
      return false;
    }
    if (candidate.macroEce() != incumbent.macroEce()) {
      return candidate.macroEce() < incumbent.macroEce();
    }
    return candidate.lastRankEce() < incumbent.lastRankEce();
  }

  /**
   * 指標が有限で、周辺確率の行和・列和契約を満たすかを調べる。
   *
   * @param metrics 検証する評価指標
   * @return 採用比較へ使用できるなら {@code true}
   */
  public static boolean structurallyValid(EpsilonGrpInference.EvalMetrics metrics) {
    return metrics.examples() > 0
        && Float.isFinite(metrics.marginalNll())
        && Float.isFinite(metrics.marginalBrier())
        && Float.isFinite(metrics.ordinalRps())
        && metrics.ordinalRps() >= 0.0f
        && metrics.ordinalRps() <= 1.0f
        && Float.isFinite(metrics.macroEce())
        && Float.isFinite(metrics.lastRankEce())
        && metrics.invalidProbabilities() == 0
        && Float.isFinite(metrics.maxRowSumError())
        && Float.isFinite(metrics.maxColumnSumError())
        && metrics.maxRowSumError() <= MAX_MARGINAL_SUM_ERROR
        && metrics.maxColumnSumError() <= MAX_MARGINAL_SUM_ERROR;
  }
}

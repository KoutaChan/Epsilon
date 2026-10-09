package com.epsilon.cfr.data;

/**
 * 主経路の行動抽選前に予約した、一つの分岐比較の結果を保持する。行動の包含確率には、主経路での抽選と追加枝での評価をともに反映する。
 * 追加枝の継続経路に対する重みには、分岐の起点で強制した行動を含めない。
 */
public record CfrBranchOutcome(
    long decisionId,
    double mainInclusionProbability,
    int alternativeSlot,
    double alternativeInclusionProbability,
    double alternativeUtility,
    double alternativeLogSuffixWeight) {
  public CfrBranchOutcome {
    if (decisionId < 0
        || !probability(mainInclusionProbability)
        || alternativeSlot < -1
        || !Double.isFinite(alternativeUtility)) {
      throw new IllegalArgumentException("Invalid CFR branch outcome");
    }
    CfrDataChecks.logWeight(alternativeLogSuffixWeight);
    if (alternativeSlot == -1) {
      if (mainInclusionProbability != 1
          || alternativeInclusionProbability != 0
          || alternativeUtility != 0
          || alternativeLogSuffixWeight != Double.NEGATIVE_INFINITY) {
        throw new IllegalArgumentException("Absent CFR alternative must have neutral metadata");
      }
    } else if (!probability(alternativeInclusionProbability)) {
      throw new IllegalArgumentException("Invalid CFR alternative inclusion probability");
    }
  }

  private static boolean probability(double value) {
    return Double.isFinite(value) && value > 0 && value <= 1;
  }
}

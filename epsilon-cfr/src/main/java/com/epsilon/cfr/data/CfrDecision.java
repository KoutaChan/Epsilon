package com.epsilon.cfr.data;

/** 抽選した一つの行動選択を記録する。同時応答の優先順位によって実行されなかった選択も含む。 */
public record CfrDecision(
    long decisionId,
    int seat,
    int[] legalActionIds,
    int selectedSlot,
    double[] targetPolicy,
    double behaviorProbability,
    ObservationPayload observation) {
  public CfrDecision {
    if (decisionId < 0
        || seat < 0
        || seat >= 4
        || !Double.isFinite(behaviorProbability)
        || behaviorProbability <= 0
        || behaviorProbability > 1) {
      throw new IllegalArgumentException("Invalid CFR decision metadata");
    }
    CfrDataChecks.decision(legalActionIds, selectedSlot, targetPolicy, observation);
    legalActionIds = legalActionIds.clone();
    targetPolicy = targetPolicy.clone();
  }

  @Override
  public int[] legalActionIds() {
    return legalActionIds.clone();
  }

  @Override
  public double[] targetPolicy() {
    return targetPolicy.clone();
  }

  public int actionCount() {
    return legalActionIds.length;
  }

  public double selectedTargetProbability() {
    return targetPolicy[selectedSlot];
  }
}

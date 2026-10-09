package com.epsilon.cfr.data;

import java.util.Arrays;
import java.util.Objects;

/** 両方の学習で共有する、変更不可の観測と教師値。後悔学習用と平均方策学習用の保存対象はそれぞれ独立に選ぶ。 */
public record CfrTrainingRow(
    CfrSampleId id,
    int[] legalActionIds,
    int selectedSlot,
    double[] targetPolicy,
    double utility,
    double logRegretWeight,
    double logAverageWeight,
    ObservationPayload observation,
    double[] explicitRegretTargets) {
  public CfrTrainingRow(
      CfrSampleId id,
      int[] legalActionIds,
      int selectedSlot,
      double[] targetPolicy,
      double utility,
      double logRegretWeight,
      double logAverageWeight,
      ObservationPayload observation) {
    this(
        id,
        legalActionIds,
        selectedSlot,
        targetPolicy,
        utility,
        logRegretWeight,
        logAverageWeight,
        observation,
        null);
  }

  public CfrTrainingRow {
    Objects.requireNonNull(id, "id");
    CfrDataChecks.decision(legalActionIds, selectedSlot, targetPolicy, observation);
    if (!Double.isFinite(utility)) throw new IllegalArgumentException("Nonfinite utility");
    CfrDataChecks.logWeight(logRegretWeight);
    CfrDataChecks.logWeight(logAverageWeight);
    legalActionIds = legalActionIds.clone();
    targetPolicy = targetPolicy.clone();
    if (explicitRegretTargets != null) {
      if (explicitRegretTargets.length != legalActionIds.length)
        throw new IllegalArgumentException("CFR regret target shape mismatch");
      for (double target : explicitRegretTargets) {
        if (!Double.isFinite(target))
          throw new IllegalArgumentException("Nonfinite CFR regret target");
      }
      explicitRegretTargets = explicitRegretTargets.clone();
    }
  }

  @Override
  public int[] legalActionIds() {
    return legalActionIds.clone();
  }

  @Override
  public double[] targetPolicy() {
    return targetPolicy.clone();
  }

  @Override
  public double[] explicitRegretTargets() {
    return explicitRegretTargets == null ? null : explicitRegretTargets.clone();
  }

  public int actionCount() {
    return legalActionIds.length;
  }

  public double selectedTargetProbability() {
    return targetPolicy[selectedSlot];
  }

  public double[] regretTargets() {
    if (explicitRegretTargets != null) return explicitRegretTargets.clone();
    double[] targets = new double[legalActionIds.length];
    Arrays.fill(targets, -utility * selectedTargetProbability());
    targets[selectedSlot] += utility;
    return targets;
  }

  /** この行の直列化後のバイト数を返す。外側の保存形式が付けるヘッダーや長さ情報は含めない。 */
  public long serializedBytes() {
    return 65L
        + 12L * legalActionIds.length
        + observation.size()
        + (explicitRegretTargets == null ? 0 : 8L * explicitRegretTargets.length);
  }

  public boolean contentEquals(CfrTrainingRow other) {
    return id.equals(other.id)
        && selectedSlot == other.selectedSlot
        && Double.compare(utility, other.utility) == 0
        && Double.compare(logRegretWeight, other.logRegretWeight) == 0
        && Double.compare(logAverageWeight, other.logAverageWeight) == 0
        && Arrays.equals(legalActionIds, other.legalActionIds)
        && Arrays.equals(targetPolicy, other.targetPolicy)
        && Arrays.equals(explicitRegretTargets, other.explicitRegretTargets)
        && observation.equals(other.observation);
  }
}

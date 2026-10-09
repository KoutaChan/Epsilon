package com.epsilon.cfr.collection;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.ai.grp.EpsilonGrpInference;
import java.util.List;
import java.util.Objects;

/** 呼び出し側が所有する凍結済みの共通GRPモデルを、境界評価器として利用する。 */
public final class FrozenGrpBoundary implements BoundaryValue {
  private final EpsilonGrpInference inference;
  private final EpsilonUtilityProfile utilityProfile;

  public FrozenGrpBoundary(EpsilonGrpInference inference, EpsilonUtilityProfile utilityProfile) {
    this.inference = Objects.requireNonNull(inference);
    if (utilityProfile == null || !utilityProfile.rankBased())
      throw new IllegalArgumentException("GRP boundary requires a rank utility profile");
    this.utilityProfile = utilityProfile;
  }

  @Override
  public double[] evaluate(float[] prefix) {
    return evaluateBatch(List.of(prefix)).getFirst();
  }

  @Override
  public List<double[]> evaluateBatch(List<float[]> prefixes) {
    if (prefixes.isEmpty()) return List.of();
    return inference.predictMarginalProbabilities(prefixes).stream().map(this::utilities).toList();
  }

  private double[] utilities(float[] probabilities) {
    double[] utilities = new double[4];
    for (int seat = 0; seat < 4; seat++) {
      int offset = seat * 4;
      utilities[seat] =
          EpsilonUtilityTargets.expectedRankUtilityTrustedDouble(
              utilityProfile, probabilities, offset);
    }
    return utilities;
  }
}

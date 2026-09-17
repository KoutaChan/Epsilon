package com.epsilon.major.ai.decision.training;

/** 一回の更新に使うデータ全体の平均 KL ダイバージェンスで、収集時の方策からの変化を制限する。 */
public record DecisionPolicyTrustRegion(float maximumOptimizerShardMeanKl) {

  public DecisionPolicyTrustRegion {
    if (!Float.isFinite(maximumOptimizerShardMeanKl) || maximumOptimizerShardMeanKl < 0.0f) {
      throw new IllegalArgumentException(
          "maximumOptimizerShardMeanKl must be finite and non-negative");
    }
  }

  public boolean enabled() {
    return maximumOptimizerShardMeanKl > 0.0f;
  }

  EpsilonDecisionTrainingRejection checkMeanRolloutPolicyKl(float observedMean) {
    if (!Float.isFinite(observedMean)) {
      throw new IllegalStateException("Non-finite mean rollout policy KL: " + observedMean);
    }
    return enabled() && observedMean > maximumOptimizerShardMeanKl
        ? EpsilonDecisionTrainingRejection.rejected(
            EpsilonDecisionTrainingRejection.Code.ROLLOUT_POLICY_KL_MEAN_EXCEEDED,
            "rolloutPolicyKlGuard=REJECTED,meanRolloutPolicyKl="
                + observedMean
                + ",optimizerShardMeanLimit="
                + maximumOptimizerShardMeanKl)
        : EpsilonDecisionTrainingRejection.PASSED;
  }
}

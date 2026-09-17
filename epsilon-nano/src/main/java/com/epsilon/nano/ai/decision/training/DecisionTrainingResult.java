package com.epsilon.nano.ai.decision.training;

import java.util.Objects;

/** 自己対局学習で得た方策、価値、処理性能の指標。 */
public record DecisionTrainingResult(
    int optimizerSteps,
    int microBatches,
    float loss,
    float actorLoss,
    float entropyBonusLoss,
    float behaviorCloningLoss,
    float valueLoss,
    float entropy,
    float rolloutEntropy,
    float behaviorToRolloutPolicyKl,
    float chosenProb,
    float behaviorProbMean,
    float behaviorProbMin,
    float meanRolloutPolicyKl,
    float maximumRolloutPolicyKl,
    float actorWeight,
    PolicyRatioMetrics policyRatios,
    long positiveScalarAdvantageSamples,
    long negativeScalarAdvantageSamples,
    long zeroScalarAdvantageSamples,
    EpsilonDecisionTrainingRejection rejection,
    DecisionTrainingPerformanceSnapshot performance) {

  public DecisionTrainingResult {
    Objects.requireNonNull(policyRatios, "policyRatios");
    Objects.requireNonNull(rejection, "rejection");
    Objects.requireNonNull(performance, "performance");
    if (optimizerSteps < 0 || microBatches < 0) {
      throw new IllegalArgumentException("training result counts must be non-negative");
    }
    if (positiveScalarAdvantageSamples < 0
        || negativeScalarAdvantageSamples < 0
        || zeroScalarAdvantageSamples < 0) {
      throw new IllegalArgumentException("advantage sign counts must be non-negative");
    }
    requireFinite(
        loss,
        actorLoss,
        entropyBonusLoss,
        behaviorCloningLoss,
        valueLoss,
        entropy,
        rolloutEntropy,
        behaviorToRolloutPolicyKl,
        chosenProb,
        behaviorProbMean,
        behaviorProbMin,
        meanRolloutPolicyKl,
        maximumRolloutPolicyKl,
        actorWeight);
    if (!rejection.rejected() && (optimizerSteps == 0 || microBatches == 0)) {
      throw new IllegalStateException(
          "accepted training result requires optimizer steps and micro-batches");
    }
  }

  public boolean rejected() {
    return rejection.rejected();
  }

  public String reason() {
    return rejection.reason();
  }

  private static void requireFinite(float... values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        throw new IllegalStateException("training result contains a non-finite metric: " + value);
      }
    }
  }

  /** 一種類の比率を有効方策学習の重みで集約した値。 */
  public record RatioMetrics(float mean, float minimum, float maximum) {

    public RatioMetrics {
      requireFinite(mean, minimum, maximum);
      if (minimum < 0.0f || mean < minimum || maximum < mean) {
        throw new IllegalArgumentException(
            "ratio metrics must satisfy 0 <= minimum <= mean <= maximum");
      }
    }

    static RatioMetrics empty() {
      return new RatioMetrics(0.0f, 0.0f, 0.0f);
    }
  }

  /** 元の探索比、探索学習への寄与、方策更新、実効方策モデル比を分離した選択行動診断値。 */
  public record PolicyRatioMetrics(
      RatioMetrics rawExplorationRatio,
      RatioMetrics explorationCreditWeight,
      RatioMetrics policyUpdate,
      RatioMetrics effectiveActorRatio,
      float policyUpdateClipFraction,
      float effectiveActorRatioEffectiveSampleFraction) {

    public PolicyRatioMetrics {
      Objects.requireNonNull(rawExplorationRatio, "rawExplorationRatio");
      Objects.requireNonNull(explorationCreditWeight, "explorationCreditWeight");
      Objects.requireNonNull(policyUpdate, "policyUpdate");
      Objects.requireNonNull(effectiveActorRatio, "effectiveActorRatio");
      requireFinite(policyUpdateClipFraction);
      requireFinite(effectiveActorRatioEffectiveSampleFraction);
      if (policyUpdateClipFraction < 0.0f || policyUpdateClipFraction > 1.0f + 1.0e-5f) {
        throw new IllegalArgumentException("policyUpdateClipFraction must be in [0, 1]");
      }
      if (effectiveActorRatioEffectiveSampleFraction < 0.0f
          || effectiveActorRatioEffectiveSampleFraction > 1.0f + 1.0e-5f) {
        throw new IllegalArgumentException(
            "effectiveActorRatioEffectiveSampleFraction must be in [0, 1]");
      }
    }

    static PolicyRatioMetrics empty() {
      RatioMetrics empty = RatioMetrics.empty();
      return new PolicyRatioMetrics(empty, empty, empty, empty, 0.0f, 0.0f);
    }
  }
}

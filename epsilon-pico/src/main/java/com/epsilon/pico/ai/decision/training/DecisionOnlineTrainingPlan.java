package com.epsilon.pico.ai.decision.training;

import java.util.Objects;

/** 自己対局で選択した行動のデータを順次読み込み、方策と価値関数を更新するための実行計画。 */
public record DecisionOnlineTrainingPlan(
    int microBatchSize,
    int maximumDeviceTransitionCells,
    float policyUpdateClipRange,
    float explorationCreditMix,
    float entropyCoefficient,
    DecisionPolicyTrustRegion trustRegion,
    DecisionPolicySignalMultipliers policySignalMultipliers,
    int ppoEpochs,
    int optimizerStepsPerEpoch,
    boolean debugActorValidationEnabled) {

  public DecisionOnlineTrainingPlan {
    policySignalMultipliers =
        Objects.requireNonNull(policySignalMultipliers, "policySignalMultipliers");
    trustRegion = Objects.requireNonNull(trustRegion, "trustRegion");
    if (microBatchSize <= 0 || maximumDeviceTransitionCells <= 0) {
      throw new IllegalArgumentException(
          "online training batch size and transition-cell limit must be positive");
    }
    if (ppoEpochs <= 0 || optimizerStepsPerEpoch <= 0) {
      throw new IllegalArgumentException(
          "online training ppoEpochs and optimizerStepsPerEpoch must be positive");
    }
    try {
      Math.multiplyExact(ppoEpochs, optimizerStepsPerEpoch);
    } catch (ArithmeticException overflow) {
      throw new IllegalArgumentException(
          "online training total optimizer steps overflow", overflow);
    }
    if (optimizerStepsPerEpoch > 1 && !policySignalMultipliers.uniform()) {
      throw new IllegalArgumentException(
          "multi-step PPO epoch requires equal policy signal multipliers");
    }
    if (!Float.isFinite(policyUpdateClipRange)
        || policyUpdateClipRange <= 0.0f
        || policyUpdateClipRange >= 1.0f) {
      throw new IllegalArgumentException(
          "online training policyUpdateClipRange must be finite and in (0, 1)");
    }
    if (!Float.isFinite(explorationCreditMix)
        || explorationCreditMix <= 0.0f
        || explorationCreditMix > 1.0f) {
      throw new IllegalArgumentException(
          "online training explorationCreditMix must be finite and in (0, 1]");
    }
    if (!Float.isFinite(entropyCoefficient)
        || entropyCoefficient < 0.0f
        || entropyCoefficient > DecisionOnlineLossConfig.MAXIMUM_ENTROPY_COEFFICIENT) {
      throw new IllegalArgumentException(
          "online training entropyCoefficient must be finite and in [0, "
              + DecisionOnlineLossConfig.MAXIMUM_ENTROPY_COEFFICIENT
              + "]");
    }
  }

  /** Test・small-実行向けに遷移-格納枠上限を実質無制限とする最小計画。 */
  public DecisionOnlineTrainingPlan(int microBatchSize, DecisionPolicyTrustRegion trustRegion) {
    this(
        microBatchSize,
        Integer.MAX_VALUE,
        DecisionOnlineLossConfig.DEFAULT_POLICY_UPDATE_CLIP_RANGE,
        DecisionOnlineLossConfig.DEFAULT_EXPLORATION_CREDIT_MIX,
        DecisionOnlineLossConfig.DEFAULT_ENTROPY_COEFFICIENT,
        trustRegion,
        DecisionPolicySignalMultipliers.allCategories(),
        1,
        1,
        false);
  }

  /** クリップ幅を明示し、探索による選択の学習への寄与には標準値を使うTest・small-実行向け最小計画。 */
  public static DecisionOnlineTrainingPlan withPolicyUpdateClipRange(
      int microBatchSize, float policyUpdateClipRange, DecisionPolicyTrustRegion trustRegion) {
    return new DecisionOnlineTrainingPlan(
        microBatchSize,
        Integer.MAX_VALUE,
        policyUpdateClipRange,
        DecisionOnlineLossConfig.DEFAULT_EXPLORATION_CREDIT_MIX,
        DecisionOnlineLossConfig.DEFAULT_ENTROPY_COEFFICIENT,
        trustRegion,
        DecisionPolicySignalMultipliers.allCategories(),
        1,
        1,
        false);
  }

  /** この新しく生成した学習サイクルから実行する方策・価値オプティマイザー確定総数。 */
  public int totalOptimizerSteps() {
    return Math.multiplyExact(ppoEpochs, optimizerStepsPerEpoch);
  }

  /** 標準クリップ幅を使う本番計画。 */
  public DecisionOnlineTrainingPlan(
      int microBatchSize,
      int maximumDeviceTransitionCells,
      DecisionPolicyTrustRegion trustRegion,
      DecisionPolicySignalMultipliers policySignalMultipliers,
      int optimizerStepsPerPass,
      boolean debugActorValidationEnabled) {
    this(
        microBatchSize,
        maximumDeviceTransitionCells,
        DecisionOnlineLossConfig.DEFAULT_POLICY_UPDATE_CLIP_RANGE,
        DecisionOnlineLossConfig.DEFAULT_EXPLORATION_CREDIT_MIX,
        DecisionOnlineLossConfig.DEFAULT_ENTROPY_COEFFICIENT,
        trustRegion,
        policySignalMultipliers,
        1,
        optimizerStepsPerPass,
        debugActorValidationEnabled);
  }

  /** 探索による選択の学習への寄与混合率は標準値を使い、指定した更新回数数を1 エポックだけ実行する。 */
  public DecisionOnlineTrainingPlan(
      int microBatchSize,
      int maximumDeviceTransitionCells,
      float policyUpdateClipRange,
      DecisionPolicyTrustRegion trustRegion,
      DecisionPolicySignalMultipliers policySignalMultipliers,
      int optimizerStepsPerPass,
      boolean debugActorValidationEnabled) {
    this(
        microBatchSize,
        maximumDeviceTransitionCells,
        policyUpdateClipRange,
        DecisionOnlineLossConfig.DEFAULT_EXPLORATION_CREDIT_MIX,
        DecisionOnlineLossConfig.DEFAULT_ENTROPY_COEFFICIENT,
        trustRegion,
        policySignalMultipliers,
        1,
        optimizerStepsPerPass,
        debugActorValidationEnabled);
  }

  /** 探索による選択の学習への寄与混合率を明示し、指定した更新回数数を1 エポックだけ実行する。 */
  public DecisionOnlineTrainingPlan(
      int microBatchSize,
      int maximumDeviceTransitionCells,
      float policyUpdateClipRange,
      float explorationCreditMix,
      DecisionPolicyTrustRegion trustRegion,
      DecisionPolicySignalMultipliers policySignalMultipliers,
      int optimizerStepsPerPass,
      boolean debugActorValidationEnabled) {
    this(
        microBatchSize,
        maximumDeviceTransitionCells,
        policyUpdateClipRange,
        explorationCreditMix,
        DecisionOnlineLossConfig.DEFAULT_ENTROPY_COEFFICIENT,
        trustRegion,
        policySignalMultipliers,
        1,
        optimizerStepsPerPass,
        debugActorValidationEnabled);
  }
}

package com.epsilon.major.ai.decision.training;

/** 自己対局で選択した行動と価値の損失を、一回の逆伝播にまとめるための設定。 */
public record DecisionOnlineLossConfig(
    float policyUpdateClipRange,
    float explorationCreditMix,
    float entropyCoefficient,
    boolean branchComparisonEnabled) {

  public static final float DEFAULT_POLICY_UPDATE_CLIP_RANGE = 0.2f;
  public static final float DEFAULT_EXPLORATION_CREDIT_MIX = 0.10f;
  public static final float DEFAULT_ENTROPY_COEFFICIENT = 0.0f;
  public static final float MAXIMUM_ENTROPY_COEFFICIENT = 0.05f;

  public DecisionOnlineLossConfig {
    if (!Float.isFinite(policyUpdateClipRange)
        || policyUpdateClipRange <= 0.0f
        || policyUpdateClipRange >= 1.0f) {
      throw new IllegalArgumentException("policyUpdateClipRange must be finite and in (0, 1)");
    }
    if (!Float.isFinite(explorationCreditMix)
        || explorationCreditMix <= 0.0f
        || explorationCreditMix > 1.0f) {
      throw new IllegalArgumentException("explorationCreditMix must be finite and in (0, 1]");
    }
    if (!Float.isFinite(entropyCoefficient)
        || entropyCoefficient < 0.0f
        || entropyCoefficient > MAXIMUM_ENTROPY_COEFFICIENT) {
      throw new IllegalArgumentException(
          "entropyCoefficient must be finite and in [0, " + MAXIMUM_ENTROPY_COEFFICIENT + "]");
    }
  }

  public DecisionOnlineLossConfig withBranchComparison(boolean enabled) {
    return new DecisionOnlineLossConfig(
        policyUpdateClipRange, explorationCreditMix, entropyCoefficient, enabled);
  }

  public static DecisionOnlineLossConfig create() {
    return new DecisionOnlineLossConfig(
        DEFAULT_POLICY_UPDATE_CLIP_RANGE,
        DEFAULT_EXPLORATION_CREDIT_MIX,
        DEFAULT_ENTROPY_COEFFICIENT,
        false);
  }

  /** PPO クリップ幅を明示し、探索分の学習への寄与には標準値を使う。 */
  public static DecisionOnlineLossConfig createWithPolicyUpdateClipRange(
      float policyUpdateClipRange) {
    return new DecisionOnlineLossConfig(
        policyUpdateClipRange, DEFAULT_EXPLORATION_CREDIT_MIX, DEFAULT_ENTROPY_COEFFICIENT, false);
  }

  /** PPO クリップ幅と探索分の学習への寄与混合率を明示する。 */
  public static DecisionOnlineLossConfig create(
      float policyUpdateClipRange, float explorationCreditMix) {
    return new DecisionOnlineLossConfig(
        policyUpdateClipRange, explorationCreditMix, DEFAULT_ENTROPY_COEFFICIENT, false);
  }

  /** PPO クリップ幅、探索分の学習への寄与混合率、最終的な行動分布のエントロピー係数を明示する。 */
  public static DecisionOnlineLossConfig create(
      float policyUpdateClipRange, float explorationCreditMix, float entropyCoefficient) {
    return new DecisionOnlineLossConfig(
        policyUpdateClipRange, explorationCreditMix, entropyCoefficient, false);
  }
}

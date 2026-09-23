package com.epsilon.major.ai.decision;

import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.major.ai.decision.data.EpsilonDecisionDataException;

/** 価値と方策で使う遷移列を分けて V-trace 教師値を計算し、HL-Gauss 損失の値域を検証する。 */
public final class EpsilonDecisionReturns {
  private EpsilonDecisionReturns() {}

  /** 現在の行動の補正係数を局所 TD 誤差と後続トレースの双方へ適用する。 */
  public static float scalarVTraceTarget(
      float currentValue,
      float nextValue,
      float nextTarget,
      float lambda,
      float coefficient) {
    return (1.0f - coefficient) * currentValue
        + coefficient * ((1.0f - lambda) * nextValue + lambda * nextTarget);
  }

  /** 現在の行動の補正を重ねず、後続の価値トレースから Actor の行動価値を推定する。 */
  public static float actorLookaheadTarget(float nextValue, float nextTarget, float lambda) {
    return (1.0f - lambda) * nextValue + lambda * nextTarget;
  }

  /** CPUの教師入力境界で有限性と余白込み値域を確認する。範囲内の値は丸めない。 */
  public static float requireValueTarget(float value, EpsilonUtilityProfile profile) {
    EpsilonDecisionHlGauss.Support support = EpsilonDecisionHlGauss.support(profile);
    if (!Float.isFinite(value)
        || value < (float) support.minimum()
        || value > (float) support.maximum()) {
      throw new EpsilonDecisionDataException("Value target outside HL-Gauss support: " + value);
    }
    return value;
  }

  /** 探索前後の比率を混合してから 1 で切り、V-trace の局所補正とトレースに使う。 */
  public static float selectedTraceCoefficient(
      float rolloutProbability, float behaviorProbability, float explorationCreditMix) {
    if (!Float.isFinite(rolloutProbability)
        || rolloutProbability < 0.0f
        || rolloutProbability > 1.0f) {
      throw new IllegalArgumentException("rolloutProbability must be finite and in [0,1]");
    }
    if (!Float.isFinite(behaviorProbability)
        || behaviorProbability <= 0.0f
        || behaviorProbability > 1.0f) {
      throw new IllegalArgumentException("behaviorProbability must be finite and in (0,1]");
    }
    if (!Float.isFinite(explorationCreditMix)
        || explorationCreditMix <= 0.0f
        || explorationCreditMix > 1.0f) {
      throw new IllegalArgumentException("explorationCreditMix must be finite and in (0,1]");
    }
    if (explorationCreditMix == 1.0f) {
      return 1.0f;
    }
    double ratio = rolloutProbability / (double) behaviorProbability;
    double retainedRatio = (1.0 - explorationCreditMix) * ratio + explorationCreditMix;
    return (float) Math.min(1.0, retainedRatio);
  }
}

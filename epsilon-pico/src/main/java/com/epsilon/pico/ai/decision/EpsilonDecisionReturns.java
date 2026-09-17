package com.epsilon.pico.ai.decision;

import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionDataException;

/**
 * Retrace による収益の推定と、HL-Gauss の教師値の検証を行う。
 *
 * <p>価値学習と方策学習では対象となる判断の列を分け、それぞれの次の判断に対応する係数で予測値と教師値を混合する。
 */
public final class EpsilonDecisionReturns {
  private EpsilonDecisionReturns() {}

  /** 検証済みの予測と教師値を、同じ判断列の次Decision係数で線形混合する。 */
  public static float scalarRetraceTarget(
      float nextValue, float nextTarget, float lambda, float coefficient) {
    float traceWeight = lambda * coefficient;
    return (1.0f - traceWeight) * nextValue + traceWeight * nextTarget;
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

  /** 選択行動の対局生成/探索適用後の比へ探索による選択の学習への寄与を残し、Retrace係数 {@code c=min(1,q_ret)} を返す。 */
  public static float selectedRetraceCoefficient(
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

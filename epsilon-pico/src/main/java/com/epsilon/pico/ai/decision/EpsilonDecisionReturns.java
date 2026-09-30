package com.epsilon.pico.ai.decision;

import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionDataException;

/**
 * 探索補正付きの Value トレースと、HL-Gauss の教師値の検証を行う。
 *
 * <p>価値学習と方策学習では対象となる判断の列を分ける。選択行動の補正係数は現在の判断の TD 差分に掛け、
 * 後続判断へのトレースにも同じ係数を掛ける。
 */
public final class EpsilonDecisionReturns {
  private EpsilonDecisionReturns() {}

  /** 現在の判断に対応する補正係数を局所 TD 差分と後続トレースの両方へ適用する。 */
  public static float scalarVTraceTarget(
      float currentValue, float nextValue, float nextTarget, float lambda, float coefficient) {
    double continuation = (1.0 - lambda) * nextValue + lambda * nextTarget;
    return (float) ((1.0 - coefficient) * currentValue + coefficient * continuation);
  }

  /** Actor 判断列の次の補正済み Value から、選択行動の継続価値を推定する。 */
  public static float actorLookaheadTarget(float nextValue, float nextTarget, float lambda) {
    return (float) ((1.0 - lambda) * nextValue + lambda * nextTarget);
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

  /** 探索前と探索後の選択確率から、共通の Value / Actor 補正係数を返す。 */
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

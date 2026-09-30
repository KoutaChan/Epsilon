package com.epsilon.nano.ai.decision;

import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionDataException;

/** 価値学習と方策学習で遡る手順を分けた局内の補正付き教師値を計算し、HL-Gauss 教師の範囲を検証する。 */
public final class EpsilonDecisionReturns {
  private EpsilonDecisionReturns() {}

  /** 現在のTD誤差と後続トレースを同じ選択行動の補正係数で重み付けする。 */
  public static float scalarVTraceTarget(
      float currentValue, float nextValue, float nextTarget, float lambda, float coefficient) {
    float lookahead = actorLookaheadTarget(nextValue, nextTarget, lambda);
    return (1.0f - coefficient) * currentValue + coefficient * lookahead;
  }

  /** 選択行動の先で観測した価値と、補正済みの後続価値から方策用のQ推定値を作る。 */
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

  /** 選択行動の探索前/探索後の比へ探索学習への寄与を残し、局所補正係数 {@code rho=min(1,q)} を返す。 */
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

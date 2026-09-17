package com.epsilon.config.settings;

/**
 * 方策の各分岐で使う探索用の混合確率と、最終的な合法手の確率下限を指定する。
 *
 * <p>探索用の混合確率は、モデルの方策と合法候補の一様分布を混合するとき、一様分布側へ割り当てる確率を表す。
 *
 * @param terminalGateExplorationMass 和了・流局宣言と続行を分ける終了分岐の探索用の混合確率
 * @param callGateExplorationMass 鳴きを見送るか、鳴くかを選ぶ分岐の探索用混合確率
 * @param meldTypeExplorationMass チー・ポン・大明槓を選ぶ分岐の探索用混合確率
 * @param meldCandidateExplorationMass 同じ鳴きの種類の中で、牌の組み合わせや赤牌の使い方を選ぶ分岐の探索用混合確率
 * @param kanGateExplorationMass 槓を見送るか、槓をするかを選ぶ分岐の探索用混合確率
 * @param kanTypeExplorationMass 暗槓・加槓を選ぶ分岐の探索用混合確率
 * @param kanCandidateExplorationMass 同じ槓の種類の中で、使う牌を選ぶ分岐の探索用混合確率
 * @param discardIdentityExplorationMass 打牌候補を選ぶ分岐の探索用混合確率
 * @param riichiGateExplorationMass ダマテンかリーチかを選ぶ分岐の探索用混合確率
 * @param minimumLeafProbability 分岐ごとの確率を掛け合わせた後も、各合法手に残す確率の下限
 * @param adaptiveExploration 方策の不確実性に応じて、各分岐の探索用の混合確率を調整する設定
 */
@SettingsPrefix("epsilon.decision.rollout.fullSupport")
public record DecisionFullSupportSettings(
    @Setting("terminalGateExplorationMass")
        @Default("0.004")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float terminalGateExplorationMass,
    @Setting("callGateExplorationMass")
        @Default("0.004")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float callGateExplorationMass,
    @Setting("meldTypeExplorationMass")
        @Default("0.004")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float meldTypeExplorationMass,
    @Setting("meldCandidateExplorationMass")
        @Default("0.004")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float meldCandidateExplorationMass,
    @Setting("kanGateExplorationMass")
        @Default("0.004")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float kanGateExplorationMass,
    @Setting("kanTypeExplorationMass")
        @Default("0.004")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float kanTypeExplorationMass,
    @Setting("kanCandidateExplorationMass")
        @Default("0.004")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float kanCandidateExplorationMass,
    @Setting("discardIdentityExplorationMass")
        @Default("0.010")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float discardIdentityExplorationMass,
    @Setting("riichiGateExplorationMass")
        @Default("0.004")
        @Range(min = 0.0, max = 1.0, maxInclusive = false)
        float riichiGateExplorationMass,
    @Setting("minimumLeafProbability")
        @Default("0.0")
        @Range(min = 0.0, max = 1.0 / 64.0, maxInclusive = false)
        float minimumLeafProbability,
    @Setting("adaptiveExploration") AdaptiveExplorationSettings adaptiveExploration) {

  /** 入れ子の設定の必須性と、適応後もノード混合係数が確率として有効なことを検証する。 */
  public DecisionFullSupportSettings {
    if (adaptiveExploration == null) {
      throw new IllegalArgumentException("adaptive exploration settings must not be null");
    }
    double maximumScaledNodeMass =
        maximumNodeExplorationMass(
                terminalGateExplorationMass,
                callGateExplorationMass,
                meldTypeExplorationMass,
                meldCandidateExplorationMass,
                kanGateExplorationMass,
                kanTypeExplorationMass,
                kanCandidateExplorationMass,
                discardIdentityExplorationMass,
                riichiGateExplorationMass)
            * (double) adaptiveExploration.maximumScale();
    if (adaptiveExploration.schedule() == DecisionExplorationSchedule.PERCENTILE_SCALE
        && maximumScaledNodeMass >= 1.0) {
      throw new IllegalArgumentException(
          "adaptive exploration maximum scale must keep every node exploration mass below 1");
    }
  }

  /**
   * @return 全ての分岐の探索用混合確率と、最終的な合法手の確率下限をゼロにした設定
   */
  public static DecisionFullSupportSettings disabled() {
    return SettingsLoader.validate(
        new DecisionFullSupportSettings(
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            AdaptiveExplorationSettings.normal()));
  }

  /**
   * @return いずれかの分岐の探索用混合確率、または最終的な合法手の確率下限が正ならtrue
   */
  public boolean enabled() {
    return terminalGateExplorationMass > 0.0f
        || callGateExplorationMass > 0.0f
        || meldTypeExplorationMass > 0.0f
        || meldCandidateExplorationMass > 0.0f
        || kanGateExplorationMass > 0.0f
        || kanTypeExplorationMass > 0.0f
        || kanCandidateExplorationMass > 0.0f
        || discardIdentityExplorationMass > 0.0f
        || riichiGateExplorationMass > 0.0f
        || minimumLeafProbability > 0.0f;
  }

  /**
   * @return 全ての分岐で探索用の混合確率が正ならtrue
   */
  public boolean allNodesExplored() {
    return terminalGateExplorationMass > 0.0f
        && callGateExplorationMass > 0.0f
        && meldTypeExplorationMass > 0.0f
        && meldCandidateExplorationMass > 0.0f
        && kanGateExplorationMass > 0.0f
        && kanTypeExplorationMass > 0.0f
        && kanCandidateExplorationMass > 0.0f
        && discardIdentityExplorationMass > 0.0f
        && riichiGateExplorationMass > 0.0f;
  }

  /**
   * @return 全ての分岐の探索と、最終的な合法手の確率下限の両方が有効ならtrue
   */
  public boolean guaranteesFullLeafCoverage() {
    return allNodesExplored() && minimumLeafProbability > 0.0f;
  }

  private static float maximumNodeExplorationMass(float first, float... remaining) {
    float maximum = first;
    for (float value : remaining) {
      maximum = Math.max(maximum, value);
    }
    return maximum;
  }

  /** 方策の不確実性が過去の分布で何パーセントの位置にあるかに応じて、探索用の混合確率の倍率を指定する。 */
  public record AdaptiveExplorationSettings(
      @Setting("schedule") DecisionExplorationSchedule schedule,
      @Setting("entropyWeight") @Range(min = 0.0, max = 1.0) float entropyWeight,
      @Setting("minimumScale") @Range(min = 0.0, minInclusive = false, max = 1.0)
          float minimumScale,
      @Setting("maximumScale") @Range(min = 1.0) float maximumScale,
      @Setting("percentileExponent") @Positive float percentileExponent) {

    /** 列挙型の必須性と探索倍率境界の順序を検証する。 */
    public AdaptiveExplorationSettings {
      if (schedule == null) {
        throw new IllegalArgumentException("adaptive exploration schedule must not be null");
      }
      if (minimumScale > maximumScale) {
        throw new IllegalArgumentException(
            "adaptive exploration scales must satisfy minimumScale <= maximumScale");
      }
    }

    /**
     * @return 不確実性による倍率調整を行わず、設定されたノード探索用の混合確率を保つ設定
     */
    public static AdaptiveExplorationSettings normal() {
      return new AdaptiveExplorationSettings(
          DecisionExplorationSchedule.NORMAL, 0.5f, 0.1f, 5.0f, 4.5f);
    }

    /**
     * @return 確率上位2候補の差の小ささへ割り当てる不確実性スコアの重み
     */
    public float topTwoAmbiguityWeight() {
      return 1.0f - entropyWeight;
    }
  }
}

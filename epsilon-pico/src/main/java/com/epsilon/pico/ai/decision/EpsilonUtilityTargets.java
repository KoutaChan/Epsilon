package com.epsilon.pico.ai.decision;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.core.ScoreRanking;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionDataException;
import com.epsilon.pico.config.settings.DecisionSettings;

/** ルールごとの得点・順位を、Decision の価値学習に用いる効用へ変換する。 */
public final class EpsilonUtilityTargets {

  private static final EpsilonUtilityProfile[] PROFILES = EpsilonUtilityProfile.values();

  private EpsilonUtilityTargets() {}

  /** 保存通し番号を検証して、そのサンプル自身の効用の定義を返します。 */
  public static EpsilonUtilityProfile profile(int ordinal) {
    if (ordinal < 0 || ordinal >= PROFILES.length) {
      throw new EpsilonDecisionDataException("Invalid utility profile: " + ordinal);
    }
    return PROFILES[ordinal];
  }

  /** モデル学習入口で保存サンプルとモデルの設定一致を検査します。 */
  public static void requireTrainingProfile(int sampleProfile, EpsilonUtilityProfile expected) {
    if (profile(sampleProfile) != expected) {
      throw new EpsilonDecisionDataException(
          "Decision utility profile mismatch: sample=" + sampleProfile + " expected=" + expected);
    }
  }

  /**
   * 有効設定の効用の定義を内部列挙型へ変換する。
   *
   * @return 推論・評価に使う効用の定義
   */
  public static EpsilonUtilityProfile configuredProfile() {
    return DecisionSettings.defaults().utilityProfile();
  }

  /**
   * 学習サンプルが従うべき効用の定義を返す。
   *
   * @return 学習時の効用の定義
   */
  public static EpsilonUtilityProfile configuredTrainingProfile() {
    return configuredProfile();
  }

  /**
   * サンプルの効用の定義が有効な学習設定と一致することを要求する。
   *
   * @param sampleProfile サンプルに保存された設定通し番号
   * @throws EpsilonDecisionDataException 設定とサンプルが一致しない場合
   */
  public static void requireConfiguredTrainingProfile(int sampleProfile) {
    int configured = configuredTrainingProfile().ordinal();
    if (sampleProfile != configured) {
      throw new EpsilonDecisionDataException(
          "Decision utility profile mismatch: sample="
              + sampleProfile
              + " configured="
              + configured);
    }
  }

  /**
   * 終局得点から全効用の定義の教師を作る。
   *
   * @param seat 教師を作る席
   * @param finalScores 席ごとの終局得点
   * @return 設定通し番号順の効用教師値
   */
  public static float[] fromFinalScores(int seat, int[] finalScores) {
    int[] ranks = ScoreRanking.byScoreThenSeat(finalScores);
    return fromScoresAndRanks(seat, finalScores, ranks);
  }

  /**
   * 確定得点と順位から全効用の定義の教師を作る。
   *
   * @param seat 教師を作る席
   * @param scores 席ごとの終局得点
   * @param ranks 席ごとの 0始まりの最終順位
   * @return 設定通し番号順の効用教師値
   */
  public static float[] fromScoresAndRanks(int seat, int[] scores, int[] ranks) {
    float[] utility = new float[EpsilonDecisionConstants.UTILITY_PROFILE_COUNT];
    for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
      utility[profile.ordinal()] =
          switch (profile) {
            case SCORE -> scoreUtility(scores[seat]);
            default -> profile.utilityForRank(ranks[seat]);
          };
    }
    return utility;
  }

  /**
   * 素点を25,000点基準の学習尺度へ正規化する。
   *
   * @param score 終局得点
   * @return スコア設定のスカラー効用
   */
  public static float scoreUtility(int score) {
    return ((score - 25000) / 1000.0f) / 50.0f;
  }

  /**
   * 0始まりの最終順位の one-hot 分布を返す。
   *
   * @param finalRank 0始まりの最終順位
   * @return 長さ4の one-hot 順位教師値
   */
  public static float[] oneHotRank(int finalRank) {
    if (finalRank < 0 || finalRank >= EpsilonDecisionConstants.PLAYERS) {
      throw new IllegalArgumentException("finalRank must be 0-3: " + finalRank);
    }
    float[] out = new float[EpsilonDecisionConstants.PLAYERS];
    out[finalRank] = 1.0f;
    return out;
  }

  /**
   * 順位-based 効用の定義で 4-way 最終-順位分布をスカラー価値へ復号する。
   *
   * @param profile 順位効用を持つ設定
   * @param rankProbabilities 1着から4着までの予測確率
   * @return 予測順位分布の期待効用
   */
  public static float expectedRankUtility(
      EpsilonUtilityProfile profile, float[] rankProbabilities) {
    float inverse = (float) (1.0 / EpsilonGrpRanks.requireRankProbabilityMass(rankProbabilities));
    requireRankBasedProfile(profile);
    float value = 0.0f;
    for (int rank = 0; rank < rankProbabilities.length; rank++) {
      // 旧来のfloat正規化と同じ演算順を保ち、正規化配列の割当だけを省く。
      float probability = rankProbabilities[rank] * inverse;
      value += probability * profile.utilityForRank(rank);
    }
    return value;
  }

  /** 検証・正規化済み順位分布を再コピーせず期待効用へ射影する。 */
  public static float expectedRankUtilityTrusted(
      EpsilonUtilityProfile profile, float[] rankProbabilities) {
    requireRankBasedProfile(profile);
    float value = 0.0f;
    for (int rank = 0; rank < rankProbabilities.length; rank++) {
      value += rankProbabilities[rank] * profile.utilityForRank(rank);
    }
    return value;
  }

  private static void requireRankBasedProfile(EpsilonUtilityProfile profile) {
    if (profile == null || !profile.rankBased()) {
      throw new IllegalArgumentException(
          "Rank distribution cannot decode utility profile: " + profile);
    }
  }
}

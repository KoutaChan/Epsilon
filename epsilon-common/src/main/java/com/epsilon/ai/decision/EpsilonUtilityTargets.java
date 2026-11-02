package com.epsilon.ai.decision;

import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;

/** 各モデルで共通の効用定義を用い、得点・実順位・予測順位分布を学習と評価の尺度へ変換する。 */
public final class EpsilonUtilityTargets {
  private static final EpsilonUtilityProfile[] PROFILES = EpsilonUtilityProfile.values();

  private EpsilonUtilityTargets() {}

  /** 保存されたプロファイル番号から効用定義を取得する。 */
  public static EpsilonUtilityProfile profile(int ordinal) {
    if (ordinal < 0 || ordinal >= PROFILES.length) {
      throw new IllegalArgumentException("Invalid utility profile: " + ordinal);
    }
    return PROFILES[ordinal];
  }

  /** 学習サンプルとモデルの効用定義の一致を確認する。 */
  public static void requireTrainingProfile(int sampleProfile, EpsilonUtilityProfile expected) {
    if (profile(sampleProfile) != expected) {
      throw new IllegalArgumentException(
          "Utility profile mismatch: sample=" + sampleProfile + " expected=" + expected);
    }
  }

  /** 確定した得点と0始まりの順位を、一つの効用へ変換する。 */
  public static float utility(EpsilonUtilityProfile profile, int score, int rank) {
    return profile == EpsilonUtilityProfile.SCORE
        ? scoreUtility(score)
        : profile.utilityForRank(rank);
  }

  /** 確定した得点と0始まりの順位を、倍精度の効用へ変換する。 */
  public static double utilityDouble(EpsilonUtilityProfile profile, int score, int rank) {
    return profile == EpsilonUtilityProfile.SCORE
        ? scoreUtilityDouble(score)
        : profile.utilityForRankDouble(rank);
  }

  /** 終局得点から指定席の効用を求める。同点は小さい席番号を上位とする。 */
  public static float fromFinalScores(EpsilonUtilityProfile profile, int seat, int[] finalScores) {
    if (profile == EpsilonUtilityProfile.SCORE) {
      return scoreUtility(finalScores[seat]);
    }
    int rank = 0;
    for (int other = 0; other < finalScores.length; other++) {
      if (finalScores[other] > finalScores[seat]
          || finalScores[other] == finalScores[seat] && other < seat) {
        rank++;
      }
    }
    return profile.utilityForRank(rank);
  }

  /** 終局得点から、プロファイル番号順の全効用を作る。 */
  public static float[] fromFinalScores(int seat, int[] finalScores) {
    return fromScoresAndRanks(seat, finalScores, ScoreRanking.byScoreThenSeat(finalScores));
  }

  /** 確定した得点と0始まりの順位から、プロファイル番号順の全効用を作る。 */
  public static float[] fromScoresAndRanks(int seat, int[] scores, int[] ranks) {
    float[] values = new float[PROFILES.length];
    for (EpsilonUtilityProfile profile : PROFILES) {
      values[profile.ordinal()] = utility(profile, scores[seat], ranks[seat]);
    }
    return values;
  }

  /** 素点を25,000点基準、50,000点差を1とする尺度へ正規化する。 */
  public static float scoreUtility(int score) {
    return ((score - 25000) / 1000.0f) / 50.0f;
  }

  /** 素点を、倍精度で共通の得点尺度へ正規化する。 */
  public static double scoreUtilityDouble(int score) {
    return ((score - 25000) / 1000.0) / 50.0;
  }

  /** 0始まりの最終順位をワンホット分布へ変換する。 */
  public static float[] oneHotRank(int finalRank) {
    if (finalRank < 0 || finalRank >= GameState.NUM_PLAYERS) {
      throw new IllegalArgumentException("finalRank must be 0-3: " + finalRank);
    }
    float[] out = new float[GameState.NUM_PLAYERS];
    out[finalRank] = 1.0f;
    return out;
  }

  /** 順位分布を検証・正規化して、期待効用へ変換する。 */
  public static float expectedRankUtility(
      EpsilonUtilityProfile profile, float[] rankProbabilities) {
    float inverse = (float) (1.0 / EpsilonGrpRanks.requireRankProbabilityMass(rankProbabilities));
    requireRankBasedProfile(profile);
    float value = 0.0f;
    for (int rank = 0; rank < GameState.NUM_PLAYERS; rank++) {
      float probability = rankProbabilities[rank] * inverse;
      value += probability * profile.utilityForRank(rank);
    }
    return value;
  }

  /** 検証・正規化済みの順位分布を、再コピーせず期待効用へ変換する。 */
  public static float expectedRankUtilityTrusted(
      EpsilonUtilityProfile profile, float[] rankProbabilities) {
    return expectedRankUtilityTrusted(profile, rankProbabilities, 0);
  }

  /** 検証・正規化済みの連続4要素を期待効用へ変換する。席別GRP配列も直接参照できる。 */
  public static float expectedRankUtilityTrusted(
      EpsilonUtilityProfile profile, float[] rankProbabilities, int offset) {
    requireRankBasedProfile(profile);
    float value = 0.0f;
    for (int rank = 0; rank < GameState.NUM_PLAYERS; rank++) {
      value += rankProbabilities[offset + rank] * profile.utilityForRank(rank);
    }
    return value;
  }

  /** 検証・正規化済みの連続4要素を、倍精度の順位効用で加算する。 */
  public static double expectedRankUtilityTrustedDouble(
      EpsilonUtilityProfile profile, float[] rankProbabilities, int offset) {
    requireRankBasedProfile(profile);
    double value = 0.0;
    for (int rank = 0; rank < GameState.NUM_PLAYERS; rank++) {
      value += rankProbabilities[offset + rank] * profile.utilityForRankDouble(rank);
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

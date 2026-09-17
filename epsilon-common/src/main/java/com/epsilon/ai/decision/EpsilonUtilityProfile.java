package com.epsilon.ai.decision;

/** 行動選択モデルとGRPモデルが、順位をどの効用値として評価するかを定義する。 */
public enum EpsilonUtilityProfile {
  /** 順位間を等間隔に評価する標準の順位効用値。 */
  PLACEMENT(new float[] {1.5f, 0.5f, -0.5f, -1.5f}),

  /** トップ率を強く評価する効用値。 */
  TOP(new float[] {1.0f, 0.2f, -0.2f, -0.6f}),

  /** 4着回避を最優先する効用値。 */
  LAST_AVOIDANCE(new float[] {0.4f, 0.2f, 0.0f, -1.0f}),

  /** 順位効用値を使わず、素点経路を選ぶ設定。 */
  SCORE(new float[] {0.0f, 0.0f, 0.0f, 0.0f}),

  /** 天鳳9段運用向けの共通効用値。厳密な9段pt比ではなく {@code [2/3, 1/3, 0, -1]} に固定。 */
  TENHOU(new float[] {2.0f / 3.0f, 1.0f / 3.0f, 0.0f, -1.0f}),

  /** Mリーグの順位点傾向を表す効用値。 */
  MLEAGUE(new float[] {1.0f, 0.2f, -0.2f, -0.6f});

  private final float[] rankUtility;

  EpsilonUtilityProfile(float[] rankUtility) {
    this.rankUtility = rankUtility;
  }

  /**
   * 指定順位の効用値を返す。
   *
   * @param rank 0始まりの順位
   * @return 設定固有の順位効用値
   */
  public float utilityForRank(int rank) {
    if (rank < 0 || rank >= rankUtility.length) {
      throw new IllegalArgumentException("rank must be 0-3: " + rank);
    }
    return rankUtility[rank];
  }

  /**
   * 全順位の効用値を複製して返す。
   *
   * @return 1着から4着までの効用値
   */
  public float[] rankUtility() {
    return rankUtility.clone();
  }

  /**
   * 設定が順位効用値を使うかを返す。
   *
   * @return {@link #SCORE} 以外なら {@code true}
   */
  public boolean rankBased() {
    return this != SCORE;
  }
}

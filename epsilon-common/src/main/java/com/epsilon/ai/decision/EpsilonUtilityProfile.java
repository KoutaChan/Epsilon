package com.epsilon.ai.decision;

/** 行動選択モデルとGRPモデルが、順位をどの効用値として評価するかを定義する。 */
public enum EpsilonUtilityProfile {
  /** 順位間を等間隔に評価する標準の順位効用値。 */
  PLACEMENT(new double[] {1.5, 0.5, -0.5, -1.5}),

  /** トップ率を強く評価する効用値。 */
  TOP(new double[] {1.0, 0.2, -0.2, -0.6}),

  /** 4着回避を最優先する効用値。 */
  LAST_AVOIDANCE(new double[] {0.4, 0.2, 0.0, -1.0}),

  /** 順位効用値を使わず、素点経路を選ぶ設定。 */
  SCORE(new double[] {0.0, 0.0, 0.0, 0.0}),

  /** 天鳳9段運用向けの共通効用値。厳密な9段pt比ではなく {@code [2/3, 1/3, 0, -1]} に固定。 */
  TENHOU(new double[] {2.0 / 3.0, 1.0 / 3.0, 0.0, -1.0}),

  /** Mリーグの順位点傾向を表す効用値。 */
  MLEAGUE(new double[] {1.0, 0.2, -0.2, -0.6}),

  /**
   * 平均順位を等間隔に評価する効用値 {@code [1, 1/3, -1/3, -1]}。Epsilon-Major の AVERAGE_RANK 版の網（価値の定義に profile=AVERAGE_RANK を
   * 持つチェックポイント）を読むための互換用。番号は保存した行の profile の番号に入るので、末尾に置く。
   */
  AVERAGE_RANK(new double[] {1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0});

  private final double[] rankUtility;

  EpsilonUtilityProfile(double[] rankUtility) {
    this.rankUtility = rankUtility;
  }

  /**
   * 指定順位の効用値を返す。
   *
   * @param rank 0始まりの順位
   * @return 設定固有の順位効用値
   */
  public float utilityForRank(int rank) {
    return (float) utilityForRankDouble(rank);
  }

  /** 倍精度の計算へ、floatへ丸める前の順位効用を返す。 */
  public double utilityForRankDouble(int rank) {
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
    float[] values = new float[rankUtility.length];
    for (int rank = 0; rank < values.length; rank++) {
      values[rank] = (float) rankUtility[rank];
    }
    return values;
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

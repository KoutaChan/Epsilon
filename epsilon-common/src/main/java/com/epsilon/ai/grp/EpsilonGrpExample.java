package com.epsilon.ai.grp;

/**
 * GRPの学習と評価に使う、ある局の開始時点までの特徴量系列と最終順位の組。
 *
 * <p>特徴量と順位は全席分をまとめて持つため、同じ対局の同じ局について席ごと・判断ごとに重複生成しない。対局IDは学習用と検証用のデータを対局単位で分けるために使う。
 *
 * @param gameId データ分割に使う対局ID
 * @param sequence 局の開始時点ごとのGRP特徴量を時系列順に並べた配列
 * @param finalRanksCode 4人の最終順位を0〜23に可逆符号化した教師データ
 */
public record EpsilonGrpExample(long gameId, float[] sequence, int finalRanksCode) {

  /**
   * GRP 特徴量と教師の順位コードを検証し、入力元の変更から保護するために配列を複製する。
   *
   * @param gameId 学習用と検証用の分割を対局単位に保つ対局 ID
   * @param sequence 1局以上を含む GRP 特徴量系列
   * @param finalRanksCode 4人の終局順位を表す順位コード
   */
  public EpsilonGrpExample {
    sequence = EpsilonGrpFeature.validatedCopy(sequence);
    if (EpsilonGrpFeature.steps(sequence) == 0) {
      throw new IllegalArgumentException("GRP example requires a non-empty sequence");
    }
    EpsilonGrpRanks.requireCode(finalRanksCode);
  }
}

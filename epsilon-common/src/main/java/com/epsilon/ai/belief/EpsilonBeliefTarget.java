package com.epsilon.ai.belief;

import com.epsilon.core.Tile;

/**
 * 全席の手牌を参照できる牌譜再生または自己対戦の局面から作る、Beliefモデルの教師データ。
 *
 * <p>観測者以外の3人を相対席1〜3の順で扱う。観測者自身の手牌や待ち、リーチや副露などの公開情報は予測対象に含めない。
 * 牌山は未確認の牌から他家手牌を除いた残りとして扱い、直接学習しない。値の正規化や損失の計算は学習処理側で行う。
 *
 * <p>受け入れ牌マスクは、対象の他家の手牌と公開牌から見て残り枚数がある改善牌を表す。役の有無やフリテンは判定しない。副露・槓を各3枚と数えて14枚ある手牌では、マスクはすべて0になる。
 *
 * @param opponentHandCounts 他家ごと・牌種ごとの手牌枚数。副露と暗槓を除く
 * @param hiddenTileCounts 観測者から見えない牌の残数。手牌の損失計算で使用し、出力用の一次元配列には含めない
 * @param opponentShanten 他家ごとのシャンテン数
 * @param opponentTenpai 他家ごとのテンパイ判定。シャンテン数が0以下なら1、それ以外は0。和了形も1に含む
 * @param opponentWaitMask 他家ごと・牌種ごとの受け入れ牌の有無。引くとシャンテン数が改善する牌なら1で、テンパイ時は待ち牌に対応する
 */
public record EpsilonBeliefTarget(
    float[] opponentHandCounts,
    float[] hiddenTileCounts,
    float[] opponentShanten,
    float[] opponentTenpai,
    float[] opponentWaitMask) {

  /** 新規生成した教師配列の所有権を受け取り、長さと有限性を検証する。 */
  public EpsilonBeliefTarget {
    require(opponentHandCounts, EpsilonBeliefLayout.OPPONENT_HAND_SIZE, "opponentHandCounts");
    require(hiddenTileCounts, Tile.NUM_TILE_TYPES, "hiddenTileCounts");
    require(opponentShanten, EpsilonBeliefLayout.OPPONENT_COUNT, "opponentShanten");
    require(opponentTenpai, EpsilonBeliefLayout.OPPONENT_COUNT, "opponentTenpai");
    require(opponentWaitMask, EpsilonBeliefLayout.OPPONENT_WAIT_SIZE, "opponentWaitMask");
  }

  /**
   * 所有する教師をバッチ配列の指定位置へ書き込む。観測者から見えない牌の残数は損失用の別入力として保持する。
   *
   * @param destination 手牌枚数、受け入れ牌マスク、シャンテン数とテンパイ判定を格納する配列
   * @param offset 書き込み開始位置
   */
  public void writeTo(float[] destination, int offset) {
    System.arraycopy(opponentHandCounts, 0, destination, offset, opponentHandCounts.length);
    offset += opponentHandCounts.length;
    System.arraycopy(opponentWaitMask, 0, destination, offset, opponentWaitMask.length);
    offset += opponentWaitMask.length;
    for (int i = 0; i < EpsilonBeliefLayout.OPPONENT_COUNT; i++) {
      int base = offset + i * EpsilonBeliefLayout.OPPONENT_SCALAR_COUNT;
      destination[base + EpsilonBeliefLayout.SCALAR_SHANTEN] = opponentShanten[i];
      destination[base + EpsilonBeliefLayout.SCALAR_TENPAI] = opponentTenpai[i];
    }
  }

  private static void require(float[] values, int expected, String label) {
    if (values.length != expected) {
      throw new IllegalArgumentException(
          label + " length must be " + expected + ", got " + values.length);
    }
    for (float value : values) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException(label + " contains non-finite value");
      }
    }
  }
}

package com.epsilon.util;

/** 再現可能な乱数シードを、用途を区別する固定値と連番から生成する。 */
public final class SeedMixer {

  private static final long GOLDEN_GAMMA = 0x9E3779B97F4A7C15L;

  private SeedMixer() {}

  /**
   * 64ビットの入力にSplitMix64系の変換を適用し、入力のビット変化を出力全体に拡散する。
   *
   * @param value 混合する入力値
   * @return 入力ビットを全体へ拡散した決定論的な値
   */
  public static long mix64(long value) {
    value ^= value >>> 30;
    value *= 0xBF58476D1CE4E5B9L;
    value ^= value >>> 27;
    value *= 0x94D049BB133111EBL;
    value ^= value >>> 31;
    return value;
  }

  /**
   * 基準シード、用途ごとの定数、連番から互いに分離した決定論的シードを導出する。
   *
   * @param baseSeed 実験全体の基準シード
   * @param salt 用途を分離する固定値
   * @param index 対局やサンプルの0始まり連番
   * @return 指定組み合わせに対応する混合済みシード
   */
  public static long indexed(long baseSeed, long salt, long index) {
    return mix64(baseSeed + salt + index * GOLDEN_GAMMA);
  }
}

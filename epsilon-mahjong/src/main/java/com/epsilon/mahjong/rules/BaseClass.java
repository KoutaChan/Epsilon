package com.epsilon.mahjong.rules;

import java.util.Arrays;

/** 天鳳の点数表に出てくる基本点を、小さい順に全て並べる。満貫未満は 符 × 2^(2 ＋ 翻)、役満は 8000 × 倍率（1〜6）。 */
public enum BaseClass {
  B240(240),
  B320(320),
  B400(400),
  B480(480),
  B560(560),
  B640(640),
  B720(720),
  B800(800),
  B880(880),
  B960(960),
  B1120(1120),
  B1280(1280),
  B1440(1440),
  B1600(1600),
  B1760(1760),
  B1920(1920),
  MANGAN(2000),
  HANEMAN(3000),
  BAIMAN(4000),
  SANBAIMAN(6000),
  YAKUMAN(8000),
  YAKUMAN_2(16_000),
  YAKUMAN_3(24_000),
  YAKUMAN_4(32_000),
  YAKUMAN_5(40_000),
  YAKUMAN_6(48_000);

  private static final BaseClass[] VALUES = values();

  /** 定数の宣言の順（基本点の小さい順）に並べた基本点。二分探索で引く。 */
  private static final int[] BASES = Arrays.stream(VALUES).mapToInt(BaseClass::base).toArray();

  private final int base;

  BaseClass(int base) {
    this.base = base;
  }

  public int base() {
    return base;
  }

  /** 基本点に当たる定数を返す。表に無い値なら IllegalArgumentException を投げる。 */
  public static BaseClass of(int base) {
    int index = Arrays.binarySearch(BASES, base);
    if (index < 0) {
      throw new IllegalArgumentException("Not a Tenhou base point: " + base);
    }
    return VALUES[index];
  }
}

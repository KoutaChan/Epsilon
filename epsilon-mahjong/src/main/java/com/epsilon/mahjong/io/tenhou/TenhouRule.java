package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.rules.Variant;

/** 天鳳の卓の種別を表すビット。宣言の順に bit 0 から並ぶ。牌譜の GO の type と、ライブの JOIN の種別は、同じビットで書く。 */
public enum TenhouRule {
  VERSUS_HUMAN,
  NO_RED,
  NO_OPEN_TANYAO,
  EAST_SOUTH,
  THREE_PLAYER,
  TOKUJOU,
  FAST,
  JOUKYUU;

  /**
   * 対人の東南戦（赤あり、喰いタンあり）の種別のビットを返す。
   *
   * @param variant 卓のルール
   */
  public static int hanchan(Variant variant) {
    int hanchan = VERSUS_HUMAN.bit() | EAST_SOUTH.bit();
    return switch (variant) {
      case TENHOU_4P -> hanchan;
      case TENHOU_3P -> hanchan | THREE_PLAYER.bit();
    };
  }

  int bit() {
    return 1 << ordinal();
  }

  boolean in(int type) {
    return (type & bit()) != 0;
  }
}

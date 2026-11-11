package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;

/**
 * 出力ヘッドを切り替える段階。次局がどの局かで決まる。宣言の順に、東場、南場の初めから最後の 2 つ前の局まで、南場の最後の
 * 1 つ前の局、南場の最後の局から先（西場を含む）。四人打ちなら東場、南 1〜2 局、南 3 局、南 4 局から先。三人打ちなら東場、
 * 南 1 局、南 2 局、南 3 局から先。
 */
public enum GrpPhase {
  EAST,
  SOUTH,
  PENULT,
  FINAL;

  /**
   * 次局の段階を返す。局の番号が 0 から西場の最後の局までに入らなければ IllegalArgumentException を投げる。
   *
   * @param nextRound 次局の番号（東 1 局が 0）
   */
  public static GrpPhase of(Variant variant, int nextRound) {
    if (nextRound < 0 || nextRound > variant.finalRound()) {
      throw new IllegalArgumentException("Round " + nextRound + " is outside " + variant);
    }
    if (nextRound < variant.seats()) {
      return EAST;
    }
    int penult = variant.lastRound() - 1;
    if (nextRound < penult) {
      return SOUTH;
    }
    return nextRound == penult ? PENULT : FINAL;
  }
}

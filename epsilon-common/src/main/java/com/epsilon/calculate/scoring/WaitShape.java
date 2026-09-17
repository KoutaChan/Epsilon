package com.epsilon.calculate.scoring;

/** 面子分解上で和了牌が完成させた待ち形。待ち符とロン時の暗刻扱いを集約する。 */
public enum WaitShape {
  /** 雀頭を完成させる単騎待ち。 */
  TANKI(2),

  /** 刻子を完成させる双碰待ち。 */
  SHANPON(0),

  /** 順子の両端いずれかを完成させる両面待ち。 */
  RYANMEN(0),

  /** 順子中央を完成させる嵌張待ち。 */
  KANCHAN(2),

  /** 123の3または789の7だけを待つ辺張待ち。 */
  PENCHAN(2);

  private final int waitFu;
  private static final WaitShape[] VALUES = values();

  WaitShape(int waitFu) {
    this.waitFu = waitFu;
  }

  int waitFu() {
    return waitFu;
  }

  boolean opensTripletOnRon() {
    return this == SHANPON;
  }

  public static WaitShape fromOrdinal(int ordinal) {
    return VALUES[ordinal];
  }

  public static WaitShape sequence(int firstTile, int winTile) {
    if (winTile == firstTile + 1) {
      return KANCHAN;
    }
    if (firstTile % 9 == 0 && winTile == firstTile + 2
        || firstTile % 9 == 6 && winTile == firstTile) {
      return PENCHAN;
    }
    return RYANMEN;
  }
}

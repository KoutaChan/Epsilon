package com.epsilon.core;

/** 萬子・筒子・索子の赤5所有状態を3 ビットで表す。 */
public final class AkaTileMask {

  /** 赤5を一枚も含まないマスク。 */
  public static final int EMPTY = 0;

  private static final int M5_BIT = 1;
  private static final int P5_BIT = 1 << 1;
  private static final int S5_BIT = 1 << 2;

  private AkaTileMask() {}

  /** 指定牌種の赤5を含むマスクを返す。 */
  public static int includeTile(int mask, int tileType) {
    return mask | requireBit(tileType);
  }

  /** 指定牌が赤5の場合だけ、その色のビットを含める。 */
  public static int includeTileIfAka(int mask, int tileType, boolean isAkaTile) {
    return isAkaTile ? includeTile(mask, tileType) : mask;
  }

  /** 指定牌種の赤5を除いたマスクを返す。 */
  public static int excludeTile(int mask, int tileType) {
    return mask & ~requireBit(tileType);
  }

  /** 指定牌種の赤5を含むかを返す。 */
  public static boolean containsTile(int mask, int tileType) {
    int bit = bit(tileType);
    return bit != 0 && (mask & bit) != 0;
  }

  private static int requireBit(int tileType) {
    int bit = bit(tileType);
    if (bit == 0) {
      throw new IllegalArgumentException("not an aka-capable tile type: " + tileType);
    }
    return bit;
  }

  private static int bit(int tileType) {
    return switch (tileType) {
      case Tile.M5 -> M5_BIT;
      case Tile.P5 -> P5_BIT;
      case Tile.S5 -> S5_BIT;
      default -> 0;
    };
  }
}

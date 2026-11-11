package com.epsilon.agent.observe.codec;

import com.epsilon.mahjong.tile.Tiles;

/**
 * 赤の 5 を色ごとの 1 bit で表した値（bit 0 が萬、1 が筒、2 が索）。記録の手牌の赤と表示牌の赤のフィールドは、この値で書く。種類の
 * 集合（bit k が種類 k）で表した赤の 5 と、互いに変換できる。メモリは確保しない。
 */
public final class AkaMask {

  /** 3 色の全てに赤がある値（値の上限）。 */
  public static final int ALL = (1 << Tiles.NUMBER_SUITS) - 1;

  private AkaMask() {}

  /**
   * 赤の 5 の種類の集合を、色ごとのビットにする。
   *
   * @param redKinds 赤の 5 の種類の集合（5 萬・5 筒・5 索の部分集合）
   * @return 色ごとのビット
   * @throws IllegalArgumentException 5 萬・5 筒・5 索でない種類を含むとき
   */
  public static int suitMask(long redKinds) {
    if ((redKinds & ~Tiles.FIVES) != 0) {
      throw new IllegalArgumentException("Not a set of red fives: 0x" + Long.toHexString(redKinds));
    }
    int mask = 0;
    for (int suit = 0; suit < Tiles.NUMBER_SUITS; suit++) {
      if (Tiles.contains(redKinds, five(suit))) {
        mask |= 1 << suit;
      }
    }
    return mask;
  }

  /**
   * 色ごとのビットを、赤の 5 の種類の集合にする。
   *
   * @param suitMask 色ごとのビット（0〜7）
   * @return 赤の 5 の種類の集合
   * @throws IllegalArgumentException 色ごとのビットが 0〜7 の外のとき
   */
  public static long redKinds(int suitMask) {
    if (suitMask < 0 || suitMask > ALL) {
      throw new IllegalArgumentException("Not a per-suit red mask: " + suitMask);
    }
    long kinds = 0;
    for (int suit = 0; suit < Tiles.NUMBER_SUITS; suit++) {
      if ((suitMask >>> suit & 1) != 0) {
        kinds |= Tiles.bit(five(suit));
      }
    }
    return kinds;
  }

  /**
   * 指定した種類の赤の 5 があるかを返す。
   *
   * @param suitMask 色ごとのビット（0〜7）
   * @param kind 牌の種類
   * @return その種類の赤の 5 があれば true。5 萬・5 筒・5 索でない種類は false
   * @throws IllegalArgumentException 色ごとのビットが 0〜7 の外のとき
   */
  public static boolean has(int suitMask, int kind) {
    return Tiles.contains(redKinds(suitMask), kind);
  }

  /** その色の 5 の種類。 */
  private static int five(int suit) {
    return suit * Tiles.SUIT_SIZE + Tiles.FIVE;
  }
}

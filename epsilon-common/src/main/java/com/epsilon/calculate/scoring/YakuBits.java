package com.epsilon.calculate.scoring;

/** 役集合を表す {@code long} の生成・正規化・集計を行う。 */
public final class YakuBits {

  private static final ScoringYaku[] YAKUS = ScoringYaku.values();
  private static final long YAKUMAN_MASK = buildYakumanMask();
  private static final long MENZEN_YAKU_MASK = buildNormalYakuMask(true);
  private static final long OPEN_YAKU_MASK = buildNormalYakuMask(false);

  private YakuBits() {}

  public static long when(ScoringYaku yaku, boolean established) {
    return established ? yaku.bit() : 0L;
  }

  public static boolean contains(long bits, ScoringYaku yaku) {
    return (bits & yaku.bit()) != 0L;
  }

  /**
   * 役満を優先し、副露時に翻を持たない門前役を除外する。
   *
   * <p>立直とダブル立直など、同じ系統の役の排他性は各役の判定処理が保証する。
   */
  public static long effective(long rawBits, boolean menzen) {
    long yakumanBits = rawBits & YAKUMAN_MASK;
    if (yakumanBits != 0L) {
      return yakumanBits;
    }
    return rawBits & (menzen ? MENZEN_YAKU_MASK : OPEN_YAKU_MASK);
  }

  public static boolean hasScoringYaku(long rawBits, boolean menzen) {
    return effective(rawBits, menzen) != 0L;
  }

  public static int han(long bits, boolean menzen) {
    int han = 0;
    long remaining = bits & ~YAKUMAN_MASK;
    while (remaining != 0L) {
      int ordinal = Long.numberOfTrailingZeros(remaining);
      han += YAKUS[ordinal].han(menzen);
      remaining &= remaining - 1;
    }
    return han;
  }

  public static int yakumanCount(long bits) {
    return Long.bitCount(bits & YAKUMAN_MASK);
  }

  public static boolean hasYakuman(long bits) {
    return (bits & YAKUMAN_MASK) != 0L;
  }

  public static long yakumans(long bits) {
    return bits & YAKUMAN_MASK;
  }

  public static long of(ScoringYaku... yakus) {
    long bits = 0L;
    for (ScoringYaku yaku : yakus) bits |= yaku.bit();
    return bits;
  }

  private static long buildYakumanMask() {
    long mask = 0L;
    for (ScoringYaku yaku : YAKUS) {
      if (yaku.isYakuman()) {
        mask |= yaku.bit();
      }
    }
    return mask;
  }

  private static long buildNormalYakuMask(boolean menzen) {
    long mask = 0L;
    for (ScoringYaku yaku : YAKUS) {
      if (!yaku.isYakuman() && yaku.han(menzen) > 0) {
        mask |= yaku.bit();
      }
    }
    return mask;
  }
}

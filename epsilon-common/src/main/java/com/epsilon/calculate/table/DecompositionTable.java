package com.epsilon.calculate.table;

/**
 * 萬子・筒子・索子・字牌ごとの局部和了分解と、事前計算済み待ち・役特徴を参照する。
 *
 * <p>返す値はすべて不変のテーブル上のプリミティブ型であり、配列の複製や候補オブジェクトの生成は行わない。
 */
public final class DecompositionTable {

  private static final int PATTERN_COUNT_BITS = 5;
  private static final int PATTERN_COUNT_MASK = (1 << PATTERN_COUNT_BITS) - 1;
  private static final int PATTERN_HAS_JANTOU = 1 << 31;

  private static final char[] SUIT_STATES_BY_COUNT_CODE =
      ShapeTableLoader.suitPatternStatesByCountCode();
  private static final char[] HONOR_STATES_BY_COUNT_CODE =
      ShapeTableLoader.honorPatternStatesByCountCode();
  private static final int[] SUIT_RANGES_BY_STATE = ShapeTableLoader.suitPatternRangesByState();
  private static final int[] HONOR_RANGES_BY_STATE = ShapeTableLoader.honorPatternRangesByState();
  private static final int[] SUIT_PATTERNS = ShapeTableLoader.suitPatterns();
  private static final int[] HONOR_PATTERNS = ShapeTableLoader.honorPatterns();
  private static final long[] SUIT_MACHI_TYPE_MATRICES = ShapeTableLoader.suitMachiTypeMatrices();
  private static final long[] HONOR_MACHI_TYPE_MATRICES = ShapeTableLoader.honorMachiTypeMatrices();
  private static final long[] SUIT_FEATURE_PACKETS = ShapeTableLoader.suitFeaturePackets();
  private static final long[] HONOR_FEATURE_PACKETS = ShapeTableLoader.honorFeaturePackets();

  private DecompositionTable() {}

  public static int suitCountCode(int[] tileCounts, int offset) {
    return ShapeTableLoader.suitCountCode(tileCounts, offset);
  }

  public static int honorCountCode(int[] tileCounts) {
    return ShapeTableLoader.honorCountCode(tileCounts);
  }

  public static int suitPatternRange(int countCode) {
    if (countCode < 0 || countCode >= SUIT_STATES_BY_COUNT_CODE.length) {
      throw new IllegalArgumentException("invalid suit count code: " + countCode);
    }
    return SUIT_RANGES_BY_STATE[SUIT_STATES_BY_COUNT_CODE[countCode]];
  }

  public static int honorPatternRange(int countCode) {
    if (countCode < 0 || countCode >= HONOR_STATES_BY_COUNT_CODE.length) {
      throw new IllegalArgumentException("invalid honor count code: " + countCode);
    }
    return HONOR_RANGES_BY_STATE[HONOR_STATES_BY_COUNT_CODE[countCode]];
  }

  public static int patternCount(int range) {
    return range & PATTERN_COUNT_MASK;
  }

  public static int patternStart(int range) {
    return (range & ~PATTERN_HAS_JANTOU) >>> PATTERN_COUNT_BITS;
  }

  public static int suitPattern(int index) {
    return SUIT_PATTERNS[index];
  }

  public static int honorPattern(int index) {
    return HONOR_PATTERNS[index];
  }

  public static long suitMachiTypeMatrix(int index) {
    return SUIT_MACHI_TYPE_MATRICES[index];
  }

  public static long honorMachiTypeMatrix(int index) {
    return HONOR_MACHI_TYPE_MATRICES[index];
  }

  public static long suitFeaturePacket(int index) {
    return SUIT_FEATURE_PACKETS[index];
  }

  public static long honorFeaturePacket(int index) {
    return HONOR_FEATURE_PACKETS[index];
  }
}

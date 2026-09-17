package com.epsilon.calculate.table;

/** 生成済みの一次元配列を、そのまま読み込む手牌構造テーブルの形式。 */
public final class ShapeTableFormat {
  public static final String RESOURCE_PATH = "/calculate/hand-shape-v4.bin";
  public static final int MAGIC = 0x4d4a4c54;
  public static final int VERSION = 4;
  public static final int SECTION_COUNT = 16;
  static final int SUIT_CODES = 1_953_125;
  static final int HONOR_CODES = 78_125;
  static final int INVALID_KEY = 0xff;

  private ShapeTableFormat() {}
}

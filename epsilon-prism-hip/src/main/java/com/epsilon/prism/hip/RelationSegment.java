package com.epsilon.prism.hip;

/** 関係コードの区分。宣言順がパックしたコードの下位からの順と、表の行の順を決める。 */
public enum RelationSegment {
  TYPE_PAIR,
  TILE_FIRST,
  TILE_SECOND,
  SEAT,
  RIICHI,
  TIME,
  LINK_SAFETY;

  public static final int ENTRIES = 16;

  /** この区分のコード 0 に対応する行。 */
  public int firstRow() {
    return ordinal() * ENTRIES;
  }
}

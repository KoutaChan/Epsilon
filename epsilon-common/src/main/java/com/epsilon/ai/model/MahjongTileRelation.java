package com.epsilon.ai.model;

import com.epsilon.core.Tile;

/**
 * 注意機構が参照する、2つの牌種の関係を表す。
 *
 * <p>同じ色の数牌では、参照先の数字から基準牌の数字を引いた符号付きの差を使い、上下の方向を区別する。異なる色の数牌と字牌は、順子を構成しないため別の分類で扱う。
 */
public enum MahjongTileRelation {
  SELF,
  SAME_SUIT_DELTA_MINUS_8,
  SAME_SUIT_DELTA_MINUS_7,
  SAME_SUIT_DELTA_MINUS_6,
  SAME_SUIT_DELTA_MINUS_5,
  SAME_SUIT_DELTA_MINUS_4,
  SAME_SUIT_DELTA_MINUS_3,
  SAME_SUIT_DELTA_MINUS_2,
  SAME_SUIT_DELTA_MINUS_1,
  SAME_SUIT_DELTA_PLUS_1,
  SAME_SUIT_DELTA_PLUS_2,
  SAME_SUIT_DELTA_PLUS_3,
  SAME_SUIT_DELTA_PLUS_4,
  SAME_SUIT_DELTA_PLUS_5,
  SAME_SUIT_DELTA_PLUS_6,
  SAME_SUIT_DELTA_PLUS_7,
  SAME_SUIT_DELTA_PLUS_8,
  CROSS_SUIT_SAME_RANK,
  CROSS_SUIT_OTHER,
  WIND_PAIR,
  DRAGON_PAIR,
  WIND_DRAGON,
  NUMBER_HONOR;

  private static final long[] RELATION_IDS = createRelationIds();

  public static MahjongTileRelation between(int queryTile, int keyTile) {
    if (queryTile == keyTile) {
      return SELF;
    }

    boolean queryNumber = Tile.isNumberTile(queryTile);
    boolean keyNumber = Tile.isNumberTile(keyTile);
    if (queryNumber && keyNumber) {
      if (Tile.suitOf(queryTile) == Tile.suitOf(keyTile)) {
        return sameSuitDelta(Tile.numberOf(keyTile) - Tile.numberOf(queryTile));
      }
      return Tile.numberOf(queryTile) == Tile.numberOf(keyTile)
          ? CROSS_SUIT_SAME_RANK
          : CROSS_SUIT_OTHER;
    }
    if (queryNumber || keyNumber) {
      return NUMBER_HONOR;
    }
    if (Tile.isWind(queryTile) && Tile.isWind(keyTile)) {
      return WIND_PAIR;
    }
    if (Tile.isDragon(queryTile) && Tile.isDragon(keyTile)) {
      return DRAGON_PAIR;
    }
    return WIND_DRAGON;
  }

  public static long[] relationIds() {
    return RELATION_IDS;
  }

  private static MahjongTileRelation sameSuitDelta(int delta) {
    return switch (delta) {
      case -8 -> SAME_SUIT_DELTA_MINUS_8;
      case -7 -> SAME_SUIT_DELTA_MINUS_7;
      case -6 -> SAME_SUIT_DELTA_MINUS_6;
      case -5 -> SAME_SUIT_DELTA_MINUS_5;
      case -4 -> SAME_SUIT_DELTA_MINUS_4;
      case -3 -> SAME_SUIT_DELTA_MINUS_3;
      case -2 -> SAME_SUIT_DELTA_MINUS_2;
      case -1 -> SAME_SUIT_DELTA_MINUS_1;
      case 1 -> SAME_SUIT_DELTA_PLUS_1;
      case 2 -> SAME_SUIT_DELTA_PLUS_2;
      case 3 -> SAME_SUIT_DELTA_PLUS_3;
      case 4 -> SAME_SUIT_DELTA_PLUS_4;
      case 5 -> SAME_SUIT_DELTA_PLUS_5;
      case 6 -> SAME_SUIT_DELTA_PLUS_6;
      case 7 -> SAME_SUIT_DELTA_PLUS_7;
      case 8 -> SAME_SUIT_DELTA_PLUS_8;
      default -> throw new IllegalStateException("Unexpected same-suit delta: " + delta);
    };
  }

  private static long[] createRelationIds() {
    long[] relationIds = new long[Tile.NUM_TILE_TYPES * Tile.NUM_TILE_TYPES];
    for (int queryTile = 0; queryTile < Tile.NUM_TILE_TYPES; queryTile++) {
      for (int keyTile = 0; keyTile < Tile.NUM_TILE_TYPES; keyTile++) {
        relationIds[queryTile * Tile.NUM_TILE_TYPES + keyTile] =
            between(queryTile, keyTile).ordinal();
      }
    }
    return relationIds;
  }
}

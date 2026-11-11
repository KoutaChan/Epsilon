package com.epsilon.agent.facts;

import com.epsilon.mahjong.tile.Tiles;

/**
 * Attention で使う、牌の種類どうしの関係。同じ色の数牌どうしは、キーの数字からクエリの数字を引いた符号つきの差（−8〜+8）で
 * 分ける。色の違う数牌どうしや字牌とは順子を作らないので、差を使わずに別の関係にする。
 *
 * <p>定数とその順、{@link #relationIds} の中身はネットワークとの契約。ネットワークは、重みの表と関係ごとのバイアスを定数の ordinal で引く。
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

  private static final long[] RELATION_IDS = relationTable();

  /**
   * クエリの牌から見た、キーの牌の関係を返す。
   *
   * @param query クエリの牌の種類
   * @param key キーの牌の種類
   * @return 2 つの牌の関係。同じ種類なら {@link #SELF}
   */
  public static MahjongTileRelation between(int query, int key) {
    if (query == key) {
      return SELF;
    }
    Tiles.Group queryGroup = Tiles.Group.of(query);
    Tiles.Group keyGroup = Tiles.Group.of(key);
    if (queryGroup.suited() && keyGroup.suited()) {
      if (queryGroup == keyGroup) {
        return sameSuit(Tiles.number(key) - Tiles.number(query));
      }
      return Tiles.number(query) == Tiles.number(key) ? CROSS_SUIT_SAME_RANK : CROSS_SUIT_OTHER;
    }
    if (queryGroup.suited() || keyGroup.suited()) {
      return NUMBER_HONOR;
    }
    if (queryGroup == keyGroup) {
      return queryGroup == Tiles.Group.WIND ? WIND_PAIR : DRAGON_PAIR;
    }
    return WIND_DRAGON;
  }

  /** 牌の種類の全ての組について、関係の ordinal を並べた表を返す。添字はクエリ × 34 ＋ キー。呼ぶたびに新しい配列を返す。 */
  public static long[] relationIds() {
    return RELATION_IDS.clone();
  }

  private static MahjongTileRelation sameSuit(int delta) {
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
      default -> throw new IllegalArgumentException("No same-suit delta " + delta);
    };
  }

  private static long[] relationTable() {
    long[] table = new long[Tiles.KINDS * Tiles.KINDS];
    for (int query = 0; query < Tiles.KINDS; query++) {
      for (int key = 0; key < Tiles.KINDS; key++) {
        table[query * Tiles.KINDS + key] = between(query, key).ordinal();
      }
    }
    return table;
  }
}

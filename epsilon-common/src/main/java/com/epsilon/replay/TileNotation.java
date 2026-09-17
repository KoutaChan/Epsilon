package com.epsilon.replay;

import com.epsilon.core.Tile;

/** mjai / 雀魂の表記を牌種と赤牌属性へ解釈する。物理牌IDの割り当ては読み込み処理が管理する。 */
public final class TileNotation {
  private static final String HONORS = "ESWNPFC";

  private TileNotation() {}

  public static int parseTileType(String tile) {
    if (tile != null && tile.length() == 1) {
      int honor = HONORS.indexOf(tile.charAt(0));
      if (honor >= 0) return 27 + honor;
    }
    if (tile == null || tile.length() < 2 || tile.length() > 3) throw invalidTileNotation(tile);
    int number = tile.charAt(0) - '0';
    char suit = tile.charAt(1);
    boolean suffix = tile.length() == 3;
    if (suffix && (tile.charAt(2) != 'r' || number != 5 || suit == 'z'))
      throw invalidTileNotation(tile);
    if (suit == 'z') {
      if (number < 1 || number > 7) throw invalidTileNotation(tile);
      return 26 + number;
    }
    int suitIndex = "mps".indexOf(suit);
    if (suitIndex < 0 || number < 0 || number > 9) throw invalidTileNotation(tile);
    return suitIndex * 9 + (number == 0 ? 4 : number - 1);
  }

  /** 同じ牌種・赤属性を指す代表 ID。牌譜内で一意な物理 ID ではない。 */
  public static int representativeTileId(String tile) {
    int type = parseTileType(tile);
    return type * 4 + (hasRedMarker(tile) ? 0 : 1);
  }

  public static boolean isRed(String tile) {
    parseTileType(tile);
    return hasRedMarker(tile);
  }

  public static String formatPhysicalTile(int id) {
    int type = Tile.typeOf(id);
    if (type >= 27) return String.valueOf(HONORS.charAt(type - 27));
    return (type % 9 + 1) + String.valueOf("mps".charAt(type / 9)) + (Tile.isAka(id) ? "r" : "");
  }

  private static boolean hasRedMarker(String tile) {
    return tile.charAt(0) == '0' || tile.endsWith("r");
  }

  private static IllegalArgumentException invalidTileNotation(String tile) {
    return new IllegalArgumentException("Invalid tile notation: " + tile);
  }
}

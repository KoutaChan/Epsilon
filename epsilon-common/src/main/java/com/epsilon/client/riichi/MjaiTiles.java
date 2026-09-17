package com.epsilon.client.riichi;

import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.google.gson.JsonArray;

/** MJAI の牌文字列と、観測手牌内だけで使う物理コピー ID を対応付ける。 */
final class MjaiTiles {
  private static final String[] HONORS = {"E", "S", "W", "N", "P", "F", "C"};

  private MjaiTiles() {}

  /** 通常牌は代表コピー、赤牌は赤コピー、非公開牌は -1 に変換する。 */
  static int parse(String tile) {
    for (int index = 0; index < HONORS.length; index++) {
      if (HONORS[index].equals(tile)) return (27 + index) * 4;
    }
    if (tile.equals("?")) return -1;
    if (tile.length() < 2 || tile.length() > 3) {
      throw new IllegalArgumentException("Invalid MJAI tile: " + tile);
    }
    int number = tile.charAt(0) - '1';
    int suit =
        switch (tile.charAt(1)) {
          case 'm' -> 0;
          case 'p' -> 1;
          case 's' -> 2;
          default -> -1;
        };
    boolean red = tile.length() == 3;
    if (number < 0 || number > 8 || suit < 0 || (red && (number != 4 || tile.charAt(2) != 'r'))) {
      throw new IllegalArgumentException("Invalid MJAI tile: " + tile);
    }
    return (suit * 9 + number) * 4 + (number == 4 && !red ? 1 : 0);
  }

  static String format(int physicalTileId) {
    int type = Tile.typeOf(physicalTileId);
    if (type >= 27) return HONORS[type - 27];
    return Integer.toString(type % 9 + 1)
        + "mps".charAt(type / 9)
        + (Tile.isAka(physicalTileId) ? "r" : "");
  }

  /** 同じ通常牌のコピーを、現在の手牌と衝突しない ID に割り当てる。 */
  static int nextPhysicalTile(Hand hand, String tile) {
    int first = parse(tile);
    if (first < 0 || Tile.isAka(first)) return first;
    int limit = (Tile.typeOf(first) + 1) * 4;
    for (int candidate = first; candidate < limit; candidate++) {
      if (!hand.containsPhysicalTile(candidate)) return candidate;
    }
    throw new IllegalArgumentException("MJAI draw exceeds four copies: " + tile);
  }

  static int[] initialHand(JsonArray tiles) {
    int[] copies = new int[Tile.NUM_TILE_TYPES];
    int[] result = new int[tiles.size()];
    for (int index = 0; index < result.length; index++) {
      int first = parse(tiles.get(index).getAsString());
      if (first < 0) throw new IllegalArgumentException("Own MJAI hand is masked");
      int type = Tile.typeOf(first);
      result[index] = Tile.isAka(first) ? first : first + copies[type]++;
      if (Tile.typeOf(result[index]) != type) {
        throw new IllegalArgumentException("MJAI hand exceeds four copies");
      }
    }
    return result;
  }

  /** 自家の副露で実際に取り除く牌だけを、現在の手牌から解決する。 */
  static int[] consumed(Hand hand, JsonArray tiles) {
    int[] result = new int[tiles.size()];
    for (int index = 0; index < result.length; index++) {
      int representative = parse(tiles.get(index).getAsString());
      int ordinal = 0;
      for (int previous = 0; previous < index; previous++) {
        if (Tile.typeOf(result[previous]) == Tile.typeOf(representative)
            && Tile.isAka(result[previous]) == Tile.isAka(representative)) ordinal++;
      }
      result[index] =
          hand.physicalTileId(Tile.typeOf(representative), Tile.isAka(representative), ordinal);
    }
    return result;
  }

  static boolean containsAka(JsonArray tiles, boolean physicalIds) {
    for (var tile : tiles) {
      if (Tile.isAka(physicalIds ? tile.getAsInt() : parse(tile.getAsString()))) return true;
    }
    return false;
  }

  /** 最大四枚の消費牌を、通常コピーの違いを除いた多重集合で比較する。 */
  static boolean sameConsumed(JsonArray physicalIds, JsonArray tileNames) {
    if (physicalIds.size() != tileNames.size()) return false;
    for (var physical : physicalIds) {
      int count = 0;
      for (var candidate : physicalIds) {
        if (sameTile(physical.getAsInt(), candidate.getAsInt())) count++;
      }
      for (var name : tileNames) {
        if (sameTile(physical.getAsInt(), parse(name.getAsString()))) count--;
      }
      if (count != 0) return false;
    }
    return true;
  }

  static boolean sameTile(int left, int right) {
    return Tile.typeOf(left) == Tile.typeOf(right) && Tile.isAka(left) == Tile.isAka(right);
  }
}

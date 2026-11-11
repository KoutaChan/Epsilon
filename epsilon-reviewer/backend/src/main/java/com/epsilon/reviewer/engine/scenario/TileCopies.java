package com.epsilon.reviewer.engine.scenario;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.reviewer.engine.view.TileCode;

/**
 * 牌の記号ごとの写し（物理牌）と、そのうち使っている物理牌を持つ。ある記号の写しとは、ルールの牌の集合のうち、種類と赤かどうかが
 * 記号と同じ物理牌のこと。割り当て方は mahjong の {@code TileCopies} と同じで、使っていない写しを番号の小さい順に使う。
 */
final class TileCopies {
  private final TileSet set;
  private final boolean[] used = new boolean[Tiles.PHYSICAL_IDS];

  TileCopies(Variant variant) {
    set = variant.tiles();
  }

  /** 指定の牌の写しの数。protocol/result.ts の tileCopies と同じ値になる。 */
  static int count(Variant variant, String code) {
    return com.epsilon.mahjong.record.TileCopies.copies(variant, TileCode.face(code));
  }

  /** 物理牌が指定の牌の写しか。 */
  boolean isCopy(int physicalId, String code) {
    Face face = TileCode.face(code);
    return Tiles.kind(physicalId) == face.kind() && set.red(physicalId) == face.red();
  }

  void use(int physicalId) {
    used[physicalId] = true;
  }

  void free(int physicalId) {
    used[physicalId] = false;
  }

  /**
   * 指定の牌の写しのうち、使っていないもので番号の最も小さいものを使う。
   *
   * @throws IllegalStateException 使っていない写しが残っていない
   */
  int take(String code) {
    int kind = TileCode.kind(code);
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(kind, copy);
      if (set.contains(tile) && !used[tile] && isCopy(tile, code)) {
        used[tile] = true;
        return tile;
      }
    }
    throw new IllegalStateException("No unused copy of " + code + " is left");
  }
}

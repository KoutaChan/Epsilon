package com.epsilon.reviewer.engine.scenario;

import com.epsilon.core.Tile;
import com.epsilon.reviewer.engine.view.TileCode;
import java.util.stream.IntStream;

/**
 * 牌の記号ごとの写し（物理牌）と、そのうち使っている物理牌。記号の写しは、牌種と赤の印が記号と同じ物理牌（赤の 5 は 1 枚、ほかの 5 は 3 枚、
 * ほかは 4 枚）。
 */
final class TileCopies {
  private final boolean[] used = new boolean[Tile.NUM_TILES];

  /** 牌 code の写しの数（protocol/result.ts の tileCopies と同じ）。 */
  static int count(String code) {
    return (int) copies(code).count();
  }

  static boolean isCopy(int physicalTileId, String code) {
    return Tile.typeOf(physicalTileId) == TileCode.type(code)
        && Tile.isAka(physicalTileId) == TileCode.red(code);
  }

  void use(int... physicalTileIds) {
    for (int id : physicalTileIds) used[id] = true;
  }

  void free(int physicalTileId) {
    used[physicalTileId] = false;
  }

  /**
   * 牌 code の使っていない一番小さい写しを使う。
   *
   * @throws IllegalStateException 使っていない写しが無い
   */
  int take(String code) {
    int id =
        copies(code)
            .filter(copy -> !used[copy])
            .findFirst()
            .orElseThrow(() -> new IllegalStateException("No unused copy of " + code + " is left"));
    used[id] = true;
    return id;
  }

  /** 牌 code の写し（小さい順）。 */
  private static IntStream copies(String code) {
    int first = TileCode.type(code) * Tile.TILES_PER_TYPE;
    return IntStream.range(first, first + Tile.TILES_PER_TYPE).filter(id -> isCopy(id, code));
  }
}

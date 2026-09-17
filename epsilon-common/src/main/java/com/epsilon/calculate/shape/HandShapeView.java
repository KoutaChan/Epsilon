package com.epsilon.calculate.shape;

import com.epsilon.core.Tile;

/** 牌種の形状と確定面子数だけを読む契約。赤牌・副露内容・局況は含まない。 */
public interface HandShapeView {
  /** 所有状態を再利用先へ転記する。符号化方式はこの契約から公開しない。 */
  void copyShapeInto(HandShapeState destination);

  /** 物理牌の集計へ、牌種枚数だけを転記する。形状索引は不要な得点評価に使う。 */
  void copyTileCountsInto(HandShapeState.TileCountBuffer destination);

  int count(int tileType);

  int concealedTileCount();

  int meldCount();

  long concealedTileTypeMask();

  long concealedPairTileTypeMask();

  default int distinctConcealedTileTypeCount() {
    return Long.bitCount(concealedTileTypeMask());
  }

  default int concealedPairTileTypeCount() {
    return Long.bitCount(concealedPairTileTypeMask());
  }

  default int kokushiTileTypeCount() {
    return Long.bitCount(concealedTileTypeMask() & Tile.TERMINAL_OR_HONOR_TYPE_MASK);
  }

  default int kokushiPairTileTypeCount() {
    return Long.bitCount(concealedPairTileTypeMask() & Tile.TERMINAL_OR_HONOR_TYPE_MASK);
  }

  default int effectiveTileCount() {
    return concealedTileCount() + 3 * meldCount();
  }

  /** 保存や外部からの入力を扱うときだけ配列へコピーする。探索中の複製には使用しない。 */
  default void copyConcealedTileCountsTo(int[] destination) {
    if (destination.length < Tile.NUM_TILE_TYPES) {
      throw new IllegalArgumentException("destination must contain 34 entries");
    }
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      destination[tile] = count(tile);
    }
  }

  default int[] copyConcealedTileCounts() {
    int[] result = new int[Tile.NUM_TILE_TYPES];
    copyConcealedTileCountsTo(result);
    return result;
  }
}

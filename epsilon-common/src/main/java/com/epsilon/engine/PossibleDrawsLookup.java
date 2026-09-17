package com.epsilon.engine;

import com.epsilon.calculate.shape.HandShapeView;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/** 手牌形から求めた候補牌のうち、自分の確定面子を含めると5枚目になる牌種を除く。 */
final class PossibleDrawsLookup {
  private static final long ALL_TYPES = (1L << Tile.NUM_TILE_TYPES) - 1;

  private PossibleDrawsLookup() {}

  /** 公開牌は判定に使わない。待ち牌が場にすべて出ている場合と、自家が4枚持っている場合を区別する。 */
  static long possibleDraws(HandShapeView shape, HandView hand) {
    long possible = ALL_TYPES;
    for (long pairs = shape.concealedPairTileTypeMask(); pairs != 0L; pairs &= pairs - 1) {
      int tile = Long.numberOfTrailingZeros(pairs);
      if (shape.count(tile) >= Tile.TILES_PER_TYPE) possible &= ~(1L << tile);
    }
    long fixedTypes = 0L;
    for (int index = 0; index < hand.meldCount(); index++) {
      fixedTypes |= hand.meld(index).tileTypeMask();
    }
    for (; fixedTypes != 0L; fixedTypes &= fixedTypes - 1) {
      int tile = Long.numberOfTrailingZeros(fixedTypes);
      long bit = 1L << tile;
      int owned = shape.count(tile);
      for (int index = 0; index < hand.meldCount(); index++) {
        Meld meld = hand.meld(index);
        if ((meld.tileTypeMask() & bit) != 0L) {
          owned += meld.type() == Meld.Type.CHI ? 1 : meld.size();
        }
      }
      if (owned >= Tile.TILES_PER_TYPE) possible &= ~bit;
    }
    return possible;
  }
}

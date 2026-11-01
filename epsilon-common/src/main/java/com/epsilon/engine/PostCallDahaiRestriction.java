package com.epsilon.engine;

import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/**
 * チー・ポン直後の喰い替え禁止牌。鳴いた後の打牌（{@link PostCallDiscards}）と、鳴きの後に打てる牌が残るかの応答の検査
 * （{@link ResponseActionGenerator}）が読む。
 *
 * @param forbiddenMask ビット {@code tileType} が1なら、その牌種を直後に捨てられない34-ビットマスク
 */
record PostCallDahaiRestriction(long forbiddenMask) {

  private static final PostCallDahaiRestriction[] PON = new PostCallDahaiRestriction[34];
  private static final PostCallDahaiRestriction[] CHI = new PostCallDahaiRestriction[81];

  static {
    for (int tileType = 0; tileType < PON.length; tileType++) {
      PON[tileType] = new PostCallDahaiRestriction(tileMask(tileType));
    }
    for (int sequenceBase = 0; sequenceBase < 27; sequenceBase++) {
      if (Tile.numberOf(sequenceBase) > 6) {
        continue;
      }
      for (int calledOffset = 0; calledOffset < 3; calledOffset++) {
        int calledTile = sequenceBase + calledOffset;
        CHI[sequenceBase * 3 + calledOffset] =
            new PostCallDahaiRestriction(chiForbiddenMask(sequenceBase, calledTile));
      }
    }
  }

  /**
   * 鳴いた副露の直後の喰い替え禁止集合を返す。
   *
   * @param call チーかポン
   * @return チーは現物と筋、ポンは同じ牌を禁じる制約
   * @throws IllegalArgumentException チー・ポンでないとき
   */
  static PostCallDahaiRestriction after(Meld call) {
    return switch (call.type()) {
      case CHI -> afterChi(call.baseTileType(), call.calledTileType());
      case PON -> afterPon(call.calledTileType());
      default -> throw new IllegalArgumentException("post-call discards require CHI or PON: " + call);
    };
  }

  /**
   * ポン直後の喰い替え禁止集合を返す。
   *
   * @param calledTile 鳴いた牌種ID
   * @return 鳴いた牌と同牌を禁じる制約
   */
  static PostCallDahaiRestriction afterPon(int calledTile) {
    Tile.requireValidType(calledTile, "calledTile");
    return PON[calledTile];
  }

  /**
   * チー直後の喰い替え禁止集合を返す。
   *
   * @param sequenceBaseTile 順子で最も小さい牌種
   * @param calledTile 鳴いた牌種ID
   * @return 現物と筋の喰い替えを禁じる制約
   */
  static PostCallDahaiRestriction afterChi(int sequenceBaseTile, int calledTile) {
    if (sequenceBaseTile < Tile.M1
        || sequenceBaseTile >= Tile.TON
        || Tile.numberOf(sequenceBaseTile) > 6
        || calledTile < sequenceBaseTile
        || calledTile > sequenceBaseTile + 2) {
      throw new IllegalArgumentException(
          "invalid chi sequence/called tile: " + sequenceBaseTile + '/' + calledTile);
    }
    PostCallDahaiRestriction cached = CHI[sequenceBaseTile * 3 + calledTile - sequenceBaseTile];
    if (cached == null) {
      throw new IllegalArgumentException(
          "invalid chi sequence/called tile: " + sequenceBaseTile + '/' + calledTile);
    }
    return cached;
  }

  private static long chiForbiddenMask(int sequenceBaseTile, int calledTile) {
    long mask = tileMask(calledTile);
    int suitBase = Tile.suitOf(calledTile) * 9;
    int sequenceHigh = sequenceBaseTile + 2;
    if (calledTile == sequenceBaseTile) {
      int suji = sequenceHigh + 1;
      if (suji < suitBase + 9) {
        mask |= tileMask(suji);
      }
    } else if (calledTile == sequenceHigh) {
      int suji = sequenceBaseTile - 1;
      if (suji >= suitBase) {
        mask |= tileMask(suji);
      }
    }
    return mask;
  }

  private static long tileMask(int tileType) {
    return 1L << tileType;
  }
}

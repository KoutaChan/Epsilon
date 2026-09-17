package com.epsilon.engine;

import com.epsilon.core.Tile;

/**
 * チー・ポン直後の喰い替え禁止牌。
 *
 * @param forbiddenMask ビット {@code tileType} が1なら、その牌種を直後に捨てられない34-ビットマスク
 */
public record PostCallDahaiRestriction(long forbiddenMask) {

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
   * ポン直後の喰い替え禁止集合を返す。
   *
   * @param calledTile 鳴いた牌種ID
   * @return 鳴いた牌と同牌を禁じる制約
   */
  public static PostCallDahaiRestriction afterPon(int calledTile) {
    Tile.requireValidType(calledTile, "calledTile");
    return PON[calledTile];
  }

  /**
   * チー直後の喰い替え禁止集合を返す。
   *
   * @param chiTiles 生成した順子3牌種
   * @param calledTile 鳴いた牌種ID
   * @return 現物と筋の喰い替えを禁じる制約
   */
  public static PostCallDahaiRestriction afterChi(int[] chiTiles, int calledTile) {
    int sequenceBaseTile = Math.min(chiTiles[0], Math.min(chiTiles[1], chiTiles[2]));
    return afterChi(sequenceBaseTile, calledTile);
  }

  public static PostCallDahaiRestriction afterChi(int sequenceBaseTile, int calledTile) {
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

  /**
   * 指定牌種が直後の打牌として禁止されるかを返す。
   *
   * @param tileType 判定する牌種ID
   * @return 禁止牌なら {@code true}
   */
  public boolean forbids(int tileType) {
    return (forbiddenMask & tileMask(tileType)) != 0L;
  }

  private static long tileMask(int tileType) {
    return 1L << tileType;
  }
}

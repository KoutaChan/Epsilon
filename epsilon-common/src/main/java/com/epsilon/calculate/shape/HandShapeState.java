package com.epsilon.calculate.shape;

import com.epsilon.core.Tile;

/**
 * 実際の手牌と、行動候補の評価や探索で使う仮の手牌を、共通の牌種別枚数として保持する。枚数は配列ではなくビット列などの数値で保持し、牌の追加・削除では影響する色の情報だけを更新する。
 *
 * <p>所有者以外は変更してはならない。{@code load}は値をコピーし、元の手牌への参照を保持しない。
 */
public final class HandShapeState implements HandShapeView {
  private static final int[] POW5 = {1, 5, 25, 125, 625, 3125, 15625, 78125, 390625};
  private long low;
  private long high;
  private int man;
  private int pin;
  private int sou;
  private int honors;
  private int tileCount;
  private int fixedMelds;
  private long types;
  private long pairs;

  public void load(HandShapeView source) {
    source.copyShapeInto(this);
  }

  @Override
  public void copyShapeInto(HandShapeState destination) {
    destination.low = low;
    destination.high = high;
    destination.man = man;
    destination.pin = pin;
    destination.sou = sou;
    destination.honors = honors;
    destination.tileCount = tileCount;
    destination.fixedMelds = fixedMelds;
    destination.types = types;
    destination.pairs = pairs;
  }

  /** 得点評価の物理牌集計へ、手牌（副露・暗槓を除く）の枚数だけを転記する。 */
  @Override
  public void copyTileCountsInto(TileCountBuffer destination) {
    destination.low = low;
    destination.high = high;
    if (containsInvalidCount(low) || containsInvalidCount(high)) {
      throw new IllegalArgumentException("concealed tile count exceeds four");
    }
  }

  /** 検証を先に完了し、不正な配列で現在の状態を壊さない。 */
  public void load(int[] counts, int meldCount) {
    requireMeldCount(meldCount);
    if (counts.length != Tile.NUM_TILE_TYPES) {
      throw new IllegalArgumentException("counts must contain 34 entries");
    }
    int total = 0;
    for (int count : counts) {
      if (count < 0 || count > Tile.TILES_PER_TYPE) {
        throw new IllegalArgumentException("tile count must be between zero and four");
      }
      total += count;
    }
    if (total + 3 * meldCount > 14) {
      throw new IllegalArgumentException("hand must contain at most 14 effective tiles");
    }
    clear();
    fixedMelds = meldCount;
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      int count = counts[tile];
      if (count != 0) {
        setCount(tile, 0, count);
      }
    }
  }

  public void clear() {
    low = high = types = pairs = 0L;
    man = pin = sou = honors = tileCount = fixedMelds = 0;
  }

  public void setMeldCount(int count) {
    requireMeldCount(count);
    fixedMelds = count;
  }

  public void add(int tileType) {
    Tile.requireValidType(tileType, "tileType");
    int previous = count(tileType);
    if (previous == Tile.TILES_PER_TYPE) {
      throw new IllegalStateException("Cannot add fifth copy of " + Tile.name(tileType));
    }
    setCount(tileType, previous, previous + 1);
  }

  public void remove(int tileType) {
    Tile.requireValidType(tileType, "tileType");
    int previous = count(tileType);
    if (previous == 0) {
      throw new IllegalStateException("Tile is not in hand: " + Tile.name(tileType));
    }
    setCount(tileType, previous, previous - 1);
  }

  private void setCount(int tile, int before, int after) {
    int shift = (tile < 21 ? tile : tile - 21) * 3;
    long mask = 7L << shift;
    if (tile < 21) low = (low & ~mask) | (long) after << shift;
    else high = (high & ~mask) | (long) after << shift;
    int delta = (after - before) * POW5[tile % 9];
    switch (tile / 9) {
      case 0 -> man += delta;
      case 1 -> pin += delta;
      case 2 -> sou += delta;
      case 3 -> honors += delta;
      default -> throw new AssertionError(tile);
    }
    tileCount += after - before;
    long bit = 1L << tile;
    types = after == 0 ? types & ~bit : types | bit;
    pairs = after < 2 ? pairs & ~bit : pairs | bit;
  }

  private static void requireMeldCount(int count) {
    if (count < 0 || count > 4) {
      throw new IllegalArgumentException("meld count must be between zero and four");
    }
  }

  @Override
  public int count(int tileType) {
    return countFrom(low, high, tileType);
  }

  @Override
  public int concealedTileCount() {
    return tileCount;
  }

  @Override
  public int meldCount() {
    return fixedMelds;
  }

  public long packedConcealedTileCountsLow() {
    return low;
  }

  public long packedConcealedTileCountsHigh() {
    return high;
  }

  public int manzuCountCode() {
    return man;
  }

  public int pinzuCountCode() {
    return pin;
  }

  public int souzuCountCode() {
    return sou;
  }

  public int honorCountCode() {
    return honors;
  }

  @Override
  public long concealedTileTypeMask() {
    return types;
  }

  @Override
  public long concealedPairTileTypeMask() {
    return pairs;
  }

  private static int countFrom(long low, long high, int tileType) {
    return (int) ((tileType < 21 ? low >>> (tileType * 3) : high >>> ((tileType - 21) * 3)) & 7L);
  }

  private static boolean containsInvalidCount(long counts) {
    return ((counts >>> 2) & (counts | counts >>> 1) & 0x1249249249249249L) != 0L;
  }

  /** 手牌（副露・暗槓を除く）と確定面子の物理枚数を集計する再利用領域。符号化は外へ公開しない。 */
  public static final class TileCountBuffer {
    private long low;
    private long high;

    public int count(int tileType) {
      return countFrom(low, high, tileType);
    }

    public void add(int tileType) {
      if (count(tileType) >= Tile.TILES_PER_TYPE) {
        throw new IllegalArgumentException("more than four physical tiles of type " + tileType);
      }
      if (tileType < 21) low += 1L << (tileType * 3);
      else high += 1L << ((tileType - 21) * 3);
    }
  }
}

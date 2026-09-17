package com.epsilon.core;

/**
 * 局内で公開されたドラ表示牌、表ドラ、裏ドラ、および牌種ごとのドラ枚数を管理する。
 *
 * <p>{@link Wall}は王牌を含む物理的な牌の配置を保持し、このクラスは表示牌の公開状態を保持する。
 * 通常の処理では内部配列を公開せず、局面を独立して保存するときにだけコピーコンストラクターを使う。
 */
public final class DoraState {

  public static final int MAX_INDICATORS = 5;

  private static final int TILE_BITS = 6;
  private static final long TILE_MASK = (1L << TILE_BITS) - 1L;

  private final byte[] doraMultiplicity = new byte[Tile.NUM_TILE_TYPES];
  private final byte[] indicatorMultiplicity = new byte[Tile.NUM_TILE_TYPES];
  private long indicatorTileTypesPacked;
  private long doraTileTypesPacked;
  private long uraDoraTileTypesPacked;
  private int indicatorCount;
  private int visibleIndicatorAkaMask;
  private int uraKnownMask;

  DoraState() {}

  DoraState(DoraState other) {
    indicatorTileTypesPacked = other.indicatorTileTypesPacked;
    doraTileTypesPacked = other.doraTileTypesPacked;
    uraDoraTileTypesPacked = other.uraDoraTileTypesPacked;
    indicatorCount = other.indicatorCount;
    visibleIndicatorAkaMask = other.visibleIndicatorAkaMask;
    uraKnownMask = other.uraKnownMask;
    System.arraycopy(other.doraMultiplicity, 0, doraMultiplicity, 0, Tile.NUM_TILE_TYPES);
    System.arraycopy(other.indicatorMultiplicity, 0, indicatorMultiplicity, 0, Tile.NUM_TILE_TYPES);
  }

  /** 現在公開されているドラ表示牌数。 */
  public int indicatorCount() {
    return indicatorCount;
  }

  /** 表示順のドラ表示牌種。 */
  public int indicatorTileType(int index) {
    requireRevealedIndex(index);
    return packedTile(indicatorTileTypesPacked, index);
  }

  /** 表示順の表ドラ牌種。 */
  public int doraTileType(int index) {
    requireRevealedIndex(index);
    return packedTile(doraTileTypesPacked, index);
  }

  /** 指定牌種が表ドラになる表示枚数。 */
  public int doraMultiplicity(int tileType) {
    return doraMultiplicity[tileType] & 0xff;
  }

  /** 指定牌種がドラ表示牌として公開されている枚数。 */
  public int indicatorMultiplicity(int tileType) {
    return indicatorMultiplicity[tileType] & 0xff;
  }

  /** 公開済みドラ表示牌に含まれる赤5牌種マスク。 */
  public int visibleIndicatorAkaMask() {
    return visibleIndicatorAkaMask;
  }

  /** 表ドラの表示順に詰めたビット列。末尾の未使用ビットは常に0。 */
  public long doraTileTypesPacked() {
    return doraTileTypesPacked;
  }

  /**
   * 裏ドラの表示順に詰めたビット列。
   *
   * <p>非公開情報のため、プレイヤーや入力の符号化処理に渡してはならない。和了の評価と点数の精算だけに使用する。
   */
  public long uraDoraTileTypesPacked() {
    return uraDoraTileTypesPacked;
  }

  /** 裏ドラ牌種が判明している表示枠のビットマスク。 */
  public int uraKnownMask() {
    return uraKnownMask;
  }

  /** 指定したビット列から表示位置の牌種を取り出す。 */
  public static int packedTile(long packed, int index) {
    return (int) (packed >>> (index * TILE_BITS) & TILE_MASK);
  }

  void reset() {
    for (int index = 0; index < indicatorCount; index++) {
      indicatorMultiplicity[packedTile(indicatorTileTypesPacked, index)] = 0;
      doraMultiplicity[packedTile(doraTileTypesPacked, index)] = 0;
    }
    indicatorTileTypesPacked = 0L;
    doraTileTypesPacked = 0L;
    uraDoraTileTypesPacked = 0L;
    indicatorCount = 0;
    visibleIndicatorAkaMask = AkaTileMask.EMPTY;
    uraKnownMask = 0;
  }

  void reveal(
      int indicatorTileType, boolean indicatorIsAka, int uraIndicatorTileType, boolean uraKnown) {
    if (indicatorCount >= MAX_INDICATORS) {
      throw new IllegalStateException("too many dora indicators");
    }
    requireTileType(indicatorTileType);
    if (uraKnown) requireTileType(uraIndicatorTileType);

    int index = indicatorCount;
    int doraTileType = Tile.doraFrom(indicatorTileType);
    indicatorTileTypesPacked = appendPacked(indicatorTileTypesPacked, index, indicatorTileType);
    doraTileTypesPacked = appendPacked(doraTileTypesPacked, index, doraTileType);
    indicatorMultiplicity[indicatorTileType]++;
    doraMultiplicity[doraTileType]++;
    if (indicatorIsAka) {
      visibleIndicatorAkaMask = AkaTileMask.includeTile(visibleIndicatorAkaMask, indicatorTileType);
    }
    if (uraKnown) {
      uraDoraTileTypesPacked =
          appendPacked(uraDoraTileTypesPacked, index, Tile.doraFrom(uraIndicatorTileType));
      uraKnownMask |= 1 << index;
    }
    indicatorCount = index + 1;
  }

  private static long appendPacked(long packed, int index, int tileType) {
    return packed | (long) tileType << (index * TILE_BITS);
  }

  private void requireRevealedIndex(int index) {
    if (index < 0 || index >= indicatorCount) throw new IndexOutOfBoundsException(index);
  }

  private static void requireTileType(int tileType) {
    if (tileType < 0 || tileType >= Tile.NUM_TILE_TYPES) {
      throw new IllegalArgumentException("invalid tile type: " + tileType);
    }
  }
}

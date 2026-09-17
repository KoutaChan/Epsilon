package com.epsilon.core;

import com.epsilon.calculate.shape.HandShapeState;
import java.util.Arrays;

/** 手牌を表現するクラス。 牌種の形状と物理牌・副露の所有状態を分離して管理する。 */
public final class Hand implements HandView {

  private static final int MELD_ID_BITS = 14;
  private final HandShapeState shape = new HandShapeState();

  @Override
  public void copyShapeInto(HandShapeState destination) {
    shape.copyShapeInto(destination);
  }

  @Override
  public void copyTileCountsInto(HandShapeState.TileCountBuffer destination) {
    shape.copyTileCountsInto(destination);
  }

  private final byte[] concealedPhysicalCopyMasks;
  private final Meld[] melds;
  private final int[] meldCallAfterRiverIndexes;
  private final int[] meldKanAfterRiverIndexes;
  private final int[] meldCreationEvents;
  private final int[] meldKakanEvents;
  private int concealedAkaMask;
  private int openMeldCount;
  private int kanCount;
  private int ownedAkaMask;
  private long canonicalMeldSignature;

  /** 空の手牌（副露・暗槓を除く）と面子リストを作る。 */
  public Hand() {
    this.concealedPhysicalCopyMasks = new byte[Tile.NUM_TILE_TYPES];
    this.melds = new Meld[4];
    this.meldCallAfterRiverIndexes = new int[] {-1, -1, -1, -1};
    this.meldKanAfterRiverIndexes = new int[] {-1, -1, -1, -1};
    this.meldCreationEvents = new int[] {-1, -1, -1, -1};
    this.meldKakanEvents = new int[] {-1, -1, -1, -1};
    this.concealedAkaMask = AkaTileMask.EMPTY;
  }

  /**
   * 手牌（副露・暗槓を除く）、赤牌状態、面子発生記録を複製する。
   *
   * @param other 複製元の手牌
   */
  public Hand(Hand other) {
    shape.load(other);
    this.concealedPhysicalCopyMasks = other.concealedPhysicalCopyMasks.clone();
    this.melds = other.melds.clone();
    this.meldCallAfterRiverIndexes = other.meldCallAfterRiverIndexes.clone();
    this.meldKanAfterRiverIndexes = other.meldKanAfterRiverIndexes.clone();
    this.meldCreationEvents = other.meldCreationEvents.clone();
    this.meldKakanEvents = other.meldKakanEvents.clone();
    this.concealedAkaMask = other.concealedAkaMask;
    this.openMeldCount = other.openMeldCount;
    this.kanCount = other.kanCount;
    this.ownedAkaMask = other.ownedAkaMask;
    this.canonicalMeldSignature = other.canonicalMeldSignature;
  }

  /**
   * 通常牌として指定牌種を1枚追加する。
   *
   * @param tileType 追加する牌種
   */
  public void add(int tileType) {
    addCount(tileType);
  }

  /** スナップショットの保存や対局の分岐に使う、元の手牌と独立したコピー。 */
  public Hand snapshotCopy() {
    return new Hand(this);
  }

  /**
   * 物理牌 ID を指定して牌を追加し、赤牌情報も反映する。
   *
   * @param physicalTileId 追加する牌の物理 ID
   */
  public void addPhysicalTile(int physicalTileId) {
    int tileType = Tile.typeOf(physicalTileId);
    int physicalCopyBit = 1 << (physicalTileId & (Tile.TILES_PER_TYPE - 1));
    if ((concealedPhysicalCopyMasks[tileType] & physicalCopyBit) != 0) {
      throw new IllegalStateException("Physical tile already exists in hand: " + physicalTileId);
    }
    boolean isAkaTile = Tile.isAka(physicalTileId);
    if (isAkaTile && AkaTileMask.containsTile(ownedAkaMask, tileType)) {
      throw new IllegalStateException("Aka tile already exists in hand: " + Tile.name(tileType));
    }
    addCount(tileType);
    concealedPhysicalCopyMasks[tileType] |= (byte) physicalCopyBit;
    if (isAkaTile) {
      concealedAkaMask = AkaTileMask.includeTile(concealedAkaMask, tileType);
      ownedAkaMask = AkaTileMask.includeTile(ownedAkaMask, tileType);
    }
  }

  /**
   * 指定牌種を1枚取り除く。通常牌があれば通常牌を優先する。
   *
   * @param tileType 取り除く牌種
   */
  public void remove(int tileType) {
    if (hasNonAkaTile(tileType)) {
      removeNonAka(tileType);
    } else {
      removeAka(tileType);
    }
  }

  /**
   * 物理牌 ID を指定して赤・通常を区別して取り除く。
   *
   * @param physicalTileId 取り除く牌の物理 ID
   */
  public void removePhysicalTile(int physicalTileId) {
    int tileType = Tile.typeOf(physicalTileId);
    int physicalCopyBit = 1 << (physicalTileId & (Tile.TILES_PER_TYPE - 1));
    if ((concealedPhysicalCopyMasks[tileType] & physicalCopyBit) == 0) {
      if (!Tile.isAka(physicalTileId) && hasNonAkaTile(tileType)) {
        // 牌種だけで構築されたテストや復元処理の手牌の匿名通常牌を、指定実牌へ対応付けて消費する。
        removeNonAka(tileType);
        return;
      }
      throw new IllegalStateException("Physical tile is not in hand: " + physicalTileId);
    }
    concealedPhysicalCopyMasks[tileType] &= (byte) ~physicalCopyBit;
    removeCount(tileType);
    if (Tile.isAka(physicalTileId)) {
      concealedAkaMask = AkaTileMask.excludeTile(concealedAkaMask, tileType);
      ownedAkaMask = AkaTileMask.excludeTile(ownedAkaMask, tileType);
    }
  }

  /**
   * 指定牌種の赤牌を取り除く。
   *
   * @param tileType 赤5の牌種
   */
  public void removeAka(int tileType) {
    if (!hasAkaTile(tileType)) {
      throw new IllegalStateException("Cannot remove aka tile not in hand: " + Tile.name(tileType));
    }
    concealedPhysicalCopyMasks[tileType] &= (byte) ~1;
    removeCount(tileType);
    concealedAkaMask = AkaTileMask.excludeTile(concealedAkaMask, tileType);
    ownedAkaMask = AkaTileMask.excludeTile(ownedAkaMask, tileType);
  }

  /**
   * 指定牌種の非赤牌を取り除き、赤牌状態は変えない。
   *
   * @param tileType 取り除く牌種
   */
  public void removeNonAka(int tileType) {
    if (!hasNonAkaTile(tileType)) {
      throw new IllegalStateException(
          "Cannot remove non-aka tile not in hand: " + Tile.name(tileType));
    }
    int nonAkaMask = concealedPhysicalCopyMasks[tileType] & 0b1110;
    if (!Tile.canBeAka(tileType)) {
      nonAkaMask = concealedPhysicalCopyMasks[tileType] & 0x0f;
    }
    if (nonAkaMask != 0) {
      concealedPhysicalCopyMasks[tileType] &= (byte) ~Integer.lowestOneBit(nonAkaMask);
    }
    removeCount(tileType);
  }

  /**
   * 指定牌種の赤牌が手牌（副露・暗槓を除く）にあるかを返す。
   *
   * @param tileType 検査する牌種
   * @return 対応する赤5を持てば {@code true}
   */
  public boolean hasAkaTile(int tileType) {
    return AkaTileMask.containsTile(concealedAkaMask, tileType);
  }

  /**
   * 指定牌種の非赤牌が手牌（副露・暗槓を除く）にあるかを返す。
   *
   * @param tileType 検査する牌種
   * @return 通常牌が一枚以上あれば {@code true}
   */
  public boolean hasNonAkaTile(int tileType) {
    int tileCount = shape.count(tileType);
    return hasAkaTile(tileType) ? tileCount > 1 : tileCount > 0;
  }

  /** 指定牌種の非赤牌を2枚以上持つかを返す。赤5は各色に一枚だけなので真偽値で判定する。 */
  @Override
  public boolean hasMultipleNonAkaTiles(int tileType) {
    int tileCount = shape.count(tileType);
    return hasAkaTile(tileType) ? tileCount > 2 : tileCount > 1;
  }

  /** 指定実牌IDが手牌（副露・暗槓を除く）にあるかを返す。 */
  public boolean containsPhysicalTile(int physicalTileId) {
    int tileType = Tile.typeOf(physicalTileId);
    int physicalCopyBit = 1 << (physicalTileId & (Tile.TILES_PER_TYPE - 1));
    return (concealedPhysicalCopyMasks[tileType] & physicalCopyBit) != 0;
  }

  /** 赤・非赤条件に一致する実牌IDを、追加配列を作らず序数指定で返す。 */
  public int physicalTileId(int tileType, boolean aka, int ordinal) {
    if (ordinal < 0) {
      throw new IndexOutOfBoundsException(ordinal);
    }
    int remaining = concealedPhysicalCopyMasks[tileType] & 0x0f;
    for (int copy = 0; copy < Tile.TILES_PER_TYPE; copy++) {
      int physicalTileId = tileType * Tile.TILES_PER_TYPE + copy;
      if ((remaining & (1 << copy)) != 0 && Tile.isAka(physicalTileId) == aka) {
        if (ordinal-- == 0) {
          return physicalTileId;
        }
      }
    }
    throw new IllegalStateException(
        "Not enough matching physical tiles: " + Tile.name(tileType) + " aka=" + aka);
  }

  /** 指定牌種で最小の実牌IDを返す。 */
  public int firstPhysicalTileId(int tileType) {
    int mask = concealedPhysicalCopyMasks[tileType] & 0x0f;
    if (mask == 0) {
      throw new IllegalStateException("No physical tile in hand: " + Tile.name(tileType));
    }
    return tileType * Tile.TILES_PER_TYPE + Integer.numberOfTrailingZeros(mask);
  }

  /** 手牌（副露・暗槓を除く）を指定実牌ID列へ置き換える。牌譜入力・外部復元境界専用。 */
  public void resetPhysicalTiles(int[] physicalTileIds) {
    clear();
    for (int physicalTileId : physicalTileIds) {
      addPhysicalTile(physicalTileId);
    }
  }

  @Override
  public int concealedAkaMask() {
    return concealedAkaMask;
  }

  /**
   * 手牌（副露・暗槓を除く）または面子に赤牌を持つ牌種のビットマスクを返す。
   *
   * @return ビット 0=萬5、1=筒5、2=索5 のマスク
   */
  @Override
  public int ownedAkaMask() {
    return ownedAkaMask;
  }

  /**
   * 手牌（副露・暗槓を除く）中の指定牌種枚数を返す。
   *
   * @param tileType 対象牌種
   * @return 赤・通常を合わせた枚数
   */
  public int count(int tileType) {
    return shape.count(tileType);
  }

  /** 手牌（副露・暗槓を除く）に存在する牌種マスクを返す。 */
  public long concealedTileTypeMask() {
    return shape.concealedTileTypeMask();
  }

  /** 手牌（副露・暗槓を除く）に存在する牌種数を返す。 */
  public int distinctConcealedTileTypeCount() {
    return shape.distinctConcealedTileTypeCount();
  }

  /** 2枚以上ある手牌（副露・暗槓を除く）の牌種数を返す。 */
  public int concealedPairTileTypeCount() {
    return shape.concealedPairTileTypeCount();
  }

  /** 2枚以上ある手牌（副露・暗槓を除く）の牌種マスクを返す。 */
  public long concealedPairTileTypeMask() {
    return shape.concealedPairTileTypeMask();
  }

  /** 国士対象13牌種のうち存在する種類数を返す。 */
  public int kokushiTileTypeCount() {
    return shape.kokushiTileTypeCount();
  }

  /** 国士対象牌のうち2枚以上ある牌種数を返す。 */
  public int kokushiPairTileTypeCount() {
    return shape.kokushiPairTileTypeCount();
  }

  /** 面子の識別子を成立順に詰めたキャッシュ照合用の値を返す。 */
  public long canonicalMeldSignature() {
    return canonicalMeldSignature;
  }

  /** 確定槓子数を返す。 */
  public int kanCount() {
    return kanCount;
  }

  /**
   * 手牌（副露・暗槓を除く）の枚数配列を複製して返す。
   *
   * @return 長さ34の独立した配列
   */
  public int[] copyConcealedTileCounts() {
    return shape.copyConcealedTileCounts();
  }

  @Override
  public void copyConcealedTileCountsTo(int[] destination) {
    shape.copyConcealedTileCountsTo(destination);
  }

  /**
   * 手牌（副露・暗槓を除く）の総枚数を返す。
   *
   * @return 面子を含まない牌数
   */
  public int concealedTileCount() {
    return shape.concealedTileCount();
  }

  @Override
  public Meld meld(int meldIndex) {
    return melds[java.util.Objects.checkIndex(meldIndex, meldCount())];
  }

  /**
   * 発生記録時点不明として面子を追加する。
   *
   * @param meld 追加する面子
   */
  public void addMeld(Meld meld) {
    addMeld(meld, -1);
  }

  /**
   * 副露をタイミング付きで追加する。
   *
   * @param meld 追加する面子
   * @param afterRiverIndex 鳴いた時点の本人の河サイズ (= 鳴き直後打牌の河インデックス)。-1 は不明
   */
  public void addMeld(Meld meld, int afterRiverIndex) {
    addMeld(meld, afterRiverIndex, -1);
  }

  /** 局内の共通公開イベント連番を付ける。加槓面子を直接追加する場合、元のポンの成立時点は不明のまま保持する。 */
  void addMeld(Meld meld, int afterRiverIndex, int creationEvent) {
    requireAfterRiverIndex(afterRiverIndex);
    java.util.Objects.requireNonNull(meld, "meld");
    if (meldCount() >= melds.length) {
      throw new IllegalStateException("a hand cannot contain more than four melds");
    }
    if (meld.containsAkaTile()) {
      requireAkaTileNotOwned(meld.akaTileType());
    }
    int meldIndex = meldCount();
    shape.setMeldCount(meldIndex + 1);
    melds[meldIndex] = meld;
    meldCreationEvents[meldIndex] = meld.type() == Meld.Type.KAKAN ? -1 : creationEvent;
    meldKakanEvents[meldIndex] = meld.type() == Meld.Type.KAKAN ? creationEvent : -1;
    if (meld.type() == Meld.Type.CHI || meld.type() == Meld.Type.PON) {
      meldCallAfterRiverIndexes[meldIndex] = afterRiverIndex;
      meldKanAfterRiverIndexes[meldIndex] = -1;
    } else {
      meldCallAfterRiverIndexes[meldIndex] = -1;
      meldKanAfterRiverIndexes[meldIndex] = afterRiverIndex;
    }
    canonicalMeldSignature |= (long) meld.canonicalId() << (meldIndex * MELD_ID_BITS);
    if (!meld.preservesMenzen()) {
      openMeldCount++;
    }
    if (meld.isKan()) {
      kanCount++;
    }
    if (meld.containsAkaTile()) {
      int akaTileType = meld.akaTileType();
      ownedAkaMask = AkaTileMask.includeTile(ownedAkaMask, akaTileType);
    }
  }

  /**
   * チー・ポンの成立タイミングを本人の河インデックスで返す。
   *
   * @param meldIndex 面子インデックス
   * @return 鳴き直後打牌の河インデックス。不明または対象外なら -1
   */
  public int meldCallAfterRiverIndex(int meldIndex) {
    return meldCallAfterRiverIndexes[java.util.Objects.checkIndex(meldIndex, meldCount())];
  }

  /**
   * 大明槓・暗槓・加槓の成立タイミングを本人の河インデックスで返す。
   *
   * @param meldIndex 面子インデックス
   * @return 槓成立時の河要素数。不明または対象外なら -1
   */
  public int meldKanAfterRiverIndex(int meldIndex) {
    return meldKanAfterRiverIndexes[java.util.Objects.checkIndex(meldIndex, meldCount())];
  }

  /** 面子の最初の公開記録イベント。加槓でも元のポンのイベント連番を保つ。不明なら-1。 */
  public int meldCreationEvent(int meldIndex) {
    return meldCreationEvents[java.util.Objects.checkIndex(meldIndex, meldCount())];
  }

  /** 槍槓応答前の加槓公開イベント。加槓で-1ならイベント連番は不明、他の面子には後続イベントがない。 */
  public int meldKakanEvent(int meldIndex) {
    return meldKakanEvents[java.util.Objects.checkIndex(meldIndex, meldCount())];
  }

  /**
   * 暗槓以外の副露がなく、門前が保たれているかを返す。
   *
   * @return 門前なら {@code true}
   */
  public boolean isMenzen() {
    return openMeldCount == 0;
  }

  /**
   * 手牌が保持する確定面子数を返す。
   *
   * @return 0～4 の面子数
   */
  public int meldCount() {
    return shape.meldCount();
  }

  /**
   * 指定牌種の加槓元となるポンを返す。
   *
   * @param tileType 探す刻子の牌種
   * @return 同一インスタンスの PON 面子
   * @throws IllegalStateException 対応するポンが存在しない場合
   */
  public Meld requirePon(int tileType) {
    for (int meldIndex = 0; meldIndex < meldCount(); meldIndex++) {
      Meld meld = melds[meldIndex];
      if (meld.type() == Meld.Type.PON && meld.baseTileType() == tileType) {
        return meld;
      }
    }
    throw new IllegalStateException("PON meld not found for tileType: " + Tile.name(tileType));
  }

  /**
   * ポンを加槓に置換する。ポンの成立時点を引き継ぎ、加槓の宣言時点を記録する。
   *
   * @param ponMeld 置換対象の既存 PON インスタンス
   * @param kakanMeld 対象ポンから作った KAKAN
   * @param kanAfterRiverIndex 加槓時点の本人の河サイズ。-1 は不明
   */
  public void replacePonWithKakan(Meld ponMeld, Meld kakanMeld, int kanAfterRiverIndex) {
    replacePonWithKakan(ponMeld, kakanMeld, kanAfterRiverIndex, -1);
  }

  /** ポン成立時のイベント連番を保ち、加槓宣言のイベント連番を追加する。 */
  void replacePonWithKakan(Meld ponMeld, Meld kakanMeld, int kanAfterRiverIndex, int kakanEvent) {
    requireAfterRiverIndex(kanAfterRiverIndex);
    if (ponMeld.type() != Meld.Type.PON) {
      throw new IllegalArgumentException("replace source must be PON");
    }
    if (!kakanMeld.isKakanUpgradeOf(ponMeld)) {
      throw new IllegalArgumentException("replacement KAKAN must be derived from source PON");
    }

    int meldIndex = indexOfMeldIdentity(ponMeld);
    if (meldIndex < 0) {
      throw new IllegalArgumentException("PON meld not found");
    }
    if (kakanMeld.addedTileIsAka()) {
      requireAkaTileNotOwned(kakanMeld.baseTileType());
    }
    melds[meldIndex] = kakanMeld;
    meldKanAfterRiverIndexes[meldIndex] = kanAfterRiverIndex;
    meldKakanEvents[meldIndex] = kakanEvent;
    long meldMask = ((1L << MELD_ID_BITS) - 1L) << (meldIndex * MELD_ID_BITS);
    canonicalMeldSignature =
        (canonicalMeldSignature & ~meldMask)
            | (long) kakanMeld.canonicalId() << (meldIndex * MELD_ID_BITS);
    kanCount++;
    if (kakanMeld.addedTileIsAka()) {
      ownedAkaMask = AkaTileMask.includeTile(ownedAkaMask, kakanMeld.baseTileType());
    }
  }

  /** 手牌（副露・暗槓を除く）、赤牌マスク、面子をすべて空に戻す。 */
  public void clear() {
    Arrays.fill(concealedPhysicalCopyMasks, (byte) 0);
    Arrays.fill(melds, 0, meldCount(), null);
    shape.clear();
    Arrays.fill(meldCallAfterRiverIndexes, -1);
    Arrays.fill(meldKanAfterRiverIndexes, -1);
    Arrays.fill(meldCreationEvents, -1);
    Arrays.fill(meldKakanEvents, -1);
    concealedAkaMask = AkaTileMask.EMPTY;
    openMeldCount = 0;
    kanCount = 0;
    ownedAkaMask = AkaTileMask.EMPTY;
    canonicalMeldSignature = 0L;
  }

  @Override
  public String toString() {
    StringBuilder description = new StringBuilder();
    // 手牌のうち副露・暗槓を除いた牌を表示
    for (int tileType = 0; tileType < Tile.NUM_TILE_TYPES; tileType++) {
      description.append(
          String.valueOf(Tile.name(tileType)).repeat(Math.max(0, shape.count(tileType))));
    }
    // 副露を表示
    if (meldCount() != 0) {
      description.append(" | ");
      for (int meldIndex = 0; meldIndex < meldCount(); meldIndex++) {
        if (meldIndex > 0) {
          description.append(' ');
        }
        description.append(melds[meldIndex]);
      }
    }
    return description.toString();
  }

  private void addCount(int tileType) {
    shape.add(tileType);
  }

  private void removeCount(int tileType) {
    shape.remove(tileType);
  }

  private void requireAkaTileNotOwned(int tileType) {
    if (AkaTileMask.containsTile(ownedAkaMask, tileType)) {
      throw new IllegalStateException("Aka tile already exists in hand: " + Tile.name(tileType));
    }
  }

  private int indexOfMeldIdentity(Meld meld) {
    for (int meldIndex = 0; meldIndex < meldCount(); meldIndex++) {
      if (melds[meldIndex] == meld) {
        return meldIndex;
      }
    }
    return -1;
  }

  private static void requireAfterRiverIndex(int afterRiverIndex) {
    if (afterRiverIndex < -1) {
      throw new IllegalArgumentException("afterRiverIndex must be -1 or greater");
    }
  }
}

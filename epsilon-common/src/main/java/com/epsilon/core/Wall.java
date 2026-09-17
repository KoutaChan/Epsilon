package com.epsilon.core;

import com.epsilon.util.SeedMixer;
import java.util.Arrays;
import java.util.Random;

/**
 * 一局分の136枚の実牌IDと、通常ツモ・嶺上ツモの進行位置を管理する。
 *
 * <p>同じ乱数シードと局の生成番号から同じ牌山を再現できる。ドラの表示枚数は{@link
 * DoraState}が管理し、このクラスは王牌の所定位置にあるドラ・裏ドラ表示牌を返す。局面を保存または分岐させる場合は、コピーコンストラクターで独立した牌山を作る。
 */
final class Wall {

  private static final int TOTAL_TILES = 136;
  static final int LIVE_WALL_SIZE = 122;
  private static final int INITIAL_DORA_INDICATOR_WALL_INDEX = TOTAL_TILES - 6;
  private static final int INITIAL_URA_DORA_INDICATOR_WALL_INDEX = TOTAL_TILES - 5;
  private static final long ROUND_SHUFFLE_SALT = 0xC6BC279692B5CC83L;

  private final int[] physicalTileIds;
  private final Random random;
  private final long shuffleSeed;
  private int shuffleEpoch;
  private int drawIndex;
  private int deadWallIndex;
  private int kanDrawCount;

  /** 非決定的シードで最初の局の牌山を構築する。 */
  Wall() {
    this(new Random().nextLong(), 0, true);
  }

  /**
   * 指定シードで最初の局の牌山を構築する。
   *
   * @param shuffleSeed 局ごとのシャッフルシードを導出する基準値
   */
  Wall(long shuffleSeed) {
    this(shuffleSeed, 0, true);
  }

  private Wall(long shuffleSeed, int initialShuffleEpoch, boolean shuffleImmediately) {
    this.physicalTileIds = new int[TOTAL_TILES];
    this.random = new Random(0L);
    this.shuffleSeed = shuffleSeed;
    this.shuffleEpoch = initialShuffleEpoch;
    resetOrdered();
    if (shuffleImmediately) {
      shuffleNextRound();
    }
  }

  static Wall forGameState() {
    return forGameState(new Random().nextLong());
  }

  static Wall forGameState(long shuffleSeed) {
    // startRound() まで牌は使わない。実牌IDを一意に保ったまま初回局の生成番号を予約する。
    return new Wall(shuffleSeed, 1, false);
  }

  /**
   * 現在位置を含む独立コピーを作る。
   *
   * @param other 複製元
   */
  Wall(Wall other) {
    this.physicalTileIds = other.physicalTileIds.clone();
    this.random = new Random(0L);
    this.shuffleSeed = other.shuffleSeed;
    this.shuffleEpoch = other.shuffleEpoch;
    this.drawIndex = other.drawIndex;
    this.deadWallIndex = other.deadWallIndex;
    this.kanDrawCount = other.kanDrawCount;
  }

  /** 実牌を初期順へ戻し、次局の生成番号を使って再シャッフルする。 */
  void init() {
    resetOrdered();
    shuffleNextRound();
  }

  private void resetOrdered() {
    for (int physicalTileId = 0; physicalTileId < TOTAL_TILES; physicalTileId++) {
      physicalTileIds[physicalTileId] = physicalTileId;
    }
    drawIndex = 0;
    deadWallIndex = TOTAL_TILES - 1;
    kanDrawCount = 0;
  }

  private void shuffleNextRound() {
    shuffle(SeedMixer.indexed(shuffleSeed, ROUND_SHUFFLE_SALT, shuffleEpoch));
    shuffleEpoch++;
  }

  private void shuffle(long seed) {
    random.setSeed(seed);
    for (int destinationIndex = TOTAL_TILES - 1; destinationIndex > 0; destinationIndex--) {
      int sourceIndex = random.nextInt(destinationIndex + 1);
      int temporaryPhysicalTileId = physicalTileIds[destinationIndex];
      physicalTileIds[destinationIndex] = physicalTileIds[sourceIndex];
      physicalTileIds[sourceIndex] = temporaryPhysicalTileId;
    }
  }

  /**
   * 王牌を除く牌山の次の物理牌を自摸る。
   *
   * @return 次の物理牌 ID
   * @throws IllegalStateException 槓による王牌を除く牌山短縮を含めて残り牌がない場合
   */
  int draw() {
    if (drawIndex >= LIVE_WALL_SIZE - kanDrawCount) {
      throw new IllegalStateException("Wall is exhausted");
    }
    return physicalTileIds[drawIndex++];
  }

  /**
   * 王牌の次の嶺上牌を自摸り、以後の王牌を除く牌山を一枚短縮する。
   *
   * @return 次の嶺上物理牌 ID
   * @throws IllegalStateException 嶺上牌を使い切っている場合
   */
  int drawFromDeadWall() {
    if (deadWallIndex < LIVE_WALL_SIZE) {
      throw new IllegalStateException("Dead wall exhausted");
    }
    kanDrawCount++;
    return physicalTileIds[deadWallIndex--];
  }

  int doraIndicatorPhysicalTileIdAt(int slot) {
    requireIndicatorSlot(slot);
    return physicalTileIds[doraIndicatorWallIndex(slot)];
  }

  int uraDoraIndicatorPhysicalTileIdAt(int slot) {
    requireIndicatorSlot(slot);
    return physicalTileIds[uraDoraIndicatorWallIndex(slot)];
  }

  private static void requireIndicatorSlot(int slot) {
    if (slot < 0 || slot >= DoraState.MAX_INDICATORS) throw new IndexOutOfBoundsException(slot);
  }

  /**
   * 王牌を除く牌山から今後自摸可能な枚数を返す。
   *
   * @return 槓による短縮を反映した残り枚数
   */
  int remaining() {
    return LIVE_WALL_SIZE - kanDrawCount - drawIndex;
  }

  /**
   * 王牌を除く牌山を使い切ったかを返す。
   *
   * @return 通常自摸できる牌がなければ {@code true}
   */
  boolean isExhausted() {
    return drawIndex >= LIVE_WALL_SIZE - kanDrawCount;
  }

  /** 牌譜の再構成で、牌が不明であることを示す値。 */
  public static final int UNKNOWN_PHYSICAL_TILE_ID = -1;

  /**
   * TenhouWallDecoder が復元した完全牌山（136枚）で初期化する。
   *
   * <p>配列レイアウト:
   *
   * <ul>
   *   <li>[0..51] = 配牌
   *   <li>[52..121] = 王牌を除く牌山（ツモ順）
   *   <li>[122..131] = 槓ドラまでの表示牌・裏表示牌。初期表示牌から降順に {@code dora=130,128,...,122}, {@code
   *       ura=131,129,...,123}
   *   <li>[132..135] = 嶺上牌。{@code 135,134,133,132} の順にツモる
   * </ul>
   *
   * @param decodedPhysicalTileIds 136 要素の実牌ID配列
   */
  void initWithFullWall(int[] decodedPhysicalTileIds) {
    if (decodedPhysicalTileIds.length != TOTAL_TILES) {
      throw new IllegalArgumentException(
          "decodedPhysicalTileIds must have " + TOTAL_TILES + " elements");
    }
    System.arraycopy(decodedPhysicalTileIds, 0, physicalTileIds, 0, TOTAL_TILES);
    this.drawIndex = 52;
    this.deadWallIndex = TOTAL_TILES - 1;
    this.kanDrawCount = 0;
  }

  /**
   * 非公開の物理牌を不明として扱う、牌譜再構成用の牌山へ初期化する。
   *
   * <p>完全牌山を必要とせず、消費済み王牌を除く牌山枚数だけを復元する。既知の表示牌は {@link #setReconstructedDoraIndicator(int, int,
   * boolean)}で固定枠へ設定する。
   *
   * @param drawIndex 配牌を含め、牌山先頭から消費済みの物理牌数
   */
  void initForReconstruction(int drawIndex) {
    Arrays.fill(physicalTileIds, UNKNOWN_PHYSICAL_TILE_ID);
    this.drawIndex = drawIndex;
    this.deadWallIndex = TOTAL_TILES - 1;
    this.kanDrawCount = 0;
  }

  /**
   * 牌そのものを参照せず、再構成用の王牌を除く牌山カーソルを一枚進める。
   *
   * <p>牌譜から他家の非公開自摸を反映するときに使用する。既に王牌を除く牌山が尽きていれば何もしない。
   */
  void consumeDraw() {
    if (drawIndex < LIVE_WALL_SIZE - kanDrawCount) {
      drawIndex++;
    }
  }

  /**
   * 牌そのものを参照せず、再構成用の嶺上カーソルを一枚進める。
   *
   * <p>嶺上牌が残っている場合だけ槓自摸数とカーソルを更新する。
   */
  void consumeDeadWallDraw() {
    if (deadWallIndex >= LIVE_WALL_SIZE) {
      kanDrawCount++;
      deadWallIndex--;
    }
  }

  void setReconstructedDoraIndicator(int slot, int tileType, boolean aka) {
    requireIndicatorSlot(slot);
    int indicatorWallIndex = doraIndicatorWallIndex(slot);
    if (physicalTileIds[indicatorWallIndex] == UNKNOWN_PHYSICAL_TILE_ID) {
      physicalTileIds[indicatorWallIndex] = tileType * Tile.TILES_PER_TYPE + (aka ? 0 : 1);
    }
  }

  private static int doraIndicatorWallIndex(int indicatorIndex) {
    return INITIAL_DORA_INDICATOR_WALL_INDEX - indicatorIndex * 2;
  }

  private static int uraDoraIndicatorWallIndex(int indicatorIndex) {
    return INITIAL_URA_DORA_INDICATOR_WALL_INDEX - indicatorIndex * 2;
  }
}

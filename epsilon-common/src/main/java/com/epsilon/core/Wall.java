package com.epsilon.core;

import com.epsilon.util.SeedMixer;
import java.util.Arrays;
import java.util.Random;
import java.util.SplittableRandom;

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
  private static final int RINSHAN_START = TOTAL_TILES - 4;

  private final int[] physicalTileIds;
  private final Random random;
  private final long shuffleSeed;
  private int shuffleEpoch;
  private int drawIndex;
  private int deadWallIndex;
  private int kanDrawCount;

  // 席別ツモ列（WASO）。null なら位置順。列は物理牌 ID の並びで、配牌直後に位置 52+k の牌を席 (親+k) mod 4 に割り当てる。
  // ツモのときは自分の列の次の牌を現在の読み位置へ入れ替えてから読む。割り当ては位置だけで決まり牌の中身を見ないので、ツモの列は一様な並びのまま。
  private int[][] seatStreams;
  private int[] streamLength;
  private int[] streamCursor;
  private int[] positionOf;

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
    if (other.seatStreams != null) {
      this.seatStreams = new int[other.seatStreams.length][];
      for (int seat = 0; seat < seatStreams.length; seat++)
        this.seatStreams[seat] = other.seatStreams[seat].clone();
      this.streamLength = other.streamLength.clone();
      this.streamCursor = other.streamCursor.clone();
      this.positionOf = other.positionOf.clone();
    }
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
    seatStreams = null;
  }

  /**
   * 配牌の直後に席別ツモ列を作る。以後 {@link #draw(int)} は席ごとの列から供給する。
   *
   * @param dealer 親の席
   */
  void enableSeatStreams(int dealer) {
    if (drawIndex != 52) throw new IllegalStateException("Seat streams start right after the deal");
    seatStreams = new int[4][(LIVE_WALL_SIZE - drawIndex + 3) / 4];
    streamLength = new int[4];
    streamCursor = new int[4];
    positionOf = new int[TOTAL_TILES];
    for (int position = drawIndex; position < LIVE_WALL_SIZE; position++) {
      int seat = (dealer + position - drawIndex) % 4;
      seatStreams[seat][streamLength[seat]++] = physicalTileIds[position];
    }
    for (int position = 0; position < TOTAL_TILES; position++)
      positionOf[physicalTileIds[position]] = position;
  }

  /**
   * 指定席のツモ。席別ツモ列が有効なら自分の列の次の牌を読み位置へ入れ替えてから読み、列が尽きたら位置順に戻る。
   *
   * @param seat ツモる席
   * @return 物理牌 ID
   */
  int draw(int seat) {
    if (seatStreams == null) return draw();
    if (drawIndex >= LIVE_WALL_SIZE - kanDrawCount) {
      throw new IllegalStateException("Wall is exhausted");
    }
    int[] stream = seatStreams[seat];
    while (streamCursor[seat] < streamLength[seat]
        && positionOf[stream[streamCursor[seat]]] < drawIndex) streamCursor[seat]++;
    if (streamCursor[seat] < streamLength[seat]) {
      int tile = stream[streamCursor[seat]];
      int from = positionOf[tile];
      if (from != drawIndex) {
        int displaced = physicalTileIds[drawIndex];
        physicalTileIds[drawIndex] = tile;
        physicalTileIds[from] = displaced;
        positionOf[tile] = drawIndex;
        positionOf[displaced] = from;
      }
    }
    return physicalTileIds[drawIndex++];
  }

  boolean seatStreamsEnabled() {
    return seatStreams != null;
  }

  /**
   * 全席から見えない牌を一様に並べ替える。対象は王牌を除く牌山の残り（槓で届かなくなった位置を含む）、未使用の嶺上牌、未公開のドラ表示牌の枠、
   * すべての裏ドラ表示牌の枠。見えている牌（公開済みの表示牌を含む）は動かさない。公開済みの表示牌の裏ドラは {@link DoraState} が覚えている
   * ので、呼び出し側が写し直す（{@link GameState#reshuffleUnseen(long)}）。
   *
   * <p>置換は牌の位置ではなく牌そのものに掛け、席別ツモ列の未ツモの要素にも同じ置換を適用する。どの位置の牌を誰が引くかの規則は変わらないので、
   * 並べ替えた後の牌山も公開情報のもとで一様に従う。
   *
   * @param seed 並べ替えの乱数
   * @param revealedIndicators 公開済みのドラ表示牌の数
   */
  void reshuffleUnseen(long seed, int revealedIndicators) {
    int[] positions = new int[TOTAL_TILES];
    int count = 0;
    for (int position = drawIndex; position < LIVE_WALL_SIZE; position++)
      positions[count++] = position;
    for (int position = RINSHAN_START; position <= deadWallIndex; position++)
      positions[count++] = position;
    for (int slot = 0; slot < DoraState.MAX_INDICATORS; slot++) {
      if (slot >= revealedIndicators) positions[count++] = doraIndicatorWallIndex(slot);
      positions[count++] = uraDoraIndicatorWallIndex(slot);
    }
    int[] tiles = new int[count];
    for (int i = 0; i < count; i++) {
      tiles[i] = physicalTileIds[positions[i]];
      if (tiles[i] == UNKNOWN_PHYSICAL_TILE_ID)
        throw new IllegalStateException("Cannot reshuffle a reconstructed wall");
    }
    int[] replacement = new int[TOTAL_TILES];
    for (int tile = 0; tile < TOTAL_TILES; tile++) replacement[tile] = tile;
    var random = new SplittableRandom(seed);
    int[] shuffled = tiles.clone();
    for (int i = count - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int swap = shuffled[i];
      shuffled[i] = shuffled[j];
      shuffled[j] = swap;
    }
    for (int i = 0; i < count; i++) {
      replacement[tiles[i]] = shuffled[i];
      physicalTileIds[positions[i]] = shuffled[i];
    }
    if (seatStreams == null) return;
    for (int seat = 0; seat < seatStreams.length; seat++)
      for (int k = 0; k < streamLength[seat]; k++)
        seatStreams[seat][k] = replacement[seatStreams[seat][k]];
    for (int position = 0; position < TOTAL_TILES; position++)
      positionOf[physicalTileIds[position]] = position;
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

  /** 牌山の位置の物理牌 ID（試験用）。 */
  int physicalTileIdAt(int position) {
    return physicalTileIds[position];
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
    this.seatStreams = null;
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
    this.seatStreams = null;
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

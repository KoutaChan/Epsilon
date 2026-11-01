package com.epsilon.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * 1 席の河（捨て牌）。打牌の中身は局の {@link PublicEventLog} にあり、ここは河の順の出来事の番号と、牌の種類ごとの枚数・立直の位置の
 * 集約だけを持つ（記録は {@code log}、河の順の番号は {@code events}）。河の位置 i の欄は記録の出来事 {@link #event(int) event(i)} の欄と同じ値。
 *
 * <p>容量は {@link RoundLimits#MAX_SEAT_DISCARDS}（鳴きで他家の手番が飛ぶと 24 枚を超えうるので、巡目からは決めない）。持ち主は記録と同じ
 * {@link GameState} で、分岐は番号の配列を写すだけ（打牌ごとの物は作らない）。
 */
public final class River {

  private final PublicEventLog log;
  private final byte[] events = new byte[RoundLimits.MAX_SEAT_DISCARDS];
  private final int[] discardCountsByTileType;
  private int size;
  private long discardedTileTypeMask;
  private int riichiDeclarationIndex;
  private int riichiDeclarationSequence;
  private int postRiichiDiscardCount;
  private int postRiichiTsumogiriCount;

  /**
   * 空の河を作る。
   *
   * @param log 打牌を書く局の記録
   */
  River(PublicEventLog log) {
    this.log = log;
    this.discardCountsByTileType = new int[Tile.NUM_TILE_TYPES];
    this.riichiDeclarationIndex = -1;
    this.riichiDeclarationSequence = -1;
  }

  /**
   * 分岐の河を作る。{@code log} は {@code other} の記録を写した記録。
   *
   * @param other 写す河
   * @param log 写した局面の記録
   */
  River(River other, PublicEventLog log) {
    this.log = log;
    System.arraycopy(other.events, 0, events, 0, other.size);
    this.size = other.size;
    this.discardCountsByTileType = other.discardCountsByTileType.clone();
    this.discardedTileTypeMask = other.discardedTileTypeMask;
    this.riichiDeclarationIndex = other.riichiDeclarationIndex;
    this.riichiDeclarationSequence = other.riichiDeclarationSequence;
    this.postRiichiDiscardCount = other.postRiichiDiscardCount;
    this.postRiichiTsumogiriCount = other.postRiichiTsumogiriCount;
  }

  /**
   * 記録に足した打牌を河の最後に置き、集約を進める。
   *
   * @param event 記録の打牌の番号
   */
  void append(int event) {
    if (size == events.length) {
      throw new IllegalStateException("a seat discards at most " + events.length + " tiles");
    }
    events[size++] = (byte) event;
    int tileType = log.tile(event);
    discardCountsByTileType[tileType]++;
    discardedTileTypeMask |= 1L << tileType;
    if (riichiDeclarationIndex < 0 && log.riichi(event)) {
      riichiDeclarationIndex = size - 1;
      riichiDeclarationSequence = log.sequence(event);
    } else if (riichiDeclarationIndex >= 0) {
      postRiichiDiscardCount++;
      if (log.tsumogiri(event)) {
        postRiichiTsumogiriCount++;
      }
    }
  }

  /** 河の位置 {@code index} の打牌の、局の公開の出来事の番号。 */
  public int event(int index) {
    return events[Objects.checkIndex(index, size)];
  }

  /** 捨てた牌の34牌種IDを返す。 */
  public int tileType(int index) {
    return log.tile(event(index));
  }

  /** 捨てた席における0始まりの打牌巡目を返す。 */
  public int turnNumber(int index) {
    return log.turn(event(index));
  }

  /** リーチ宣言牌なら {@code true} を返す。 */
  public boolean riichiDeclaration(int index) {
    return log.riichi(event(index));
  }

  /** 捨てた物理牌が赤ドラなら {@code true} を返す。 */
  public boolean aka(int index) {
    return log.aka(event(index));
  }

  /** 直前の自摸牌を捨てた場合は {@code true} を返す。 */
  public boolean tsumogiri(int index) {
    return log.tsumogiri(event(index));
  }

  /** 局内で全席を通した0始まりの打牌連番。 */
  public int sequence(int index) {
    return log.sequence(event(index));
  }

  /** チー、ポンまたは大明槓で鳴かれていれば {@code true} を返す。 */
  public boolean called(int index) {
    return log.calledBy(event(index)) != PublicEventLog.NOT_CALLED;
  }

  /** 指定牌種を捨てた回数を返す。 */
  public int discardCount(int tileType) {
    return discardCountsByTileType[tileType];
  }

  /** 河に現れた牌種を34-ビットマスクで返す。 */
  public long discardedTileTypeMask() {
    return discardedTileTypeMask;
  }

  /** リーチ宣言牌の河インデックス。不在なら-1。 */
  public int riichiDeclarationIndex() {
    return riichiDeclarationIndex;
  }

  /** リーチ宣言牌の全席通し連番。不在なら-1。 */
  public int riichiDeclarationSequence() {
    return riichiDeclarationSequence;
  }

  /** リーチ宣言牌より後の打牌数を返す。 */
  public int postRiichiDiscardCount() {
    return postRiichiDiscardCount;
  }

  /** リーチ宣言後の自摸切り数を返す。 */
  public int postRiichiTsumogiriCount() {
    return postRiichiTsumogiriCount;
  }

  /**
   * 特定の牌種が河に含まれるかを返す。
   *
   * @param tileType 検索する34牌種 ID
   * @return 一度でも捨てていれば {@code true}
   */
  public boolean contains(int tileType) {
    return (discardedTileTypeMask & 1L << tileType) != 0L;
  }

  public int size() {
    return size;
  }

  /** 局の初めの空の河に戻す（記録は {@link GameState} が同時に空にする）。 */
  void clear() {
    size = 0;
    Arrays.fill(discardCountsByTileType, 0);
    discardedTileTypeMask = 0L;
    riichiDeclarationIndex = -1;
    riichiDeclarationSequence = -1;
    postRiichiDiscardCount = 0;
    postRiichiTsumogiriCount = 0;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < size; i++) {
      if (i > 0) {
        sb.append(' ');
      }
      if (riichiDeclaration(i)) {
        sb.append('*');
      }
      sb.append(Tile.name(tileType(i)));
      if (called(i)) {
        sb.append('!');
      }
    }
    return sb.toString();
  }
}

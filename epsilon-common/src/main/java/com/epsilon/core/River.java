package com.epsilon.core;

import java.util.Arrays;

/** 河（捨て牌）を管理するクラス。 */
public final class River {

  private static final int MAX_DISCARDS = 24;

  /** 河の1枚の情報。 */
  public static final class Discard {
    private int tileType;
    private int turnNumber;
    private boolean riichiDeclaration;
    private boolean aka;
    private boolean tsumogiri;
    private int sequence;
    private int creationEvent;

    private boolean calledFlag; // 鳴かれたかフラグ(変更可能)

    /**
     * 河へ記録する一枚の完全な打牌情報を生成する。
     *
     * @param tileType 捨てた牌の34牌種 ID
     * @param turnNumber 捨てた席における0始まりの打牌巡目
     * @param riichiDeclaration リーチ宣言牌か
     * @param aka 捨てた物理牌が赤ドラか
     * @param tsumogiri 自摸切りか
     * @param sequence 局内で全席を通した打牌連番。不明なら {@code -1}
     */
    private Discard(
        int tileType,
        int turnNumber,
        boolean riichiDeclaration,
        boolean aka,
        boolean tsumogiri,
        int sequence) {
      set(tileType, turnNumber, riichiDeclaration, aka, tsumogiri, sequence);
    }

    private void set(
        int tileType,
        int turnNumber,
        boolean riichiDeclaration,
        boolean aka,
        boolean tsumogiri,
        int sequence) {
      this.tileType = tileType;
      this.turnNumber = turnNumber;
      this.riichiDeclaration = riichiDeclaration;
      this.aka = aka;
      this.tsumogiri = tsumogiri;
      this.sequence = sequence;
      creationEvent = -1;
      calledFlag = false;
    }

    private void copyFrom(Discard other) {
      set(
          other.tileType,
          other.turnNumber,
          other.riichiDeclaration,
          other.aka,
          other.tsumogiri,
          other.sequence);
      this.calledFlag = other.calledFlag;
      this.creationEvent = other.creationEvent;
    }

    /** 捨てた牌の34牌種IDを返す。 */
    public int tileType() {
      return tileType;
    }

    /** 捨てた席における0始まりの打牌巡目を返す。 */
    public int turnNumber() {
      return turnNumber;
    }

    /** リーチ宣言牌なら {@code true} を返す。 */
    public boolean riichiDeclaration() {
      return riichiDeclaration;
    }

    /** 捨てた物理牌が赤ドラなら {@code true} を返す。 */
    public boolean aka() {
      return aka;
    }

    /** 直前の自摸牌を捨てた場合は {@code true} を返す。 */
    public boolean tsumogiri() {
      return tsumogiri;
    }

    /** 局内で全席を通した0始まりの打牌連番。不明なら {@code -1}。 */
    public int sequence() {
      return sequence;
    }

    /** 打牌・面子・槓の公開記録で共有する局内イベント連番。不明なら-1。 */
    public int creationEvent() {
      return creationEvent;
    }

    /**
     * この捨て牌が他家の副露に使われたかを返す。
     *
     * @return チー、ポンまたは大明槓で鳴かれていれば {@code true}
     */
    public boolean called() {
      return calledFlag;
    }
  }

  private final Discard[] discards;
  private final int[] discardCountsByTileType;
  private int size;
  private long discardedTileTypeMask;
  private int riichiDeclarationIndex;
  private int riichiDeclarationSequence;
  private int postRiichiDiscardCount;
  private int postRiichiTsumogiriCount;

  /** 空の河を生成する。 */
  public River() {
    this.discards = new Discard[MAX_DISCARDS];
    for (int index = 0; index < discards.length; index++) {
      discards[index] = new Discard(-1, -1, false, false, false, -1);
    }
    this.discardCountsByTileType = new int[Tile.NUM_TILE_TYPES];
    this.riichiDeclarationIndex = -1;
    this.riichiDeclarationSequence = -1;
  }

  /**
   * 打牌と鳴かれた状態を複製した河を生成する。
   *
   * @param other 複製元の河
   */
  public River(River other) {
    this.discards = new Discard[MAX_DISCARDS];
    for (int index = 0; index < discards.length; index++) {
      discards[index] = new Discard(-1, -1, false, false, false, -1);
    }
    this.size = other.size;
    for (int index = 0; index < size; index++) {
      this.discards[index].copyFrom(other.discards[index]);
    }
    this.discardCountsByTileType = other.discardCountsByTileType.clone();
    this.discardedTileTypeMask = other.discardedTileTypeMask;
    this.riichiDeclarationIndex = other.riichiDeclarationIndex;
    this.riichiDeclarationSequence = other.riichiDeclarationSequence;
    this.postRiichiDiscardCount = other.postRiichiDiscardCount;
    this.postRiichiTsumogiriCount = other.postRiichiTsumogiriCount;
  }

  /**
   * 通常牌の手出しを河へ追加する。
   *
   * @param tileType 捨てた牌の34牌種 ID
   * @param turnNumber 捨てた席における0始まりの打牌巡目
   * @param riichiDeclaration リーチ宣言牌か
   */
  void append(int tileType, int turnNumber, boolean riichiDeclaration) {
    append(tileType, turnNumber, riichiDeclaration, false, false, -1);
  }

  /**
   * 赤牌情報を含む手出しを河へ追加する。
   *
   * @param tileType 捨てた牌の34牌種 ID
   * @param turnNumber 捨てた席における0始まりの打牌巡目
   * @param riichiDeclaration リーチ宣言牌か
   * @param aka 捨てた物理牌が赤ドラか
   */
  void append(int tileType, int turnNumber, boolean riichiDeclaration, boolean aka) {
    append(tileType, turnNumber, riichiDeclaration, aka, false, -1);
  }

  /**
   * 赤牌・自摸切り情報を含む打牌を河へ追加する。
   *
   * @param tileType 捨てた牌の34牌種 ID
   * @param turnNumber 捨てた席における0始まりの打牌巡目
   * @param riichiDeclaration リーチ宣言牌か
   * @param aka 捨てた物理牌が赤ドラか
   * @param tsumogiri 自摸切りか
   */
  void append(
      int tileType, int turnNumber, boolean riichiDeclaration, boolean aka, boolean tsumogiri) {
    append(tileType, turnNumber, riichiDeclaration, aka, tsumogiri, -1);
  }

  /**
   * 全席通し連番を含む完全な打牌情報を河へ追加する。
   *
   * @param tileType 捨てた牌の34牌種 ID
   * @param turnNumber 捨てた席における0始まりの打牌巡目
   * @param riichiDeclaration リーチ宣言牌か
   * @param aka 捨てた物理牌が赤ドラか
   * @param tsumogiri 自摸切りか
   * @param sequence 局内で全席を通した0始まりの打牌連番。不明なら {@code -1}
   */
  void append(
      int tileType,
      int turnNumber,
      boolean riichiDeclaration,
      boolean aka,
      boolean tsumogiri,
      int sequence) {
    append(tileType, turnNumber, riichiDeclaration, aka, tsumogiri, sequence, -1);
  }

  /** GameStateが管理する公開イベント連番を付けて打牌を記録する。 */
  void append(
      int tileType,
      int turnNumber,
      boolean riichiDeclaration,
      boolean aka,
      boolean tsumogiri,
      int sequence,
      int creationEvent) {
    if (size == discards.length) {
      throw new IllegalStateException("river capacity exceeded: " + discards.length);
    }
    Discard discard = discards[size++];
    discard.set(tileType, turnNumber, riichiDeclaration, aka, tsumogiri, sequence);
    discard.creationEvent = creationEvent;
    discardCountsByTileType[tileType]++;
    discardedTileTypeMask |= 1L << tileType;
    if (riichiDeclarationIndex < 0 && riichiDeclaration) {
      riichiDeclarationIndex = size - 1;
      riichiDeclarationSequence = sequence;
    } else if (riichiDeclarationIndex >= 0) {
      postRiichiDiscardCount++;
      if (tsumogiri) {
        postRiichiTsumogiriCount++;
      }
    }
  }

  /** 最後の捨て牌を「鳴かれた」としてマークする。 */
  void markLastCalled() {
    if (size != 0) {
      discards[size - 1].calledFlag = true;
    }
  }

  /** 指定インデックスの打牌を返す。 */
  public Discard discard(int index) {
    return discards[java.util.Objects.checkIndex(index, size)];
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

  /** リーチ宣言牌の全席通し連番。不明または不在なら-1。 */
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

  /**
   * 河へ記録されている打牌数を返す。
   *
   * @return 打牌数
   */
  public int size() {
    return size;
  }

  /** 河をリセット。 */
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
      Discard discard = discards[i];
      if (i > 0) {
        sb.append(' ');
      }
      if (discard.riichiDeclaration) {
        sb.append('*');
      }
      sb.append(Tile.name(discard.tileType));
      if (discard.calledFlag) {
        sb.append('!');
      }
    }
    return sb.toString();
  }
}

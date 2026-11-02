package com.epsilon.core;

import java.util.Arrays;

/**
 * 河・副露・ドラ表示から決まる公開局面を、局進行と同時に増分更新するインデックス。
 *
 * <p>履歴は{@link PublicEventLog}が保持し、このクラスはAI入力・安全牌判定・残り牌計算で頻繁に読む集約値だけを保持する。
 */
public final class RoundPublicStateIndex {

  private final int[] visibleTileCounts = new int[Tile.NUM_TILE_TYPES];
  private final int[][] riverDiscardCountsBySeat =
      new int[GameState.NUM_PLAYERS][Tile.NUM_TILE_TYPES];
  private final int[] riverDiscardTotalsBySeat = new int[GameState.NUM_PLAYERS];
  private final long[] riverTileTypeMasksBySeat = new long[GameState.NUM_PLAYERS];
  private final long[] riichiGenbutsuTileTypeMasksBySeat = new long[GameState.NUM_PLAYERS];
  private final boolean[] riichiBySeat = new boolean[GameState.NUM_PLAYERS];
  private final int[] riichiDeclarationIndexBySeat = new int[GameState.NUM_PLAYERS];
  private final int[] riichiDeclarationSequenceBySeat = new int[GameState.NUM_PLAYERS];
  private final int[] postRiichiDiscardCountsBySeat = new int[GameState.NUM_PLAYERS];
  private final int[] postRiichiTsumogiriCountsBySeat = new int[GameState.NUM_PLAYERS];
  private int totalDiscardCount;
  private int totalKanCount;
  private final int[] kanCountsBySeat = new int[GameState.NUM_PLAYERS];
  private int visibleDiscardOrMeldAkaMask;

  RoundPublicStateIndex() {
    Arrays.fill(riichiDeclarationIndexBySeat, -1);
    Arrays.fill(riichiDeclarationSequenceBySeat, -1);
  }

  RoundPublicStateIndex(RoundPublicStateIndex other) {
    System.arraycopy(other.visibleTileCounts, 0, visibleTileCounts, 0, Tile.NUM_TILE_TYPES);
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      System.arraycopy(
          other.riverDiscardCountsBySeat[seat],
          0,
          riverDiscardCountsBySeat[seat],
          0,
          Tile.NUM_TILE_TYPES);
    }
    System.arraycopy(
        other.riverTileTypeMasksBySeat, 0, riverTileTypeMasksBySeat, 0, GameState.NUM_PLAYERS);
    System.arraycopy(
        other.riverDiscardTotalsBySeat, 0, riverDiscardTotalsBySeat, 0, GameState.NUM_PLAYERS);
    System.arraycopy(
        other.riichiGenbutsuTileTypeMasksBySeat,
        0,
        riichiGenbutsuTileTypeMasksBySeat,
        0,
        GameState.NUM_PLAYERS);
    System.arraycopy(other.riichiBySeat, 0, riichiBySeat, 0, GameState.NUM_PLAYERS);
    System.arraycopy(
        other.riichiDeclarationIndexBySeat,
        0,
        riichiDeclarationIndexBySeat,
        0,
        GameState.NUM_PLAYERS);
    System.arraycopy(
        other.riichiDeclarationSequenceBySeat,
        0,
        riichiDeclarationSequenceBySeat,
        0,
        GameState.NUM_PLAYERS);
    System.arraycopy(
        other.postRiichiDiscardCountsBySeat,
        0,
        postRiichiDiscardCountsBySeat,
        0,
        GameState.NUM_PLAYERS);
    System.arraycopy(
        other.postRiichiTsumogiriCountsBySeat,
        0,
        postRiichiTsumogiriCountsBySeat,
        0,
        GameState.NUM_PLAYERS);
    System.arraycopy(other.kanCountsBySeat, 0, kanCountsBySeat, 0, GameState.NUM_PLAYERS);
    totalDiscardCount = other.totalDiscardCount;
    totalKanCount = other.totalKanCount;
    visibleDiscardOrMeldAkaMask = other.visibleDiscardOrMeldAkaMask;
  }

  public int visibleTileCount(int tileType) {
    return visibleTileCounts[tileType];
  }

  /** 公開枚数を呼び出し側配列へ複製する。 */
  public void copyVisibleTileCountsTo(int[] destination) {
    System.arraycopy(visibleTileCounts, 0, destination, 0, Tile.NUM_TILE_TYPES);
  }

  /** 公開枚数を独立配列へ複製する。 */
  public int[] copyVisibleTileCounts() {
    return visibleTileCounts.clone();
  }

  /** 河・副露として公開済みの赤5牌種マスク。ドラ表示牌の赤マスクは{@link DoraState}が所有する。 */
  public int visibleDiscardOrMeldAkaMask() {
    return visibleDiscardOrMeldAkaMask;
  }

  public int riverDiscardCount(int seat, int tileType) {
    return riverDiscardCountsBySeat[seat][tileType];
  }

  public long riverTileTypeMask(int seat) {
    return riverTileTypeMasksBySeat[seat];
  }

  /** 指定立直者に対する現物牌種マスクを返す。 */
  public long riichiGenbutsuTileTypeMask(int seat) {
    return riichiGenbutsuTileTypeMasksBySeat[seat];
  }

  public int totalDiscardCount() {
    return totalDiscardCount;
  }

  public int totalKanCount() {
    return totalKanCount;
  }

  public int kanCount(int seat) {
    return kanCountsBySeat[seat];
  }

  public int riichiDeclarationIndex(int seat) {
    return riichiDeclarationIndexBySeat[seat];
  }

  public int discardsAfterRiichi(int seat) {
    return postRiichiDiscardCountsBySeat[seat];
  }

  public int postRiichiTsumogiriCount(int seat) {
    return postRiichiTsumogiriCountsBySeat[seat];
  }

  public int riichiDeclarationSequence(int seat) {
    return riichiDeclarationSequenceBySeat[seat];
  }

  void reset() {
    Arrays.fill(visibleTileCounts, 0);
    for (int[] counts : riverDiscardCountsBySeat) {
      Arrays.fill(counts, 0);
    }
    Arrays.fill(riverTileTypeMasksBySeat, 0L);
    Arrays.fill(riverDiscardTotalsBySeat, 0);
    Arrays.fill(riichiGenbutsuTileTypeMasksBySeat, 0L);
    Arrays.fill(riichiBySeat, false);
    Arrays.fill(riichiDeclarationIndexBySeat, -1);
    Arrays.fill(riichiDeclarationSequenceBySeat, -1);
    Arrays.fill(postRiichiDiscardCountsBySeat, 0);
    Arrays.fill(postRiichiTsumogiriCountsBySeat, 0);
    Arrays.fill(kanCountsBySeat, 0);
    totalDiscardCount = 0;
    totalKanCount = 0;
    visibleDiscardOrMeldAkaMask = AkaTileMask.EMPTY;
  }

  void addDoraIndicatorVisibility(int indicatorTileType) {
    visibleTileCounts[indicatorTileType]++;
  }

  void removeDoraIndicatorVisibility(int indicatorTileType) {
    visibleTileCounts[indicatorTileType]--;
  }

  /** 記録に足した打牌 {@code event} を数える。 */
  void addDiscard(int seat, PublicEventLog log, int event) {
    int tileType = log.tile(event);
    riverDiscardCountsBySeat[seat][tileType]++;
    riverDiscardTotalsBySeat[seat]++;
    riverTileTypeMasksBySeat[seat] |= 1L << tileType;
    visibleTileCounts[tileType]++;
    totalDiscardCount++;
    if (log.aka(event)) {
      visibleDiscardOrMeldAkaMask = AkaTileMask.includeTile(visibleDiscardOrMeldAkaMask, tileType);
    }
    if (riichiDeclarationIndexBySeat[seat] < 0 && log.riichi(event)) {
      riichiDeclarationIndexBySeat[seat] = riverDiscardTotalsBySeat[seat] - 1;
      riichiDeclarationSequenceBySeat[seat] = log.sequence(event);
    } else if (riichiDeclarationIndexBySeat[seat] >= 0) {
      postRiichiDiscardCountsBySeat[seat]++;
      if (log.tsumogiri(event)) {
        postRiichiTsumogiriCountsBySeat[seat]++;
      }
    }
    long tileBit = 1L << tileType;
    for (int riichiSeat = 0; riichiSeat < GameState.NUM_PLAYERS; riichiSeat++) {
      if (riichiBySeat[riichiSeat]) {
        riichiGenbutsuTileTypeMasksBySeat[riichiSeat] |= tileBit;
      }
    }
  }

  /** 鳴かれた打牌 {@code event} を見える枚数から外す（副露の牌として {@link #addMeld} が数え直す）。 */
  void markDiscardCalled(PublicEventLog log, int event) {
    visibleTileCounts[log.tile(event)]--;
  }

  void addMeld(int seat, Meld meld) {
    for (int tileIndex = 0; tileIndex < meld.size(); tileIndex++) {
      visibleTileCounts[meld.tileAt(tileIndex)]++;
    }
    if (meld.containsAkaTile()) {
      visibleDiscardOrMeldAkaMask =
          AkaTileMask.includeTile(visibleDiscardOrMeldAkaMask, meld.akaTileType());
    }
    if (meld.isKan()) {
      kanCountsBySeat[seat]++;
      totalKanCount++;
    }
  }

  void replacePonWithKakan(int seat, Meld kakan) {
    visibleTileCounts[kakan.baseTileType()]++;
    if (kakan.addedTileIsAka()) {
      visibleDiscardOrMeldAkaMask =
          AkaTileMask.includeTile(visibleDiscardOrMeldAkaMask, kakan.baseTileType());
    }
    kanCountsBySeat[seat]++;
    totalKanCount++;
  }

  void setRiichi(int seat, boolean riichi) {
    riichiBySeat[seat] = riichi;
    riichiGenbutsuTileTypeMasksBySeat[seat] = riichi ? riverTileTypeMasksBySeat[seat] : 0L;
  }
}

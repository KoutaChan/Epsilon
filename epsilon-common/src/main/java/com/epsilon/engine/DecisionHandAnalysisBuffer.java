package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.core.Tile;

/** 一つの候補行動を適用した手牌について、呼び出し側バッファ内で再利用する解析結果。 */
public final class DecisionHandAnalysisBuffer {

  private static final int WAIT_SCORE_STRIDE = 6;

  private final byte[] waitTileTypes = new byte[Tile.NUM_TILE_TYPES];
  private final long[] waitYakuMask = new long[Tile.NUM_TILE_TYPES * 2];
  private final int[] waitScores = new int[Tile.NUM_TILE_TYPES * WAIT_SCORE_STRIDE];

  int minimumShanten;
  int standardShanten;
  int chiitoitsuShanten;
  int kokushiShanten;
  boolean specialHandsAvailable;
  int liveImprovingCopies;
  int liveImprovingTileTypes;
  long improvingTileTypeMask;
  long shapeWaitTileTypeMask;
  int waitCount;
  long ronWaitTileTypeMask;
  long tsumoWaitTileTypeMask;
  long akaWinningTileAvailableMask;
  int doraCount;
  int ownedAkaMask;
  boolean menzen;
  int meldCount;
  int concealedTileCount;
  float improvingHitWithinOneDraw;
  float improvingHitWithinTwoDraws;
  float improvingHitWithinThreeDraws;
  int liveShapeWaitCopies;
  int intrinsicRonWaitTileTypes;
  int intrinsicRonWaitCopies;
  int intrinsicTsumoWaitTileTypes;
  int intrinsicTsumoWaitCopies;
  final TenpaiFrontierBuffer frontier = new TenpaiFrontierBuffer();
  int furitenFreeRonFrontierCopies;
  int furitenOnlyRonFrontierCopies;
  int tsumoFrontierCopies;

  void clearTransitionValues() {
    waitCount = 0;
    ronWaitTileTypeMask = 0L;
    tsumoWaitTileTypeMask = 0L;
    akaWinningTileAvailableMask = 0L;
    doraCount = 0;
    ownedAkaMask = 0;
    menzen = false;
    meldCount = 0;
    concealedTileCount = 0;
    improvingHitWithinOneDraw = 0.0f;
    improvingHitWithinTwoDraws = 0.0f;
    improvingHitWithinThreeDraws = 0.0f;
    liveShapeWaitCopies = 0;
    intrinsicRonWaitTileTypes = 0;
    intrinsicRonWaitCopies = 0;
    intrinsicTsumoWaitTileTypes = 0;
    intrinsicTsumoWaitCopies = 0;
    frontier.clear();
    furitenFreeRonFrontierCopies = 0;
    furitenOnlyRonFrontierCopies = 0;
    tsumoFrontierCopies = 0;
  }

  void addWait(
      int tileType, HandScoreBuffer ron, HandScoreBuffer tsumo, boolean akaWinningTileAvailable) {
    int waitIndex = waitCount++;
    waitTileTypes[waitIndex] = (byte) tileType;
    writeWin(waitIndex, 0, ron);
    writeWin(waitIndex, 1, tsumo);
    if (ron != null) {
      ronWaitTileTypeMask |= 1L << tileType;
    }
    if (tsumo != null) {
      tsumoWaitTileTypeMask |= 1L << tileType;
    }
    if (akaWinningTileAvailable) {
      akaWinningTileAvailableMask |= 1L << tileType;
    }
  }

  private void writeWin(int waitIndex, int winType, HandScoreBuffer value) {
    int yakuIndex = waitIndex * 2 + winType;
    int scoreIndex = waitIndex * WAIT_SCORE_STRIDE + winType * 3;
    if (value == null) {
      waitYakuMask[yakuIndex] = 0L;
      waitScores[scoreIndex] = 0;
      waitScores[scoreIndex + 1] = 0;
      waitScores[scoreIndex + 2] = 0;
      return;
    }
    waitYakuMask[yakuIndex] = value.yakuBits();
    waitScores[scoreIndex] = value.han();
    waitScores[scoreIndex + 1] = value.fu();
    waitScores[scoreIndex + 2] = value.basePoints();
  }

  public int minimumShanten() {
    return minimumShanten;
  }

  public int standardShanten() {
    return standardShanten;
  }

  public int chiitoitsuShanten() {
    return chiitoitsuShanten;
  }

  public int kokushiShanten() {
    return kokushiShanten;
  }

  public boolean specialHandsAvailable() {
    return specialHandsAvailable;
  }

  public int liveImprovingCopies() {
    return liveImprovingCopies;
  }

  public int liveImprovingTileTypes() {
    return liveImprovingTileTypes;
  }

  public long improvingTileTypeMask() {
    return improvingTileTypeMask;
  }

  public long shapeWaitTileTypeMask() {
    return shapeWaitTileTypeMask;
  }

  public boolean isDiscardFuriten(long riverTileTypeMask, int discardedTileType) {
    if (Tile.isValidType(discardedTileType)) {
      riverTileTypeMask |= 1L << discardedTileType;
    }
    return (shapeWaitTileTypeMask & riverTileTypeMask) != 0L;
  }

  public int waitCount() {
    return waitCount;
  }

  public int waitTileType(int waitIndex) {
    return waitTileTypes[waitIndex] & 0xff;
  }

  public long ronYakuMask(int waitIndex) {
    return waitYakuMask[waitIndex * 2];
  }

  public long tsumoYakuMask(int waitIndex) {
    return waitYakuMask[waitIndex * 2 + 1];
  }

  public int ronVisibleHan(int waitIndex) {
    return waitScores[waitIndex * WAIT_SCORE_STRIDE];
  }

  public int ronFu(int waitIndex) {
    return waitScores[waitIndex * WAIT_SCORE_STRIDE + 1];
  }

  public int ronBasePoints(int waitIndex) {
    return waitScores[waitIndex * WAIT_SCORE_STRIDE + 2];
  }

  public int tsumoVisibleHan(int waitIndex) {
    return waitScores[waitIndex * WAIT_SCORE_STRIDE + 3];
  }

  public int tsumoFu(int waitIndex) {
    return waitScores[waitIndex * WAIT_SCORE_STRIDE + 4];
  }

  public int tsumoBasePoints(int waitIndex) {
    return waitScores[waitIndex * WAIT_SCORE_STRIDE + 5];
  }

  public long ronWaitTileTypeMask() {
    return ronWaitTileTypeMask;
  }

  public long tsumoWaitTileTypeMask() {
    return tsumoWaitTileTypeMask;
  }

  public long akaWinningTileAvailableMask() {
    return akaWinningTileAvailableMask;
  }

  public int doraCount() {
    return doraCount;
  }

  public int ownedAkaMask() {
    return ownedAkaMask;
  }

  public boolean menzen() {
    return menzen;
  }

  public int meldCount() {
    return meldCount;
  }

  public int concealedTileCount() {
    return concealedTileCount;
  }

  public float improvingHitWithinOneDraw() {
    return improvingHitWithinOneDraw;
  }

  public float improvingHitWithinTwoDraws() {
    return improvingHitWithinTwoDraws;
  }

  public float improvingHitWithinThreeDraws() {
    return improvingHitWithinThreeDraws;
  }

  public int liveShapeWaitCopies() {
    return liveShapeWaitCopies;
  }

  public int intrinsicRonWaitTileTypes() {
    return intrinsicRonWaitTileTypes;
  }

  public int intrinsicRonWaitCopies() {
    return intrinsicRonWaitCopies;
  }

  public int intrinsicTsumoWaitTileTypes() {
    return intrinsicTsumoWaitTileTypes;
  }

  public int intrinsicTsumoWaitCopies() {
    return intrinsicTsumoWaitCopies;
  }

  public long furitenFreeRonFrontierTileTypeMask() {
    return frontier.furitenFreeRon();
  }

  public long furitenOnlyRonFrontierTileTypeMask() {
    return frontier.furitenOnlyRon();
  }

  public long tsumoFrontierTileTypeMask() {
    return frontier.tsumo();
  }

  public int furitenFreeRonFrontierCopies() {
    return furitenFreeRonFrontierCopies;
  }

  public int furitenOnlyRonFrontierCopies() {
    return furitenOnlyRonFrontierCopies;
  }

  public int tsumoFrontierCopies() {
    return tsumoFrontierCopies;
  }
}

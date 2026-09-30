package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.core.Tile;

/** 手牌解析結果を保持する再利用バッファ。{@link EngineDecisionBuffer}から借用し、次の解析まで有効。 */
public final class HandAnalysisBuffer {

  private static final int RON_SLOT = 0;
  private static final int TSUMO_SLOT = 1;
  private static final int WIN_METHOD_COUNT = 2;
  private static final int HAN_OFFSET = 0;
  private static final int FU_OFFSET = 1;
  private static final int BASE_POINTS_OFFSET = 2;
  private static final int SCORE_VALUE_COUNT = 3;
  private static final int WAIT_SCORE_STRIDE = WIN_METHOD_COUNT * SCORE_VALUE_COUNT;

  private final byte[] waitTileTypes = new byte[Tile.NUM_TILE_TYPES];
  private final long[] waitYakuBits = new long[Tile.NUM_TILE_TYPES * WIN_METHOD_COUNT];
  private final int[] waitScoreValues = new int[Tile.NUM_TILE_TYPES * WAIT_SCORE_STRIDE];
  private long ronPaoMask;
  private long tsumoPaoMask;

  int minimumShanten;
  int standardShanten;
  int chiitoitsuShanten;
  int kokushiShanten;
  boolean specialHandShantenAvailable;
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
  int ronWaitTileTypesWithoutRiichi;
  int ronWaitCopiesWithoutRiichi;
  int tsumoWaitTileTypesWithoutRiichi;
  int tsumoWaitCopiesWithoutRiichi;
  final TenpaiFrontierBuffer tenpaiFrontier = new TenpaiFrontierBuffer();
  int furitenFreeRonFrontierCopies;
  int furitenOnlyRonFrontierCopies;
  int tsumoFrontierCopies;

  void clear() {
    minimumShanten = 0;
    standardShanten = 0;
    chiitoitsuShanten = 0;
    kokushiShanten = 0;
    specialHandShantenAvailable = false;
    liveImprovingCopies = 0;
    liveImprovingTileTypes = 0;
    improvingTileTypeMask = 0L;
    shapeWaitTileTypeMask = 0L;
    waitCount = 0;
    ronWaitTileTypeMask = 0L;
    tsumoWaitTileTypeMask = 0L;
    akaWinningTileAvailableMask = 0L;
    ronPaoMask = 0L;
    tsumoPaoMask = 0L;
    doraCount = 0;
    ownedAkaMask = 0;
    menzen = false;
    meldCount = 0;
    concealedTileCount = 0;
    improvingHitWithinOneDraw = 0.0f;
    improvingHitWithinTwoDraws = 0.0f;
    improvingHitWithinThreeDraws = 0.0f;
    liveShapeWaitCopies = 0;
    ronWaitTileTypesWithoutRiichi = 0;
    ronWaitCopiesWithoutRiichi = 0;
    tsumoWaitTileTypesWithoutRiichi = 0;
    tsumoWaitCopiesWithoutRiichi = 0;
    tenpaiFrontier.clear();
    furitenFreeRonFrontierCopies = 0;
    furitenOnlyRonFrontierCopies = 0;
    tsumoFrontierCopies = 0;
  }

  void appendWait(
      int tileType,
      HandScoreBuffer ronScore,
      HandScoreBuffer tsumoScore,
      boolean akaWinningTileAvailable,
      boolean ronPaoApplies,
      boolean tsumoPaoApplies) {
    int waitIndex = waitCount++;
    waitTileTypes[waitIndex] = (byte) tileType;
    writeScore(waitIndex, RON_SLOT, ronScore, ronPaoApplies);
    writeScore(waitIndex, TSUMO_SLOT, tsumoScore, tsumoPaoApplies);
    if (ronScore != null) {
      ronWaitTileTypeMask |= 1L << tileType;
    }
    if (tsumoScore != null) {
      tsumoWaitTileTypeMask |= 1L << tileType;
    }
    if (akaWinningTileAvailable) {
      akaWinningTileAvailableMask |= 1L << tileType;
    }
  }

  private void writeScore(
      int waitIndex, int methodSlot, HandScoreBuffer score, boolean paoApplies) {
    int yakuIndex = yakuIndex(waitIndex, methodSlot);
    int scoreIndex = scoreIndex(waitIndex, methodSlot, HAN_OFFSET);
    if (score == null) {
      waitYakuBits[yakuIndex] = 0L;
      setPao(waitIndex, methodSlot, false);
      waitScoreValues[scoreIndex] = 0;
      waitScoreValues[scoreIndex + FU_OFFSET] = 0;
      waitScoreValues[scoreIndex + BASE_POINTS_OFFSET] = 0;
      return;
    }
    waitYakuBits[yakuIndex] = score.yakuBits();
    setPao(waitIndex, methodSlot, paoApplies);
    waitScoreValues[scoreIndex] = score.han();
    waitScoreValues[scoreIndex + FU_OFFSET] = score.fu();
    waitScoreValues[scoreIndex + BASE_POINTS_OFFSET] = score.basePoints();
  }

  private void setPao(int waitIndex, int methodSlot, boolean applies) {
    long bit = 1L << waitIndex;
    if (methodSlot == RON_SLOT) {
      ronPaoMask = applies ? ronPaoMask | bit : ronPaoMask & ~bit;
    } else {
      tsumoPaoMask = applies ? tsumoPaoMask | bit : tsumoPaoMask & ~bit;
    }
  }

  private static int yakuIndex(int waitIndex, int methodSlot) {
    return waitIndex * WIN_METHOD_COUNT + methodSlot;
  }

  private static int scoreIndex(int waitIndex, int methodSlot, int valueOffset) {
    return waitIndex * WAIT_SCORE_STRIDE + methodSlot * SCORE_VALUE_COUNT + valueOffset;
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

  public boolean specialHandShantenAvailable() {
    return specialHandShantenAvailable;
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

  public long ronYakuBits(int waitIndex) {
    return waitYakuBits[yakuIndex(waitIndex, RON_SLOT)];
  }

  public long tsumoYakuBits(int waitIndex) {
    return waitYakuBits[yakuIndex(waitIndex, TSUMO_SLOT)];
  }

  public int ronHanWithoutUra(int waitIndex) {
    return waitScoreValues[scoreIndex(waitIndex, RON_SLOT, HAN_OFFSET)];
  }

  public int ronFu(int waitIndex) {
    return waitScoreValues[scoreIndex(waitIndex, RON_SLOT, FU_OFFSET)];
  }

  public boolean ronPaoApplies(int waitIndex) {
    return (ronPaoMask & (1L << waitIndex)) != 0L;
  }

  public int ronBasePoints(int waitIndex) {
    return waitScoreValues[scoreIndex(waitIndex, RON_SLOT, BASE_POINTS_OFFSET)];
  }

  public int tsumoHanWithoutUra(int waitIndex) {
    return waitScoreValues[scoreIndex(waitIndex, TSUMO_SLOT, HAN_OFFSET)];
  }

  public int tsumoFu(int waitIndex) {
    return waitScoreValues[scoreIndex(waitIndex, TSUMO_SLOT, FU_OFFSET)];
  }

  public boolean tsumoPaoApplies(int waitIndex) {
    return (tsumoPaoMask & (1L << waitIndex)) != 0L;
  }

  public int tsumoBasePoints(int waitIndex) {
    return waitScoreValues[scoreIndex(waitIndex, TSUMO_SLOT, BASE_POINTS_OFFSET)];
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

  public int ronWaitTileTypesWithoutRiichi() {
    return ronWaitTileTypesWithoutRiichi;
  }

  public int ronWaitCopiesWithoutRiichi() {
    return ronWaitCopiesWithoutRiichi;
  }

  public int tsumoWaitTileTypesWithoutRiichi() {
    return tsumoWaitTileTypesWithoutRiichi;
  }

  public int tsumoWaitCopiesWithoutRiichi() {
    return tsumoWaitCopiesWithoutRiichi;
  }

  public long furitenFreeRonFrontierTileTypeMask() {
    return tenpaiFrontier.furitenFreeRon();
  }

  public long furitenOnlyRonFrontierTileTypeMask() {
    return tenpaiFrontier.furitenOnlyRon();
  }

  public long tsumoFrontierTileTypeMask() {
    return tenpaiFrontier.tsumo();
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

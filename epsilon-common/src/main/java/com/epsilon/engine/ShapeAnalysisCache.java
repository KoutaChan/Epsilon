package com.epsilon.engine;

import com.epsilon.calculate.shape.HandShapeAnalysisBuffer;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.core.HandView;

/** 判断に使うバッファ内で手牌の解析結果を再利用する、256件のダイレクトマップ方式のキャッシュ。 */
final class ShapeAnalysisCache {

  private static final int CAPACITY = 256;
  private static final int INDEX_MASK = CAPACITY - 1;

  private final long[] packedCountsLow = new long[CAPACITY];
  private final long[] packedCountsHigh = new long[CAPACITY];
  private final long[] improvingTileTypeMasks = new long[CAPACITY];
  private final long[] agariTileTypeMasks = new long[CAPACITY];
  private final byte[] meldCounts = new byte[CAPACITY];
  private final byte[] minimumShanten = new byte[CAPACITY];
  private final byte[] standardShanten = new byte[CAPACITY];
  private final byte[] sevenPairsShanten = new byte[CAPACITY];
  private final byte[] thirteenOrphansShanten = new byte[CAPACITY];
  private final boolean[] specialHandsAvailable = new boolean[CAPACITY];
  private final boolean[] occupied = new boolean[CAPACITY];
  private final HandShapeState shape = new HandShapeState();
  private final HandShapeAnalysisBuffer analysis = new HandShapeAnalysisBuffer();
  private final HandShapeAnalyzer analyzer = new HandShapeAnalyzer();

  void analyzeInto(HandView hand, DecisionHandAnalysisBuffer destination) {
    hand.copyShapeInto(shape);
    long low = shape.packedConcealedTileCountsLow();
    long high = shape.packedConcealedTileCountsHigh();
    int meldCount = hand.meldCount();
    int index = index(low, high, meldCount);
    if (occupied[index]
        && packedCountsLow[index] == low
        && packedCountsHigh[index] == high
        && meldCounts[index] == meldCount) {
      EngineDecisionAnalysisMetrics.recordShape(true);
      read(index, destination);
      return;
    }

    EngineDecisionAnalysisMetrics.recordShape(false);
    analyzer.analyzeInto(hand, analysis);
    packedCountsLow[index] = low;
    packedCountsHigh[index] = high;
    meldCounts[index] = (byte) meldCount;
    minimumShanten[index] = (byte) analysis.minimumShanten();
    standardShanten[index] = (byte) analysis.standardShanten();
    sevenPairsShanten[index] = (byte) analysis.chiitoitsuShanten();
    thirteenOrphansShanten[index] = (byte) analysis.kokushiShanten();
    specialHandsAvailable[index] = analysis.specialHandsAvailable();
    improvingTileTypeMasks[index] = analysis.ukeireTileTypeMask();
    agariTileTypeMasks[index] = analysis.agariTileTypeMask();
    occupied[index] = true;
    read(index, destination);
  }

  private void read(int index, DecisionHandAnalysisBuffer destination) {
    destination.minimumShanten = minimumShanten[index];
    destination.standardShanten = standardShanten[index];
    destination.chiitoitsuShanten = sevenPairsShanten[index];
    destination.kokushiShanten = thirteenOrphansShanten[index];
    destination.specialHandsAvailable = specialHandsAvailable[index];
    destination.improvingTileTypeMask = improvingTileTypeMasks[index];
    destination.shapeWaitTileTypeMask = agariTileTypeMasks[index];
  }

  private static int index(long low, long high, int meldCount) {
    long mixed = low ^ Long.rotateLeft(high, 23) ^ (long) meldCount * 0x9E37_79B9_7F4A_7C15L;
    mixed ^= mixed >>> 33;
    mixed *= 0xFF51_AFD7_ED55_8CCDL;
    mixed ^= mixed >>> 33;
    return (int) mixed & INDEX_MASK;
  }
}

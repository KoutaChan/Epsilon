package com.epsilon.calculate.shape;

/**
 * 手牌の解析結果を保持し、次の解析でも再利用するバッファ。
 *
 * <p>取得メソッドは直前の{@link HandShapeAnalyzer#analyzeInto}の結果を返す。複数スレッドから同時に利用してはならない。
 */
public final class HandShapeAnalysisBuffer {

  int minimumShanten;
  int standardShanten;
  int chiitoitsuShanten;
  int kokushiShanten;
  boolean specialHandsAvailable;
  long ukeireTileTypeMask;
  long agariTileTypeMask;

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

  public long ukeireTileTypeMask() {
    return ukeireTileTypeMask;
  }

  public long agariTileTypeMask() {
    return agariTileTypeMask;
  }
}

package com.epsilon.prism.noise;

/**
 * TEMPO で学習率に掛ける倍率の決め方。どちらのモードでも、教師の混ぜ具合と補助のヘッドの項に掛ける K′_t は同じで、違うのは学習率の
 * ピークと下限に掛ける倍率だけ。倍率は (0, 1] に入る。
 */
public enum TempoLrMode {
  /** 倍率 min(1, K′_t / 基準のゲイン)。アンカーの重み 1 − K′_t が大きいほど、1 ステップを小さくする。 */
  GAIN,
  /** 倍率 1。 */
  FULL;

  /**
   * ゲインが K′_t のときの学習率の倍率を返す。
   *
   * @param gain K′_t（(0, 1]）
   * @param reference {@link #GAIN} で使う基準のゲイン
   * @return 学習率の倍率
   */
  public double scale(double gain, double reference) {
    return switch (this) {
      case GAIN -> Math.min(1, gain / reference);
      case FULL -> 1;
    };
  }
}

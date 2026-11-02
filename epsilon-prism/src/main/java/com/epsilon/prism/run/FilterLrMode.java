package com.epsilon.prism.run;

/**
 * 系譜の学習率の倍率の決め方（設定 {@code epsilon.prism.filterLrMode}、{@link PrismConfig#filterLrScale(int, double)}）。
 * どちらでも教師 (1 − K′_t)·s_θt + K′_t·Ỹ と補助の頭の項の K′_t は同じ予定のままで、変わるのは学習率 lrPeak・lrFloor に掛ける倍率だけ。倍率は
 * (0, 1] に入る。
 */
public enum FilterLrMode {

  /** 倍率 min(1, K′_t / {@value PrismConfig#FILTER_LR_GAIN})。錨の重み 1 − K′_t が大きいほど 1 歩を小さくする。 */
  GAIN,

  /** 倍率 1。K′_t が減っても学習率は lrPeak・lrFloor のまま（K′_t は教師の混ぜ具合だけを決める）。 */
  FULL;

  /**
   * ゲイン K′_t のときの学習率の倍率。
   *
   * @param gain K′_t（(0, 1]）
   * @return 倍率（(0, 1]）
   */
  public double scale(double gain) {
    return switch (this) {
      case GAIN -> Math.min(1, gain / PrismConfig.FILTER_LR_GAIN);
      case FULL -> 1;
    };
  }
}

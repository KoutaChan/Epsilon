package com.epsilon.prism.noise;

/** TEMPO の K′_t に、教師のノイズがどれだけ減ったかを入れる方法。 */
public enum NoiseGain {
  /** 入れない（K′ = K）。 */
  NONE,
  /** 前の反復で教師を運の精算で補正したときの、ホールドアウトで測った R² を使う（K′ = K / (K + (1 − K)(1 − R²))）。 */
  LEDGER_R2,
  /** ノイズの比に、間引きの 1/q で増える分を掛けて使う。 */
  EFFECTIVE
}

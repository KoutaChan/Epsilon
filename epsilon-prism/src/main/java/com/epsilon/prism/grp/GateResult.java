package com.epsilon.prism.grp;

/** 候補の GRP の関門の結果（{@link GateEvaluator}）。PASS 以外は落ちた関門。 */
public enum GateResult {

  /** 全ての関門を通った。 */
  PASS,

  /** 対の MSE の差（候補 − 使用中）が −2 SE に届かない。 */
  NOT_BETTER,

  /** 順位の当てはまり（NLL・4 位の ECE。点数だけの Φ からは絶対の較正）が悪い。 */
  RANK_FIT_WORSE,

  /** 使用中からの ΔΦ か差の変化が線を越える。 */
  SHIFT_TOO_LARGE,

  /** 半荘の偶奇で分けた 2 つの候補の ΔΦ の RMS が線を越える。 */
  UNSTABLE,

  /** 点を動かして Φ が逆に動く率が線を越える。 */
  MONOTONIC_VIOLATIONS,

  /** 4000 点の傾きの使用中に対する比が範囲の外。 */
  SLOPE_OUT_OF_RANGE,

  /** 窓の半荘か取り置きのコホートが足りない。 */
  NOT_ENOUGH_DATA,

  /** 値が有限でないか、構造が grp-epsilon-v3 でない。 */
  INVALID_OUTPUT
}

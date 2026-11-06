package com.epsilon.prism.grp;

/** 候補の GRP の関門の結果（{@link GateEvaluator}）。PASS 以外は落ちた関門。 */
public enum GateResult {

  /** 全ての関門を通った。 */
  PASS,

  /** 対の MSE の差（候補 − 使用中）が −2 SE に届かない。 */
  NOT_BETTER,

  /** 点を動かして Φ が逆に動く率が線を越える。 */
  MONOTONIC_VIOLATIONS,

  /** 窓の半荘か取り置きのコホートが足りない。 */
  NOT_ENOUGH_DATA,

  /** 値が有限でないか、構造が grp-epsilon-v3 でない。 */
  INVALID_OUTPUT
}

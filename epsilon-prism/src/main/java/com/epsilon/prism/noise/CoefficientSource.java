package com.epsilon.prism.noise;

/** 行に掛ける帳簿の係数を、どの行で当てはめるか。どちらの方法でも、その行と同じ半荘の行は使わない。 */
public enum CoefficientSource {
  /** 半荘の番号の偶奇が逆の行。 */
  OTHER_PARITY,
  /** 前のコホートの行。持ち越した係数が無い反復では、代わりに偶奇が逆の行を使う。 */
  PREVIOUS_COHORT
}

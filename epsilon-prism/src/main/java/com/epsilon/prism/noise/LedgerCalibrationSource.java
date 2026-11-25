package com.epsilon.prism.noise;

/** 帳簿の補正率を当てはめる行を、どのコホートから取るか。どの選び方でも、補正率を掛ける反復のコホートの行は使わない。 */
public enum LedgerCalibrationSource {
  /** 前の完了したコホートの学習の行。反復 t のコホートで当てはめた補正率を、反復 t + 1 の教師に掛ける。 */
  PREVIOUS_COHORT
}

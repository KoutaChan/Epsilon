package com.epsilon.grp.train;

/** 学習の終わり方。STOPPED で終わった結果はネットワークを持たない（GrpTrained と OutcomeTrained のネットワークは null）。 */
public enum TrainingEnding {
  /** エポックの上限まで回したか、損失が良くならず早期終了した。 */
  COMPLETED,
  /** 停止のフラグが立ち、途中で止めた。 */
  STOPPED
}

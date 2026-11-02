package com.epsilon.prism.grp;

/** 局末の値 Φ の出どころ（設定 {@code epsilon.prism.phiSource}）。 */
public enum PhiSource {

  /** grpCheckpoint の固定の GRP に phiTemperatures・phiCalibration を重ねたもの。run の全ての反復で同じ。 */
  FIXED_GRP,

  /**
   * grp/phi.jsonl の項目（{@link PhiHistory}）。反復 t の Φ は effectiveFrom ≤ t の最後の項目で、最初の項目 g000 は phiStart で決め、
   * 自己対局で学んだ GRP（{@link PhiEntryKind#SELF_CHECKPOINT}）に反復の境目で切り替える。
   */
  SELF_GRP,

  /** 局末の点数をそのまま最終順位とみなす固定の代用（GRP を開かない）。 */
  SCORE_RANK,

  /** 局末の点数のなめらかな関数の固定の代用（{@link ScoreSmoothPhi}。GRP を開かない）。 */
  SCORE_SMOOTH
}

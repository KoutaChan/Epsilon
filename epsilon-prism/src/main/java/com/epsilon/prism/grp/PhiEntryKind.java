package com.epsilon.prism.grp;

/** grp/phi.jsonl の 1 項目の Φ の種類（{@link PhiEntry#kind}）。 */
public enum PhiEntryKind {

  /** 局末の点数をそのまま最終順位とみなす（点数の順位の効用）。GRP も較正も読まない。 */
  SCORE_RANK(true),

  /** 元の GRP（grpCheckpoint）に phiTemperatures・phiCalibration を重ねたもの。 */
  FIXED_CHECKPOINT(false),

  /** 自己対局で学んだ GRP。温度も較正の表も重ねない。 */
  SELF_CHECKPOINT(false),

  /** 局末の点数のなめらかな関数（{@link ScoreSmoothPhi}）。GRP も較正も読まない。 */
  SCORE_SMOOTH(true);

  private final boolean scoreBased;

  PhiEntryKind(boolean scoreBased) {
    this.scoreBased = scoreBased;
  }

  /** 点数だけで決まる Φ（GRP の checkpoint を読まない。{@link ScorePhi}）なら true。 */
  public boolean scoreBased() {
    return scoreBased;
  }
}

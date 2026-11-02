package com.epsilon.prism.grp;

/** SELF_GRP の最初の項目 g000 の Φ（設定 {@code epsilon.prism.phiStart}）。 */
public enum PhiStart {

  /** grpCheckpoint の GRP に phiTemperatures・phiCalibration を重ねたもの（grpCheckpoint が要る）。 */
  GRP_CHECKPOINT(PhiEntryKind.FIXED_CHECKPOINT),

  /** 局末の点数の順位（GRP なしの出発。grpCheckpoint・phiTemperatures・phiCalibration は空）。 */
  SCORE_RANK(PhiEntryKind.SCORE_RANK),

  /**
   * 局末の点数のなめらかな関数（{@link ScoreSmoothPhi}。GRP なしの出発。grpCheckpoint・phiTemperatures・phiCalibration は空）。
   */
  SCORE_SMOOTH(PhiEntryKind.SCORE_SMOOTH);

  private final PhiEntryKind kind;

  PhiStart(PhiEntryKind kind) {
    this.kind = kind;
  }

  /** g000 の項目の種類。 */
  public PhiEntryKind kind() {
    return kind;
  }
}

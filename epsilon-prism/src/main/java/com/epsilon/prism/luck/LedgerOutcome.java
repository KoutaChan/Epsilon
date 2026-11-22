package com.epsilon.prism.luck;

/**
 * 学習の段で行う運の精算の結果。
 *
 * @param report 係数の当てはめの報告（帳簿のモードが係数を当てはめないときは null）
 * @param summary 段の報告に書く要約（係数を当てはめないときは null）
 * @param correction 教師に掛ける補正。補正するモードで、帳簿のある行があったときだけ持ち、ほかは null
 * @param appliedR2 教師に掛けた補正のホールドアウトの R²。補正を掛けないときと、R² が負か有限でないときは 0。次の反復で K′ を決めるのに
 *     使う
 * @param calibration 帳簿の補正率の校正の結果（補正率を求めないモードでは null）
 */
public record LedgerOutcome(
    LedgerReport report,
    LedgerReport.Summary summary,
    LedgerCorrection correction,
    double appliedR2,
    Calibration calibration,
    RootAudit.Report roots,
    RootTraining.State rootNext,
    LedgerCorrection.ByClass comparison) {

  /** 係数を当てはめないときの結果。 */
  public static final LedgerOutcome NONE = new LedgerOutcome(null, null, null, 0);

  public LedgerOutcome(
      LedgerReport report,
      LedgerReport.Summary summary,
      LedgerCorrection correction,
      double appliedR2) {
    this(report, summary, correction, appliedR2, null, null, null, null);
  }

  /**
   * 帳簿の補正率の校正の結果。
   *
   * @param report 報告（和了の判断の教師の差は {@link LuckSettlement#gates} が足す）
   * @param current 帳簿の補正（補正率を掛けない β·L。β は教師に掛ける補正と同じ）
   * @param scale 今のコホートに掛けた補正率（出どころが無ければ null で、補正率は 1）
   * @param next 今のコホートの行で当てはめた、次の反復の補正率（当てはめられなければ null）
   */
  public record Calibration(
      LedgerReport.Calibration report,
      LedgerCorrection.ByClass current,
      LedgerCalibrator scale,
      LedgerCalibrator next) {}

  /** 判断のクラスごとの要約を返す（係数を当てはめないときは空）。 */
  public LedgerReport.Classes classes() {
    return report == null ? LedgerReport.Classes.NONE : LedgerReport.Classes.of(report);
  }

  /** 書き出す報告の全部（補正率を求めた反復では、その報告も入れる）。 */
  public LedgerReport.Full full(LedgerRecording recording) {
    return report.full(
        recording,
        calibration == null ? null : calibration.report(),
        roots,
        correction instanceof LedgerCorrection.ByClass classic ? classic.scale() : null);
  }

  /** 次の反復に渡す補正率（補正率を求めないか、当てはめられなければ null）。 */
  public LedgerCalibrator next() {
    return calibration == null ? null : calibration.next();
  }
}

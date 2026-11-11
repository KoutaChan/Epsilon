package com.epsilon.prism.luck;

/**
 * 学習の段で行う運の精算の結果。
 *
 * @param report 係数の当てはめの報告（帳簿のモードが係数を当てはめないときは null）
 * @param summary 段の報告に書く要約（係数を当てはめないときは null）
 * @param correction 教師に掛ける補正。補正するモードで、帳簿のある行があったときだけ持ち、ほかは null
 * @param appliedR2 教師に掛けた補正のホールドアウトの R²。補正を掛けないときと、R² が負か有限でないときは 0。次の反復で K′ を決めるのに
 *     使う
 */
public record LedgerOutcome(
    LedgerReport report, LedgerReport.Summary summary, LedgerCorrection correction, double appliedR2) {

  /** 係数を当てはめないときの結果。 */
  public static final LedgerOutcome NONE = new LedgerOutcome(null, null, null, 0);

  /** 判断のクラスごとの要約を返す（係数を当てはめないときは空）。 */
  public LedgerReport.Classes classes() {
    return report == null ? LedgerReport.Classes.NONE : LedgerReport.Classes.of(report);
  }
}

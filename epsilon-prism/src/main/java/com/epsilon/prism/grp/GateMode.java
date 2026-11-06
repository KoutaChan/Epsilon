package com.epsilon.prism.grp;

/** 関門の表の列（{@link GateEvaluator}）。使用中の Φ の種類で決まる。 */
public enum GateMode {

  /** 使用中が GRP: 候補が良くなったか、順位の当てはまり・変化の幅・単調・傾きを使用中と比べる。 */
  REFIT,

  /** 使用中が点数だけの Φ（最初の置き換え）: 候補が使用中より良くなったか・単調・窓の半荘の数。 */
  FIRST_REPLACEMENT;

  /** 使用中の項目の種類の列。 */
  public static GateMode of(PhiEntryKind active) {
    return active.scoreBased() ? FIRST_REPLACEMENT : REFIT;
  }
}

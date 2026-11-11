package com.epsilon.prism.grp;

/** 局末の値 Φ の出どころ。 */
public enum PhiSource {

  /** 固定の GRP。GruSinkhorn なら PhiTable を重ねる。run の全ての反復で同じ Φ を使う。 */
  FIXED_GRP,

  /**
   * 自分の GRP。grp/phi.jsonl の項目（{@link PhiHistory}）を使う。最初の項目 g000 は設定で決め、その後は自己対局の記録で学び直した
   * ネットワークに、反復の区切りで切り替える（{@link GrpLifecycle}）。
   */
  SELF_GRP,

  /** 局末の点数の順位を、そのまま最終順位とみなす固定の Φ。GRP は開かない。 */
  SCORE_RANK,

  /** 局末の点数のなめらかな関数で表す固定の Φ。GRP は開かない。 */
  SCORE_SMOOTH
}

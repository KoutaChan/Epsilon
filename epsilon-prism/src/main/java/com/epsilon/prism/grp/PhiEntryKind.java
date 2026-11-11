package com.epsilon.prism.grp;

/** grp/phi.jsonl の項目が表す Φ の種類（{@link PhiEntry#kind}）。 */
public enum PhiEntryKind {

  /** 局末の点数の順位をそのまま最終順位とみなし、その順位の効用を値にする。チェックポイントも表も読まない。 */
  SCORE_RANK(true),

  /** 局末の点数のなめらかな関数。チェックポイントも表も読まない。 */
  SCORE_SMOOTH(true),

  /** GRP のチェックポイント（grp.json と weights.f32）。中身が GruSinkhorn なら表を重ねてよい。 */
  NETWORK(false);

  private final boolean scoreBased;

  PhiEntryKind(boolean scoreBased) {
    this.scoreBased = scoreBased;
  }

  /** 点数だけで決まる Φ（チェックポイントを読まない）なら true。 */
  public boolean scoreBased() {
    return scoreBased;
  }
}

package com.epsilon.prism.grp;

/** refits.jsonl の 1 行の状態（{@link RefitRecord#status}）。 */
public enum RefitStatus {

  /** 関門を通り、候補を公開した（学習の JVM が読むのはこれだけ）。 */
  READY,

  /** 関門に落ちたか、学習の JVM の境目の点検（読めない・sha が違う・値が合わない）に落ちた。 */
  REJECTED,

  /** 学習が締め切りを過ぎて止めた。 */
  TIMEOUT,

  /** 学習か公開が例外で失敗した。 */
  FAILED,

  /** AUTO の候補が効く反復 h + L の時点で READY でなかったので入れなかった。 */
  SKIPPED_NOT_READY;

  /** 予定表の連続の失敗（grpMaxFailures）に数えるなら true。 */
  public boolean failure() {
    return this == REJECTED || this == TIMEOUT || this == FAILED;
  }
}

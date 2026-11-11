package com.epsilon.grp.train;

/**
 * 関門の種類と、候補を使用中のモデルと比べる基準。比べるのは関門のコホートでの対応のある MSE の差（候補 − 使用中）で、基準は SE の
 * 倍数で表す。毎回の再学習では「良くなった」ことまでは求めない。2 反復分の改善は、コホート 1 つで測る SE より小さいため。
 */
public enum GateKind {
  /** 毎回の再学習。差が +2 SE 以下（悪くなっていない）なら通す。 */
  REFIT(2, false, GateFailure.WORSE),
  /** 最初の切り替えと、モデルの構造の入れ替え。学習に使っていない関門の半荘で、差が −2 SE より小さい（良くなった）なら通す。 */
  REPLACEMENT(-2, true, GateFailure.NOT_BETTER);

  private final double standardErrors;
  private final boolean strict;
  private final GateFailure failure;

  GateKind(double standardErrors, boolean strict, GateFailure failure) {
    this.standardErrors = standardErrors;
    this.strict = strict;
    this.failure = failure;
  }

  /** 基準の SE の倍数（良くなったことを求める関門では負）。 */
  double standardErrors() {
    return standardErrors;
  }

  /** 使用中のモデルとの差が基準を満たせば true。NaN では満たさない。 */
  boolean accepts(Paired versusUsed) {
    double line = standardErrors * versusUsed.standardError();
    return strict ? versusUsed.mean() < line : versusUsed.mean() <= line;
  }

  /** 基準を満たさないときに報告する、落ちた理由。 */
  GateFailure failure() {
    return failure;
  }
}

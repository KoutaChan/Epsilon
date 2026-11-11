package com.epsilon.prism.run;

import com.epsilon.agent.run.Progress;
import java.util.Objects;

/**
 * PRISM の系譜の進み具合（state.json）。反復を閉じたときに書く。王者との決闘の結果を反映したときは、リグレッションの記録と始めの
 * バージョンを書き直す。
 *
 * @param next 次に回す反復（反復 next − 1 までを閉じ、バージョン next まで公開した）
 * @param totalRows これまでのコホートの行の総数
 * @param carry 次の反復へ持ち越す値
 * @param start 次の反復が打ち、生徒の学習を始めるバージョン（ふだんは next。王者へ戻したときは王者のバージョン）
 * @param regression 王者との決闘でのリグレッションの記録
 */
public record PrismState(int next, long totalRows, PrismCarry carry, int start, Regression regression)
    implements Progress {

  public PrismState {
    Objects.requireNonNull(carry, "carry");
    Objects.requireNonNull(regression, "regression");
    if (next < 1 || start < 1 || start > next) {
      throw new IllegalArgumentException("Invalid PRISM state: next " + next + ", start " + start);
    }
    if (totalRows < 0) {
      throw new IllegalArgumentException("Invalid PRISM state: rows " + totalRows);
    }
  }

  /** 始めの state（次の反復は 1 で、バージョン 1 まで公開した）。 */
  public static PrismState initial() {
    return new PrismState(1, 0, PrismCarry.NONE, 1, Regression.NONE);
  }

  /** リグレッションの記録と始めのバージョンだけを替えた state を返す。 */
  public PrismState regressed(Regression regression, int start) {
    return new PrismState(next, totalRows, carry, start, regression);
  }
}

package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.rows.Row;

/**
 * 局末の値を待つ行の、値のほかの部分。判断の時点で、エンジンを進める前に決まる。
 *
 * @param seat 判断の席
 * @param source MAIN（主経路のラベル行）・ROLLIN（写しの中のラベル行）・ROOT（根の記録）
 * @param played その経路で打った手の位置
 * @param actions 合法手の数
 * @param prior 合法手ごとの p0（{@link PolicyRequest#prior}）
 * @param support 候補集合 C(o)（{@link CandidateSet#bytes}。同値な打牌は切る方の手）
 * @param deviation ROOT の逸脱 d の位置（ほかは -1）
 * @param deviationProbability ROOT の q(d|o)（ほかは NaN）
 * @param propensity ROOT の傾向スコア β_c(o)·q(d|o)（ほかは NaN）
 * @param rootDecisionId ROLLIN は写しの根の判断 ID、ほかは -1
 * @param record 判断の記録 O と Oracle（{@link RowRecorder.Decision}）
 */
record RowDraft(
    long gameId,
    long decisionId,
    int seat,
    DecisionClass decisionClass,
    Row.Purpose purpose,
    Row.Source source,
    int played,
    int actions,
    float[] prior,
    byte[] support,
    int deviation,
    float deviationProbability,
    float propensity,
    long rootDecisionId,
    RowRecorder.Decision record) {
  /** 帳簿を付けない行の帳簿（列が 0 個）。 */
  private static final float[] NO_LEDGER = new float[0];

  /**
   * 値がそろった行を作る。p0・帳簿・局末の 3 ワードは渡した配列をそのまま持ち、候補集合は手の位置の並びをビットにする。
   *
   * @param values 合法手ごとの Y（測っていない手は NaN。ROOT では played と deviation だけ）
   * @param ledger 手ごとの運の精算の帳簿（帳簿を付けない収集では null）
   * @param outcomes 手ごとの局末の 3 ワード
   */
  Row row(float[] values, float[] ledger, long[] outcomes) {
    int bits = 0;
    for (byte position : support) bits |= 1 << Byte.toUnsignedInt(position);
    return new Row(
        gameId,
        decisionId,
        decisionClass,
        purpose,
        source,
        played,
        values,
        prior,
        bits,
        deviation,
        deviationProbability,
        propensity,
        rootDecisionId,
        ledger == null ? NO_LEDGER : ledger,
        outcomes,
        record.oracle(),
        record.observation());
  }
}

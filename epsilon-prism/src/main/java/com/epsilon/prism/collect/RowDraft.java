package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.rows.Row;

/**
 * 局末の値を待つ行の、値以外の部分。判断の時点（commit の前）に確定する。
 *
 * @param seat 判断の席
 * @param kind 手の種別（帳簿の測定の付録 {@link EventLedger#dump} の欄）
 * @param source MAIN（主経路のラベル行）・ROLLIN（写しの中のラベル行）・ROOT（根の記録）
 * @param played その経路で打った手の位置
 * @param actions 合法手の数
 * @param prior 合法手ごとの p0（{@link PolicyRequest#prior}）
 * @param support 候補集合 C(o)（{@link CandidateSet#supportBytes}）
 * @param deviation ROOT の逸脱 d の位置（ほかは -1）
 * @param deviationProbability ROOT の q(d|o)（ほかは NaN）
 * @param propensity ROOT の傾向 β_c(o)·q(d|o)（ほかは NaN）
 * @param rootDecisionId ROLLIN は写しの根の判断 ID、ほかは -1
 * @param record 判断の記録 O と神託（{@link RowRecorder.Decision}）
 */
record RowDraft(
    long gameId,
    long decisionId,
    int seat,
    Kind kind,
    DecisionClass decisionClass,
    Row.Purpose purpose,
    Row.Group group,
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
  /** 帳簿を付けない行の帳簿（欄 0）。 */
  private static final float[] NO_LEDGER = new float[0];

  /**
   * そろった行。p0・帳簿・局末の 3 語は渡した配列をそのまま持ち、候補集合は位置の並びをビットにする。
   *
   * @param values 合法手ごとの Y（測っていない手は NaN。ROOT は played と deviation の枠だけ）
   * @param ledger 手ごとの運の精算の帳簿（帳簿を付けない収集は null）
   * @param outcomes 手ごとの局末の 3 語
   */
  Row row(float[] values, float[] ledger, long[] outcomes) {
    int bits = 0;
    for (byte position : support) bits |= 1 << Byte.toUnsignedInt(position);
    return new Row(
        gameId,
        decisionId,
        decisionClass,
        purpose,
        group,
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

package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.rows.ObservationPayload;
import com.epsilon.prism.rows.PrismRow;

/**
 * 局末の値を待つ行の、値以外の部分。判断の時点（commit の前）に確定する。
 *
 * @param source MAIN（主経路のラベル行）・ROLLIN（写しの中のラベル行）・ROOT（根の記録）
 * @param played その経路で打った手の位置
 * @param prior 合法手ごとの p0（{@link com.epsilon.prism.policy.PolicyRequest#prior}）
 * @param support 候補集合 C(o)（{@link CandidateSet#supportBytes}）
 * @param rootDecisionId ROLLIN は写しの根の判断 ID、ほかは -1
 */
record RowDraft(
    int iteration,
    long gameId,
    long decisionId,
    int seat,
    Kind kind,
    Stage stage,
    DecisionClass decisionClass,
    PrismRow.Purpose purpose,
    PrismRow.Group group,
    PrismRow.Source source,
    int played,
    int actions,
    float[] prior,
    byte[] support,
    long rootDecisionId,
    ObservationPayload payload) {
  /** ラベル行（MAIN・ROLLIN）。values は合法手ごとの Y（測っていない手は NaN）。 */
  PrismRow label(float[] values) {
    if (source == PrismRow.Source.ROOT) throw new IllegalStateException("Root draft");
    return new PrismRow(
        iteration,
        gameId,
        decisionId,
        seat,
        kind,
        stage,
        decisionClass,
        purpose,
        group,
        source,
        played,
        actions,
        values,
        prior,
        support,
        -1,
        Float.NaN,
        Float.NaN,
        rootDecisionId,
        payload);
  }

  /**
   * 根の記録（ROOT）。values は played と deviation の枠だけが値（d = played なら played の 1 枠）。
   *
   * @param deviationProbability q(d|o)
   * @param propensity β_c(o)·q(d|o)
   */
  PrismRow root(float[] values, int deviation, float deviationProbability, float propensity) {
    if (source != PrismRow.Source.ROOT) throw new IllegalStateException("Label draft");
    return new PrismRow(
        iteration,
        gameId,
        decisionId,
        seat,
        kind,
        stage,
        decisionClass,
        purpose,
        group,
        source,
        played,
        actions,
        values,
        prior,
        support,
        deviation,
        deviationProbability,
        propensity,
        -1,
        payload);
  }
}

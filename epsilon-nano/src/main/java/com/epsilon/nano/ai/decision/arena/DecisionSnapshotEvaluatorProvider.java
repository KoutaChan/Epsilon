package com.epsilon.nano.ai.decision.arena;

import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluator;

/** スナップショット ID に対応するチェックポイントを読み込んだ、方策の推論器を返す。 */
public interface DecisionSnapshotEvaluatorProvider {

  EpsilonDecisionEvaluator evaluatorFor(long snapshotId) throws Exception;
}

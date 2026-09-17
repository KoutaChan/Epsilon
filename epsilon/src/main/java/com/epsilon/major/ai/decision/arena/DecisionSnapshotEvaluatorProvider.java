package com.epsilon.major.ai.decision.arena;

import com.epsilon.major.ai.decision.runtime.EpsilonDecisionEvaluator;

/** 保存済みモデルの ID から対応するチェックポイントを読み込み、重みを固定した方策評価器を取得する。 */
public interface DecisionSnapshotEvaluatorProvider {

  EpsilonDecisionEvaluator evaluatorFor(long snapshotId) throws Exception;
}

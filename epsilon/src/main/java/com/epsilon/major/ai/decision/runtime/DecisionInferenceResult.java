package com.epsilon.major.ai.decision.runtime;

import java.util.List;

/** デバイス上の一物理バッチをホスト側の最終表現へ復号した結果です。 */
public sealed interface DecisionInferenceResult
    permits DecisionInferenceResult.Predictions, DecisionInferenceResult.GreedyActionSlots {

  /** 結果に含まれる行数を返します。 */
  int size();

  /** 通常自己対局が消費する方策と任意の価値結果です。 */
  record Predictions(List<EpsilonDecisionInferenceServer.Prediction> values)
      implements DecisionInferenceResult {
    @Override
    public int size() {
      return values.size();
    }
  }

  /** 固定対戦評価が消費する各行のargmax 行動候補の位置です。 */
  record GreedyActionSlots(int[] values) implements DecisionInferenceResult {
    @Override
    public int size() {
      return values.length;
    }
  }

  /** 物理バッチが要求するホスト結果表現です。 */
  enum Kind {
    PREDICTIONS,
    GREEDY_ACTION_SLOTS
  }
}

package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;

/** 複数の後続打牌を持つ行動について、遷移集約の勾配をどの目的へ返すか。 */
public enum TransitionPooling {
  /** Nanoは打牌スコアを実際に選んだ打牌だけから学習し、鳴き表現からは更新しない。 */
  SELECTED_DISCARD_POLICY {
    @Override
    NDArray weights(NDArray scores, NDArray mask) {
      return DecisionPolicyTensorOps.maskedTransitionWeights(scores, mask);
    }
  },
  /** 回帰の候補表現は、遷移の集約重みも同じ候補目的から学習する。 */
  CANDIDATE_REPRESENTATION {
    @Override
    NDArray weights(NDArray scores, NDArray mask) {
      return NDArrays.maskedSoftmax(scores, mask, 2);
    }
  };

  abstract NDArray weights(NDArray scores, NDArray mask);
}

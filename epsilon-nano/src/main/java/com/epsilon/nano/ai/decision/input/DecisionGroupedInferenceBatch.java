package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;
import java.util.List;

/**
 * 容量区分をまたいで盤面と実在遷移を共有する推論入力。
 *
 * <p>入力と遷移集約は元の容量区分を保ち、候補計算には合法位置と元盤面の対応を渡す。 すべてのビューは転送先の管理元に属し、利用中は元の連続バッファを保持する。
 */
public record DecisionGroupedInferenceBatch(
    NDArray stateCategories,
    NDArray stateNumerics,
    NDArray playerMemoryPresentIndices,
    DecisionTransitionInputs transitions,
    NDArray transitionBatchIds,
    NDArray actionBatchIds,
    List<Group> groups) {

  public DecisionGroupedInferenceBatch {
    groups = List.copyOf(groups);
  }

  public int rowCount() {
    return Math.toIntExact(stateCategories.getShape().get(0));
  }

  public int actionCount() {
    return Math.toIntExact(actionBatchIds.size());
  }

  /** 元の容量区分と合法候補の位置、共有した遷移表現の連続範囲。 */
  public record Group(
      DecisionInferenceInputs inputs,
      NDArray transitionPresentIndices,
      NDArray actionPresentIndices,
      int transitionOffset) {}
}

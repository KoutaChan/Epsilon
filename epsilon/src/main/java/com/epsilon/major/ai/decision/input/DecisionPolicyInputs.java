package com.epsilon.major.ai.decision.input;

import ai.djl.ndarray.NDArray;

/** 方策推論に共通の状態・行動入力を保持し、学習用の密な入力と推論用に絞った入力を区別する。 */
public interface DecisionPolicyInputs {

  NDArray stateCategories();

  NDArray actionCategories();

  NDArray actionRoutes();

  NDArray stateNumerics();

  NDArray boundaryContext();

  NDArray actionNumerics();

  DecisionBucket bucket();

  /** バッチ行数を返す。 */
  default int rowCount() {
    return Math.toIntExact(stateCategories().getShape().get(0));
  }

  /** 行動 IDのパディングから合法候補マスクを導出する。 */
  default NDArray legalActionMask() {
    return actionCategories()
        .get("...,{}", DecisionInputSchema.ActionInt.ID.ordinal())
        .neq(DecisionInputSchema.PAD_ID);
  }

  /** 局カテゴリ値テンソルから指定フィールドの {@code [batch]} ビューを返す。 */
  default NDArray roundCategory(DecisionInputSchema.RoundInt field) {
    return stateCategories().get(":,{}", field.ordinal());
  }
}

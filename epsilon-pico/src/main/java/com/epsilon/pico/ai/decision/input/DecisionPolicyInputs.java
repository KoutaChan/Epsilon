package com.epsilon.pico.ai.decision.input;

import ai.djl.ndarray.NDArray;

/** 方策が参照する状態・行動の入力。固定長の学習入力と、実在する遷移だけを詰めた推論入力を共通に扱う。 */
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

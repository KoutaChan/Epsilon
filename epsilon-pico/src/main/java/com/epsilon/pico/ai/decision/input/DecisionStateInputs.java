package com.epsilon.pico.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

/**
 * 行動候補を含まない、価値予測専用の状態入力。
 *
 * <p>カテゴリ値と数値の2つのテンソルが同じ行を表す。行動候補の容量に依存する入力を受け取らない型とし、価値だけを学習する際の不要な転送を防ぐ。
 *
 * @param stateCategories カテゴリ値状態テンソル
 * @param stateNumerics 数値状態テンソル
 * @param boundaryContext 価値専用の局境界コンテキストテンソル
 */
public record DecisionStateInputs(
    NDArray stateCategories, NDArray stateNumerics, NDArray boundaryContext) {

  /** 二テンソルが同じ行数と標準形式の状態幅を持つことを検証する。 */
  public DecisionStateInputs {
    Shape categories = stateCategories.getShape();
    Shape numerics = stateNumerics.getShape();
    Shape boundary = boundaryContext.getShape();
    if (categories.dimension() != 2
        || numerics.dimension() != 2
        || categories.get(0) != numerics.get(0)
        || categories.get(0) != boundary.get(0)
        || categories.get(1) != DecisionInputSchema.STATE_INT_COUNT
        || numerics.get(1) != DecisionInputSchema.STATE_FLOAT_COUNT
        || boundary.dimension() != 2
        || boundary.get(1) != DecisionBoundaryContext.INPUT_SIZE) {
      throw new IllegalArgumentException(
          "state inputs require [B,STATE_INT_COUNT] and [B,STATE_FLOAT_COUNT]: "
              + categories
              + " / "
              + numerics
              + " / "
              + boundary);
    }
  }

  /**
   * 価値のみバッチの行数を返す。
   *
   * @return 状態テンソルに共通する先頭軸長
   */
  public int rowCount() {
    return Math.toIntExact(stateCategories.getShape().get(0));
  }
}

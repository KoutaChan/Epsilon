package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

/**
 * 方策候補を持たない価値専用順伝播の入力。
 *
 * <p>二テンソルは標準形式の状態スキーマの同じ行を表し、形状はそれぞれ {@code [batch, STATE_INT_COUNT]} と {@code [batch,
 * STATE_FLOAT_COUNT]} である。行動容量区分に依存するテンソルを型として受け取れないため、価値のみ 事前学習が誤って方策入力を転送するのを防ぐ。
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

package com.epsilon.pico.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

/**
 * 価値関数だけを事前学習するための、最小限のデバイス側バッチ。
 *
 * <p>状態入力に効用の教師値と行の重みを添える。行動候補や探索前後の方策は保持しないため、合法手の数が増えても転送量は増えない。
 *
 * @param inputs 標準形式の状態入力
 * @param playerMemoryPresentIndices プレイヤーごとの履歴表現の存在トークンを指す行優先インデックス
 * @param valueTarget {@code [batch]} の効用教師値
 * @param sampleWeight {@code [batch]} の損失重み
 */
public record DecisionValueDeviceBatch(
    DecisionStateInputs inputs,
    NDArray playerMemoryPresentIndices,
    NDArray valueTarget,
    NDArray sampleWeight) {

  /** 教師値と重みが状態入力と同じ行数・所定形状を持つことを検証する。 */
  public DecisionValueDeviceBatch {
    long rows = inputs.rowCount();
    if (playerMemoryPresentIndices.getShape().dimension() != 1
        || !valueTarget.getShape().equals(new Shape(rows))
        || !sampleWeight.getShape().equals(new Shape(rows))) {
      throw new IllegalArgumentException(
          "value batch shape mismatch: "
              + playerMemoryPresentIndices.getShape()
              + " / "
              + valueTarget.getShape()
              + " / "
              + sampleWeight.getShape());
    }
  }
}

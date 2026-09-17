package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

/**
 * 価値のみ事前学習用の最小デバイス側バッチ。
 *
 * <p>効用教師値と行重みだけを状態に添え、行動、遷移、選択行動、探索後の/対局生成
 * 方策、方策モデル教師値は保持しない。これにより合法候補容量区分が大きくても価値実行単位のホストからデバイスへの転送量は増えない。
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

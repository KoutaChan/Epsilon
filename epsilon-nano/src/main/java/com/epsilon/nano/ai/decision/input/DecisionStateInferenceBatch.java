package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;

/**
 * 価値推論用の状態入力と、ホストで確定したプレイヤー履歴の参照位置を保持する。
 *
 * @param inputs 方策候補を含まない状態入力
 * @param playerMemoryPresentIndices バッチ内の存在トークンを指す行優先インデックス
 */
public record DecisionStateInferenceBatch(
    DecisionStateInputs inputs, NDArray playerMemoryPresentIndices) {}

package com.epsilon.pico.ai.decision.input;

import ai.djl.ndarray.NDArray;

/**
 * 価値予測用の状態入力と、ホスト側で確定したプレイヤー履歴のインデックス。
 *
 * @param inputs 方策候補を含まない状態入力
 * @param playerMemoryPresentIndices バッチ内の存在トークンを指す行優先インデックス
 */
public record DecisionStateInferenceBatch(
    DecisionStateInputs inputs, NDArray playerMemoryPresentIndices) {}

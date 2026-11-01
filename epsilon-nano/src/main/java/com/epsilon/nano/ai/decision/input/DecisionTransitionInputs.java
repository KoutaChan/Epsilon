package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;

/** 遷移表現の共通入力。推論では実在遷移だけを、学習では元の行動・遷移配置を保持する。 */
public record DecisionTransitionInputs(
    NDArray categories,
    NDArray numerics,
    NDArray tileCodes,
    NDArray waitTileIds,
    NDArray waitYakus,
    NDArray waitScores,
    boolean hostCompacted) {

  public static DecisionTransitionInputs dense(DecisionNetworkInputs inputs) {
    return new DecisionTransitionInputs(
        inputs.transitionCategories(),
        inputs.transitionNumerics(),
        inputs.transitionTiles(),
        inputs.waitTileIds(),
        inputs.waitYakus(),
        inputs.waitScores(),
        false);
  }

  public static DecisionTransitionInputs compact(DecisionInferenceInputs inputs) {
    return new DecisionTransitionInputs(
        inputs.transitionCategories(),
        inputs.transitionNumerics(),
        inputs.transitionTiles(),
        inputs.waitTileIds(),
        inputs.waitYakus(),
        inputs.waitScores(),
        true);
  }
}

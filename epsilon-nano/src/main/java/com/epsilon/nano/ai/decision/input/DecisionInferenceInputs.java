package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;

/**
 * 通常の即時実行による推論で使う入力テンソル。
 *
 * <p>状態と行動は固定長のバッチ形状を持つ。遷移に対応する6本のテンソルは、CPU 側で求めた有効位置の順に {@code [present, 1, 1, ...]}
 * の形状で格納する。学習用入力や保存データの形式には使用しない。
 */
public record DecisionInferenceInputs(
    NDArray stateCategories,
    NDArray actionCategories,
    NDArray actionRoutes,
    NDArray transitionCategories,
    NDArray transitionTiles,
    NDArray waitTileIds,
    NDArray waitYakus,
    NDArray stateNumerics,
    NDArray boundaryContext,
    NDArray actionNumerics,
    NDArray transitionNumerics,
    NDArray waitScores,
    DecisionBucket bucket)
    implements DecisionPolicyInputs {

  /** 基準テンソルのバッチ軸と有効要素のみの遷移テンソルの存在する軸がそれぞれ一致することを検証する。 */
  public DecisionInferenceInputs {
    long rows = stateCategories.getShape().get(0);
    long present = transitionCategories.getShape().get(0);
    if (rows < 1
        || actionCategories.getShape().get(0) != rows
        || actionRoutes.getShape().get(0) != rows
        || stateNumerics.getShape().get(0) != rows
        || boundaryContext.getShape().get(0) != rows
        || actionNumerics.getShape().get(0) != rows
        || transitionTiles.getShape().get(0) != present
        || waitTileIds.getShape().get(0) != present
        || waitYakus.getShape().get(0) != present
        || transitionNumerics.getShape().get(0) != present
        || waitScores.getShape().get(0) != present) {
      throw new IllegalArgumentException("Decision inference tensors have inconsistent axes");
    }
  }

  /** 二本の有効要素のみの推論連続バッファから全論理ビューを構築する。 */
  public static DecisionInferenceInputs bind(
      DecisionInferenceInputLayout layout, NDArray categoricalSlab, NDArray numericSlab) {
    NDArray[] views = new NDArray[DecisionInputLayout.tensorCount()];
    for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
      DecisionInferenceInputLayout.Region region = layout.region(tensor);
      NDArray slab =
          tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL ? categoricalSlab : numericSlab;
      views[tensor.ordinal()] =
          slice(
              slab,
              region.slabOffset(),
              Math.addExact(region.slabOffset(), region.elementCount()),
              region.shape());
    }
    return new DecisionInferenceInputs(
        views[DecisionInputLayout.Tensor.STATE_CATEGORIES.ordinal()],
        views[DecisionInputLayout.Tensor.ACTION_CATEGORIES.ordinal()],
        views[DecisionInputLayout.Tensor.ACTION_ROUTES.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_CATEGORIES.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_TILES.ordinal()],
        views[DecisionInputLayout.Tensor.WAIT_TILE_IDS.ordinal()],
        views[DecisionInputLayout.Tensor.WAIT_YAKUS.ordinal()],
        views[DecisionInputLayout.Tensor.STATE_NUMERICS.ordinal()],
        views[DecisionInputLayout.Tensor.BOUNDARY_CONTEXT.ordinal()],
        views[DecisionInputLayout.Tensor.ACTION_NUMERICS.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_NUMERICS.ordinal()],
        views[DecisionInputLayout.Tensor.WAIT_SCORES.ordinal()],
        layout.bucket());
  }

  /** ホストで有効な要素の集約された遷移行数を返す。 */
  public int transitionCount() {
    return Math.toIntExact(transitionCategories.getShape().get(0));
  }

  /** 管理元へ一括で所属させるための全論理的なテンソルを返す。 */
  public NDArray[] arrays() {
    return new NDArray[] {
      stateCategories,
      actionCategories,
      actionRoutes,
      transitionCategories,
      transitionTiles,
      waitTileIds,
      waitYakus,
      stateNumerics,
      boundaryContext,
      actionNumerics,
      transitionNumerics,
      waitScores
    };
  }

  private static NDArray slice(NDArray slab, int start, int end, ai.djl.ndarray.types.Shape shape) {
    if (start == end) {
      return slab.get(new NDIndex().addSliceDim(0, 0)).reshape(shape);
    }
    return slab.get(new NDIndex().addSliceDim(start, end)).reshape(shape);
  }
}

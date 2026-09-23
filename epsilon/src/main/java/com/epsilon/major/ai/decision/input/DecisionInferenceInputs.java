package com.epsilon.major.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;

/**
 * 通常の逐次実行による推論で使う、状態・行動と実在する遷移のテンソル。
 *
 * <p>状態・行動テンソルは密なバッチ形状、遷移五テンソルはホストの存在するインデックス順に並ぶ {@code [present, 1, 1, ...]} 形状である。標準形式の
 * NDListと学習チェックポイントスキーマには参加しない。
 */
public record DecisionInferenceInputs(
    NDArray stateCategories,
    NDArray actionCategories,
    NDArray actionRoutes,
    NDArray pointLedger100,
    NDArray actionWinFacts,
    NDArray transitionCategories,
    NDArray transitionTiles,
    NDArray waitTileIds,
    NDArray waitWinFacts,
    NDArray stateNumerics,
    NDArray boundaryContext,
    NDArray transitionNumerics,
    NDArray waitRows,
    NDArray winningActions,
    DecisionBucket bucket)
    implements DecisionPolicyInputs {

  /** 基準テンソルのバッチ軸とコンパクト遷移テンソルの存在する軸がそれぞれ一致することを検証する。 */
  public DecisionInferenceInputs {
    long rows = stateCategories.getShape().get(0);
    long present = transitionCategories.getShape().get(0);
    if (rows < 1
        || actionCategories.getShape().get(0) != rows
        || actionRoutes.getShape().get(0) != rows
        || pointLedger100.getShape().get(0) != rows
        || actionWinFacts.getShape().get(0) != rows
        || stateNumerics.getShape().get(0) != rows
        || boundaryContext.getShape().get(0) != rows
        || transitionTiles.getShape().get(0) != present
        || waitTileIds.getShape().get(0) != present
        || waitWinFacts.getShape().get(0) != present
        || transitionNumerics.getShape().get(0) != present) {
      throw new IllegalArgumentException("Decision inference tensors have inconsistent axes");
    }
  }

  /** 二本のコンパクト推論連続バッファから全論理ビューを構築する。 */
  public static DecisionInferenceInputs bind(
      DecisionInferenceInputLayout layout,
      NDArray categoricalSlab,
      NDArray numericSlab,
      NDArray waitRows,
      NDArray winningActions) {
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
        views[DecisionInputLayout.Tensor.POINT_LEDGER_100.ordinal()].toType(DataType.INT32, false),
        views[DecisionInputLayout.Tensor.ACTION_WIN_FACTS.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_CATEGORIES.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_TILES.ordinal()],
        views[DecisionInputLayout.Tensor.WAIT_TILE_IDS.ordinal()],
        views[DecisionInputLayout.Tensor.WAIT_WIN_FACTS.ordinal()],
        views[DecisionInputLayout.Tensor.STATE_NUMERICS.ordinal()],
        views[DecisionInputLayout.Tensor.BOUNDARY_CONTEXT.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_NUMERICS.ordinal()],
        waitRows,
        winningActions,
        layout.bucket());
  }

  /** ホストでコンパクト化された遷移行数を返す。 */
  public int transitionCount() {
    return Math.toIntExact(transitionCategories.getShape().get(0));
  }

  /** 管理元有効範囲へ一括関連付けするための全論理的なテンソルを返す。 */
  public NDArray[] arrays() {
    return new NDArray[] {
      stateCategories,
      actionCategories,
      actionRoutes,
      pointLedger100,
      actionWinFacts,
      transitionCategories,
      transitionTiles,
      waitTileIds,
      waitWinFacts,
      stateNumerics,
      boundaryContext,
      transitionNumerics,
      waitRows,
      winningActions
    };
  }

  private static NDArray slice(NDArray slab, int start, int end, ai.djl.ndarray.types.Shape shape) {
    if (start == end) {
      return slab.get(new NDIndex().addSliceDim(0, 0)).reshape(shape);
    }
    return slab.get(new NDIndex().addSliceDim(start, end)).reshape(shape);
  }
}

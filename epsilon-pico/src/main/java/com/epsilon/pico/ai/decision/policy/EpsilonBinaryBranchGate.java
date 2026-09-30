package com.epsilon.pico.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.pico.ai.model.EpsilonMahjongStateEncoder;

/**
 * 局面、基準の選択肢、選択する側の選択肢の3つの表現から、二択のロジットを計算する。
 *
 * <p>中間層の幅は最大64とし、正の出力は選択する側、負の出力は基準側を優先する。和了、九種九牌、鳴き、槓、リーチを同じ形式で比較し、見送り・継続・ダマにも実際の特徴表現を与える。
 */
public final class EpsilonBinaryBranchGate extends AbstractBlock {

  private final int hiddenSize;
  private final int middleSize;
  private final Linear hidden;
  private final Linear score;

  EpsilonBinaryBranchGate(int hiddenSize) {
    this.hiddenSize = hiddenSize;
    middleSize = Math.min(hiddenSize, 64);
    hidden = addChildBlock("hidden", Linear.builder().setUnits(middleSize).build());
    score = addChildBlock("score", Linear.builder().setUnits(1).build());
  }

  /** 重みを固定したモデルの推論実行計画へ結び付ける隠れ層射影を返す。 */
  public Linear hidden() {
    return hidden;
  }

  /** 重みを固定したモデルの推論実行計画へ結び付けるスカラースコア射影を返す。 */
  public Linear scoreHead() {
    return score;
  }

  /**
   * {@code [batch, hidden]} の三表現から選択分岐の二値ロジットを返す。
   *
   * <p>{@code selectedBranch - baselineBranch} は両入力と線形従属であり、直後のLinearへ明示的に渡しても
   * 表現力は増えない。差分に対する任意の重みは比較基準/選択したの重みへ厳密に吸収できるため、独立な三表現だけを結合する。
   */
  NDArray score(
      ParameterStore parameterStore,
      NDArray state,
      NDArray baselineBranch,
      NDArray selectedBranch,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray comparison = NDArrays.concat(new NDList(state, baselineBranch, selectedBranch), 1);
    NDArray hiddenValue =
        EpsilonMahjongStateEncoder.silu(
            hidden
                .forward(parameterStore, new NDList(comparison), training, runtimeParameters)
                .singletonOrThrow());
    return score
        .forward(parameterStore, new NDList(hiddenValue), training, runtimeParameters)
        .singletonOrThrow();
  }

  /** 状態だけが全候補で共通な比較を、候補軸へブロードキャストする前に状態射影して採点する。 */
  NDArray scoreBroadcastState(
      ParameterStore parameterStore,
      NDArray state,
      NDArray baselineBranch,
      NDArray selectedBranch,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray branchComparison = NDArrays.concat(new NDList(baselineBranch, selectedBranch), 2);
    NDArray hiddenValue =
        EpsilonMahjongStateEncoder.silu(
            EpsilonPartitionedLinear.apply(
                hidden, parameterStore, training, state, branchComparison));
    return score
        .forward(parameterStore, new NDList(hiddenValue), training, runtimeParameters)
        .singletonOrThrow();
  }

  /** 有効なバッチ行だけを採点し、無効な行を0とする密なスコアへ戻す。 */
  NDArray scoreRows(
      ParameterStore parameterStore,
      NDArray state,
      NDArray baselineBranch,
      NDArray selectedBranch,
      NDArray activeRows,
      long rowCount,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long activeCount = activeRows.getShape().size();
    if (activeCount == 0) {
      return state.getManager().zeros(new Shape(rowCount, 1), state.getDataType());
    }
    NDArray activeScores =
        score(
            parameterStore,
            EpsilonMaskedRows.gather(state.reshape(rowCount, hiddenSize), activeRows),
            EpsilonMaskedRows.gather(baselineBranch.reshape(rowCount, hiddenSize), activeRows),
            EpsilonMaskedRows.gather(selectedBranch.reshape(rowCount, hiddenSize), activeRows),
            training,
            runtimeParameters);
    return EpsilonMaskedRows.scatter(activeScores.reshape(activeCount, 1), activeRows, rowCount);
  }

  /** 有効な行動だけを採点し、{@code [batch, action]} の密なスコアへ戻す。 */
  NDArray scoreActions(
      ParameterStore parameterStore,
      NDArray state,
      NDArray baselineBranch,
      NDArray selectedBranch,
      NDArray activeActions,
      long rowCount,
      int actionCapacity,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long activeCount = activeActions.getShape().size();
    long denseActionCount = Math.multiplyExact(rowCount, actionCapacity);
    if (activeCount == 0) {
      return state.getManager().zeros(new Shape(rowCount, actionCapacity), state.getDataType());
    }
    NDArray activeRows = activeActions.floorDivide(actionCapacity).toType(DataType.INT32, false);
    NDArray activeScores =
        score(
            parameterStore,
            EpsilonMaskedRows.gather(state.reshape(rowCount, hiddenSize), activeRows),
            EpsilonMaskedRows.gather(
                baselineBranch.reshape(denseActionCount, hiddenSize), activeActions),
            EpsilonMaskedRows.gather(
                selectedBranch.reshape(denseActionCount, hiddenSize), activeActions),
            training,
            runtimeParameters);
    return EpsilonMaskedRows.scatter(
            activeScores.reshape(activeCount, 1), activeActions, denseActionCount)
        .reshape(rowCount, actionCapacity);
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    hidden.initialize(manager, dataType, new Shape(-1, hiddenSize * 3L));
    score.initialize(manager, dataType, new Shape(-1, middleSize));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    throw new UnsupportedOperationException("Use score with explicit branch embeddings");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, 1)};
  }
}

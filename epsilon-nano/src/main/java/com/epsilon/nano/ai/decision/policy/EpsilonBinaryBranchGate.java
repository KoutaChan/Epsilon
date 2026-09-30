package com.epsilon.nano.ai.decision.policy;

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
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;

/**
 * 局面、基準側の分岐、選択側の分岐の3つの表現から、二つの行動の分岐を比較する。
 *
 * <p>3つの表現を連結して隠れ層へ射影し、選択側と基準側の差を表すロジットを返す。両分岐の表現差を追加の入力として連結する必要はない。ロン、ツモ、九種九牌、鳴き、槓、リーチの判断で同じ計算形式を使う。
 */
public final class EpsilonBinaryBranchGate extends AbstractBlock {

  private final int hiddenSize;
  private final Linear hidden;
  private final Linear score;

  EpsilonBinaryBranchGate(int hiddenSize) {
    this.hiddenSize = hiddenSize;
    hidden = addChildBlock("hidden", Linear.builder().setUnits(hiddenSize).build());
    score = addChildBlock("score", Linear.builder().setUnits(1).build());
  }

  /** 凍結推論実行計画へ関連付ける隠れ層射影を返す。 */
  public Linear hidden() {
    return hidden;
  }

  /** 凍結推論実行計画へ関連付けるスカラースコア射影を返す。 */
  public Linear scoreHead() {
    return score;
  }

  /**
   * {@code [batch, hidden]} の三表現から選択分岐の二値ロジットを返す。
   *
   * <p>{@code selectedBranch - baselineBranch} は両入力値と線形従属であり、直後のLinearへ明示的に渡しても
   * 表現力は増えない。差分に対する任意の重みは基準側と選択側の重みに厳密に吸収できるため、独立な三表現だけを結合する。
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
    score.initialize(manager, dataType, new Shape(-1, hiddenSize));
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

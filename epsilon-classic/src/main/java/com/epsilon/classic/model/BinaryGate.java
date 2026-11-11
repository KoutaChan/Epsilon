package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;

/**
 * 二択のロジットを求めるブロック。局面・基準の側・選ぶ側の 3 つの表現を受け取り、ロジットを返す。正なら選ぶ側、負なら基準の側を勧める。
 * ロン・ツモ・九種九牌・鳴き・槓・立直の二択は、どれも同じ作りのブロックを使う。
 *
 * <p>Major のロンとツモでは、点数の計算で求めた和了の結果（{@link PointProjection#GATE_FEATURES}）を射影し、隠れ層で活性化関数を通す前の
 * 値に直接足す。この射影の重みは 0 で初期化する。
 */
final class BinaryGate extends AbstractBlock {

  private final int hidden;
  private final int directWidth;
  private final Linear hiddenProjection;
  private final Linear directFeatureProjection;
  private final Linear score;

  /**
   * @param hidden 受け取る表現 1 つの幅
   * @param directWidth 直接足す特徴の幅（使わなければ 0）
   */
  BinaryGate(int hidden, int directWidth) {
    this.hidden = hidden;
    this.directWidth = directWidth;
    hiddenProjection = addChildBlock("hidden", Linear.builder().setUnits(Widths.GATE).build());
    if (directWidth > 0) {
      directFeatureProjection =
          addChildBlock("directFeatureProjection", Linear.builder().setUnits(Widths.GATE).optBias(false).build());
      directFeatureProjection.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0f));
    } else {
      directFeatureProjection = null;
    }
    score = addChildBlock("score", Linear.builder().setUnits(1).build());
  }

  /**
   * @param state 局面 [行, 幅]
   * @param baseline 基準の側 [行, 幅]
   * @param selected 選ぶ側 [行, 幅]
   * @param direct 直接足す特徴 [行, 特徴]（使わない二択は null）
   * @return 選ぶ側のロジット [行, 1]
   */
  NDArray score(Pass pass, NDArray state, NDArray baseline, NDArray selected, NDArray direct) {
    NDArray preactivation =
        pass.linear(hiddenProjection, NDArrays.concat(new NDList(state, baseline, selected), 1));
    if (directFeatureProjection != null) {
      preactivation = preactivation.add(pass.linear(directFeatureProjection, direct));
    }
    return pass.linear(score, Pass.silu(preactivation));
  }

  /**
   * 合法手ごとに二択のロジットを求める。局面は合法手の軸へ広げてから並べる。
   *
   * @param state 局面 [行, 1, 幅]
   * @param baseline 基準の側 [行, 合法手, 幅]
   * @param selected 選ぶ側 [行, 合法手, 幅]
   * @return 選ぶ側のロジット [行, 合法手, 1]
   */
  NDArray scoreEach(Pass pass, NDArray state, NDArray baseline, NDArray selected) {
    NDArray states = state.broadcast(baseline.getShape());
    NDArray preactivation =
        pass.linear(hiddenProjection, NDArrays.concat(new NDList(states, baseline, selected), 2));
    return pass.linear(score, Pass.silu(preactivation));
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    hiddenProjection.initialize(manager, dataType, new Shape(-1, 3L * hidden));
    if (directFeatureProjection != null) {
      directFeatureProjection.initialize(manager, dataType, new Shape(-1, directWidth));
    }
    score.initialize(manager, dataType, new Shape(-1, Widths.GATE));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use score");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, 1)};
  }
}

package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * 点数の計算（{@link PointProjection}）で求めた和了の結果を、方策の文脈の幅へ射影する。Major だけが使う。射影はバイアスを持たず、
 * SiLU(0) = 0 なので、和了の無い所の出力はちょうど 0 になる。
 */
final class PointOutcomeEncoder extends AbstractBlock {

  private final int context;
  private final Linear projection;

  PointOutcomeEncoder(int context) {
    this.context = context;
    projection = addChildBlock("projection", Linear.builder().setUnits(context).optBias(false).build());
  }

  /**
   * @param features 和了の結果 [..., {@link PointProjection#FEATURES}]（計算の型）
   * @return [..., 文脈]
   */
  NDArray encode(Pass pass, NDArray features) {
    return Pass.silu(pass.linear(projection, features));
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    projection.initialize(manager, dataType, new Shape(-1, PointProjection.FEATURES));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use encode");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, context)};
  }
}

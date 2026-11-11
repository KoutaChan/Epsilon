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
 * 幅の違う表現を、それぞれ共通の幅へ射影して足し合わせる。全ての合法手に共通な局面の表現は、[行, 1, 幅] のまま射影してから広げて
 * 足す。バイアスは最初の射影だけが持つ。
 */
final class FeatureFusion extends AbstractBlock {

  private final int[] inputWidths;
  private final Linear[] projections;

  /**
   * @param output 共通の幅
   * @param inputWidths 表現ごとの幅（足す順）
   */
  FeatureFusion(int output, int... inputWidths) {
    this.inputWidths = inputWidths.clone();
    projections = new Linear[inputWidths.length];
    for (int index = 0; index < inputWidths.length; index++) {
      projections[index] =
          addChildBlock("component" + index, Linear.builder().setUnits(output).optBias(index == 0).build());
    }
  }

  /**
   * @param components 表現（コンストラクタに渡した幅と同じ順。幅より前の軸は、足し合わせる形へ広げられること）
   * @return 射影の和
   */
  NDArray fuse(Pass pass, NDArray... components) {
    if (components.length != projections.length) {
      throw new IllegalArgumentException(
          "Fusion takes " + projections.length + " components, not " + components.length);
    }
    NDArray sum = pass.linear(projections[0], components[0]);
    for (int index = 1; index < projections.length; index++) {
      sum = sum.add(pass.linear(projections[index], components[index]));
    }
    return sum;
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    for (int index = 0; index < projections.length; index++) {
      projections[index].initialize(manager, dataType, new Shape(-1, inputWidths[index]));
    }
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use fuse");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {inputShapes[0]};
  }
}

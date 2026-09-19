package com.epsilon.major.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.major.ai.model.EpsilonMahjongStateEncoder;

/** 固定計算した即時和了・将来待ちの結果を、方策用の共有文脈へ変換する。 */
public final class EpsilonPointOutcomeEncoder extends AbstractBlock {

  private final int outputWidth;
  private final Linear projection;

  public EpsilonPointOutcomeEncoder(int outputWidth) {
    this.outputWidth = outputWidth;
    projection =
        addChildBlock("projection", Linear.builder().setUnits(outputWidth).optBias(false).build());
  }

  NDArray encode(
      ParameterStore parameterStore,
      NDArray pointFeatures,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return EpsilonMahjongStateEncoder.silu(
        projection
            .forward(parameterStore, new NDList(pointFeatures), training, runtimeParameters)
            .singletonOrThrow());
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    projection.initialize(manager, dataType, new Shape(-1, EpsilonPointProjection.FEATURE_WIDTH));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return new NDList(
        encode(parameterStore, inputs.singletonOrThrow(), training, runtimeParameters));
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    Shape input = inputShapes[0];
    long[] output = input.getShape();
    output[output.length - 1] = outputWidth;
    return new Shape[] {new Shape(output)};
  }
}

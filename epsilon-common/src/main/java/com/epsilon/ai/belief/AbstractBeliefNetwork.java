package com.epsilon.ai.belief;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.core.Tile;

/** 系列固有の局面エンコーダーに、他家の手牌・受け入れ牌・シャンテン数・テンパイ確率を予測する共通の出力層を接続する。受け入れ牌は、テンパイ時には待ち牌を表す。 */
public abstract class AbstractBeliefNetwork extends AbstractBlock {

  private static final int TILE_OUTPUT_CHANNELS = EpsilonBeliefLayout.OPPONENT_COUNT * 2;
  private static final int TILE_OUTPUT_SIZE = TILE_OUTPUT_CHANNELS * Tile.NUM_TILE_TYPES;

  private final int hiddenSize;
  private final Linear tileHidden;
  private final LayerNorm tileNorm;
  private final Linear tileOutput;
  private final Linear scalarHidden;
  private final Linear scalarOutput;

  /** エンコーダーと特徴の集約を先に登録し、保存パラメーターの名前と順序を維持する。 */
  protected AbstractBeliefNetwork(int hiddenSize, Block stateEncoder, Block stateReadout) {
    this.hiddenSize = hiddenSize;
    addChildBlock("stateEncoder", stateEncoder);
    addChildBlock("stateReadout", stateReadout);
    tileHidden = addChildBlock("tileHidden", Linear.builder().setUnits(hiddenSize).build());
    tileNorm = addChildBlock("tileNorm", LayerNorm.builder().axis(2).build());
    tileOutput =
        addChildBlock("tileOutput", Linear.builder().setUnits(TILE_OUTPUT_CHANNELS).build());
    scalarHidden = addChildBlock("scalarHidden", Linear.builder().setUnits(hiddenSize).build());
    scalarOutput =
        addChildBlock(
            "scalarOutput", Linear.builder().setUnits(EpsilonBeliefLayout.SCALAR_SIZE).build());
  }

  public final int hiddenSize() {
    return hiddenSize;
  }

  /** 系列側でエンコーダーと特徴の集約を初期化してから共通出力層を初期化する。 */
  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    tileHidden.initialize(manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize * 2L));
    tileNorm.initialize(manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize));
    tileOutput.initialize(manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize));
    scalarHidden.initialize(manager, dataType, new Shape(-1, hiddenSize));
    scalarOutput.initialize(manager, dataType, new Shape(-1, hiddenSize));
  }

  /** 系列固有の状態テンソルから、共通の配列配置で予測値を返す。 */
  public abstract NDArray forwardBelief(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      boolean training,
      PairList<String, Object> runtimeParameters);

  /** エンコーダーの牌表現と特徴の集約結果を借用し、共通出力層を計算する。 */
  protected final NDArray projectBelief(
      ParameterStore parameterStore,
      NDArray tileEmbeddings,
      NDArray stateEmbedding,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long rowCount = stateEmbedding.getShape().get(0);
    NDArray stateContext =
        stateEmbedding
            .reshape(rowCount, 1, hiddenSize)
            .broadcast(rowCount, Tile.NUM_TILE_TYPES, hiddenSize);
    NDArray tileStateEmbeddings =
        silu(
            tileHidden
                .forward(
                    parameterStore,
                    new NDList(tileEmbeddings.concat(stateContext, 2)),
                    training,
                    runtimeParameters)
                .singletonOrThrow());
    tileStateEmbeddings =
        tileNorm
            .forward(parameterStore, new NDList(tileStateEmbeddings), training, runtimeParameters)
            .singletonOrThrow();
    NDArray tileLogits =
        tileOutput
            .forward(parameterStore, new NDList(tileStateEmbeddings), training, runtimeParameters)
            .singletonOrThrow()
            .swapAxes(1, 2)
            .reshape(rowCount, TILE_OUTPUT_SIZE);
    NDArray scalarStateEmbedding =
        silu(
            scalarHidden
                .forward(parameterStore, new NDList(stateEmbedding), training, runtimeParameters)
                .singletonOrThrow());
    NDArray scalarLogits =
        scalarOutput
            .forward(parameterStore, new NDList(scalarStateEmbedding), training, runtimeParameters)
            .singletonOrThrow();
    return tileLogits.concat(scalarLogits, 1);
  }

  private static NDArray silu(NDArray input) {
    return Activation.swish(input, 1.0f);
  }

  @Override
  protected final NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (inputs.size() != 2) {
      throw new IllegalArgumentException(
          "Belief network requires stateCategories and stateNumerics, got " + inputs.size());
    }
    return new NDList(
        forwardBelief(parameterStore, inputs.get(0), inputs.get(1), training, runtimeParameters));
  }

  @Override
  public final Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(inputShapes[0].get(0), EpsilonBeliefLayout.OUTPUT_SIZE)};
  }
}

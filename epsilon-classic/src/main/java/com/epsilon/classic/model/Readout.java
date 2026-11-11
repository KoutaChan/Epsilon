package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * 状態のエンコーダーの出力を集めて、方策か価値に使う局面の表現 [行, 幅] を 1 つ作る。方策と価値はそれぞれ別のインスタンスを使い、同じ出力を
 * 別のクエリと射影で読む。
 *
 * <p>クエリの初期値は、局の表現と、実在するトークンの平均から作る。1 つのクエリで全てのトークンを読み（実在しないトークンはキーから外す）、
 * 読んだ結果をクエリの初期値に足して正規化する。FFN の残差を足した後に、もう一度正規化する。Attention の幅だけを絞り、残差の幅は
 * 表現の幅のままにする。
 */
final class Readout extends AbstractBlock {

  private final int hidden;
  private final int attentionWidth;
  private final int headWidth;
  private final int feedForwardWidth;
  private final Linear querySeedProjection;
  private final LayerNorm queryLayerNorm;
  private final Linear queryProjection;
  private final Linear keyValueProjection;
  private final Linear contextProjection;
  private final LayerNorm feedForwardInputLayerNorm;
  private final Linear feedForwardExpansion;
  private final Linear feedForwardProjection;
  private final LayerNorm outputLayerNorm;

  /**
   * @param feedForwardWidth FFN の幅（方策と価値で違う）
   */
  Readout(Widths widths, int feedForwardWidth) {
    hidden = widths.hidden();
    attentionWidth = widths.readoutAttention();
    headWidth = attentionWidth / Widths.HEADS;
    this.feedForwardWidth = feedForwardWidth;
    querySeedProjection = addChildBlock("querySeedProjection", Linear.builder().setUnits(hidden).build());
    queryLayerNorm = addChildBlock("queryLayerNorm", LayerNorm.builder().axis(1).build());
    queryProjection = addChildBlock("queryProjection", Linear.builder().setUnits(attentionWidth).build());
    keyValueProjection =
        addChildBlock("keyValueProjection", Linear.builder().setUnits(2L * attentionWidth).optBias(false).build());
    contextProjection = addChildBlock("contextProjection", Linear.builder().setUnits(hidden).build());
    feedForwardInputLayerNorm = addChildBlock("feedForwardInputLayerNorm", LayerNorm.builder().axis(1).build());
    feedForwardExpansion =
        addChildBlock("feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
    feedForwardProjection = addChildBlock("feedForwardProjection", Linear.builder().setUnits(hidden).build());
    outputLayerNorm = addChildBlock("outputLayerNorm", LayerNorm.builder().axis(1).build());
  }

  /**
   * @param memory 状態のエンコーダーの出力（価値では勾配を止めたもの）
   * @return 局面の表現 [行, 幅]
   */
  NDArray read(Pass pass, Memory memory) {
    try (ArrayScope scope = new ArrayScope()) {
      NDArray entities = memory.entities();
      NDArray mask = memory.entityMask();
      long rows = entities.getShape().get(0);
      NDArray mean = entities.sum(new int[] {1}).div(mask.sum(new int[] {1}).reshape(rows, 1).maximum(1f));
      NDArray seed = pass.linear(querySeedProjection, memory.round().concat(mean, 1));
      NDArray queries = pass.linear(queryProjection, seed).reshape(rows, Widths.HEADS, 1, headWidth);
      NDList keyValues = pass.linear(keyValueProjection, entities).split(2, 2);
      long tokens = entities.getShape().get(1);
      NDArray keys = keyValues.get(0).reshape(rows, tokens, Widths.HEADS, headWidth).swapAxes(1, 2);
      NDArray values = keyValues.get(1).reshape(rows, tokens, Widths.HEADS, headWidth).swapAxes(1, 2);
      NDArray keyMask = mask.reshape(rows, 1, 1, tokens).log();
      NDArray context = Pass.attention(queries, keys, values, keyMask).reshape(rows, attentionWidth);
      NDArray state = pass.norm(queryLayerNorm, seed.add(pass.linear(contextProjection, context)));
      NDArray expanded = Pass.silu(pass.linear(feedForwardExpansion, pass.norm(feedForwardInputLayerNorm, state)));
      return scope.keep(pass.norm(outputLayerNorm, state.add(pass.linear(feedForwardProjection, expanded))));
    }
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape state = new Shape(-1, hidden);
    querySeedProjection.initialize(manager, dataType, new Shape(-1, 2L * hidden));
    queryLayerNorm.initialize(manager, dataType, state);
    queryProjection.initialize(manager, dataType, state);
    keyValueProjection.initialize(manager, dataType, new Shape(-1, StateEncoder.ENTITIES, hidden));
    contextProjection.initialize(manager, dataType, new Shape(-1, attentionWidth));
    feedForwardInputLayerNorm.initialize(manager, dataType, state);
    feedForwardExpansion.initialize(manager, dataType, state);
    feedForwardProjection.initialize(manager, dataType, new Shape(-1, feedForwardWidth));
    outputLayerNorm.initialize(manager, dataType, state);
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use read");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, hidden)};
  }
}

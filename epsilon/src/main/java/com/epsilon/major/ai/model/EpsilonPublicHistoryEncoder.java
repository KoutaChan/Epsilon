package com.epsilon.major.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.major.ai.decision.input.DecisionPublicHistory;

/** 公開イベントの相対時系列を読み、4席の要約だけを残差更新する。 */
final class EpsilonPublicHistoryEncoder extends AbstractBlock {
  private static final int HEADS = EpsilonMahjongStateEncoder.ATTENTION_HEADS;
  private final int hiddenSize;
  private final int attentionWidth;
  private final int feedForwardWidth;
  private final LayerNorm queryNorm;
  private final Linear queryProjection;
  private final IdEmbedding relationTable;
  private final Linear contextProjection;
  private final LayerNorm feedForwardNorm;
  private final Linear feedForwardExpansion;
  private final Linear feedForwardProjection;

  EpsilonPublicHistoryEncoder(int hiddenSize) {
    this.hiddenSize = hiddenSize;
    attentionWidth = Math.min(hiddenSize, EpsilonDecisionArchitecture.POLICY_CONTEXT_WIDTH);
    feedForwardWidth =
        Math.min(hiddenSize * 2, EpsilonDecisionArchitecture.PUBLIC_HISTORY_FEED_FORWARD_WIDTH);
    queryNorm = addChildBlock("queryNorm", LayerNorm.builder().axis(2).build());
    queryProjection =
        addChildBlock(
            "queryProjection", Linear.builder().setUnits(attentionWidth).optBias(false).build());
    relationTable =
        addChildBlock(
            "relationTable",
            new IdEmbedding.Builder()
                .setDictionarySize(DecisionPublicHistory.TABLE_SIZE)
                .setEmbeddingSize(HEADS + attentionWidth)
                .build());
    contextProjection = addChildBlock("contextProjection", zeroResidual(hiddenSize));
    feedForwardNorm = addChildBlock("feedForwardNorm", LayerNorm.builder().axis(2).build());
    feedForwardExpansion =
        addChildBlock("feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
    feedForwardProjection = addChildBlock("feedForwardProjection", zeroResidual(hiddenSize));
  }

  NDArray encode(
      ParameterStore store,
      NDArray summaries,
      NDArray publicKeyValues,
      NDArray publicMask,
      NDArray relationCodes,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray queries =
        apply(
            queryProjection,
            store,
            apply(queryNorm, store, summaries, training, runtimeParameters),
            training,
            runtimeParameters);
    NDArray context =
        NDArrays.packedRelationScaledDotProductAttention(
            queries,
            publicKeyValues,
            publicMask,
            relationCodes,
            relationTable.getValue(store, summaries.getDevice(), training),
            HEADS,
            DecisionPublicHistory.ENTRIES_PER_SEGMENT,
            (float) (1.0 / Math.sqrt(attentionWidth / HEADS)));
    NDArray attended =
        summaries.add(apply(contextProjection, store, context, training, runtimeParameters));
    NDArray features = apply(feedForwardNorm, store, attended, training, runtimeParameters);
    features = apply(feedForwardExpansion, store, features, training, runtimeParameters);
    return attended.add(
        apply(
            feedForwardProjection,
            store,
            EpsilonMahjongStateEncoder.silu(features),
            training,
            runtimeParameters));
  }

  private static Linear zeroResidual(int width) {
    Linear block = Linear.builder().setUnits(width).build();
    block.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0));
    block.getDirectParameters().get("bias").setInitializer(new ConstantInitializer(0));
    return block;
  }

  private static NDArray apply(
      AbstractBlock block,
      ParameterStore store,
      NDArray input,
      boolean training,
      PairList<String, Object> parameters) {
    return block.forward(store, new NDList(input), training, parameters).singletonOrThrow();
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType type, Shape... shapes) {
    Shape summaries = new Shape(-1, DecisionPublicHistory.QUERY_COUNT, hiddenSize);
    queryNorm.initialize(manager, type, summaries);
    queryProjection.initialize(manager, type, summaries);
    relationTable.initialize(manager, type, new Shape(-1));
    contextProjection.initialize(
        manager, type, new Shape(-1, DecisionPublicHistory.QUERY_COUNT, attentionWidth));
    feedForwardNorm.initialize(manager, type, summaries);
    feedForwardExpansion.initialize(manager, type, summaries);
    feedForwardProjection.initialize(
        manager, type, new Shape(-1, DecisionPublicHistory.QUERY_COUNT, feedForwardWidth));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore store, NDList inputs, boolean training, PairList<String, Object> parameters) {
    return new NDList(
        encode(
            store,
            inputs.get(0),
            inputs.get(1),
            inputs.get(2),
            inputs.get(3),
            training,
            parameters));
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {inputShapes[0]};
  }
}

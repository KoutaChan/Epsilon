package com.epsilon.nano.ai.model;

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
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonResidualLayerNorm;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.model.fusion.EpsilonStrategicContextFusionExecution;
import java.util.ArrayList;
import java.util.List;

/** 局条件・自家手牌・4席の要約だけを相互作用させる小規模な戦略文脈エンコーダー。 */
final class EpsilonStrategicContextEncoder extends AbstractBlock {
  static final int STRATEGIC_TOKEN_COUNT = 2 + GameState.NUM_PLAYERS;
  private static final int BLOCK_COUNT = 2;
  private static final int ATTENTION_HEADS = EpsilonMahjongStateEncoder.ATTENTION_HEADS;
  private static final int MAXIMUM_HAND_ATTENTION_WIDTH = 64;
  private static final int MAXIMUM_ATTENTION_WIDTH = 128;
  private static final int MAXIMUM_FEED_FORWARD_WIDTH = 512;

  private final int hiddenSize;
  private final int handAttentionWidth;
  private final int handAttentionHeadSize;
  private final Linear handQueryProjection;
  private final Linear handKeyValueProjection;
  private final Linear handContextProjection;
  private final LayerNorm handSummaryLayerNorm;
  private final IdEmbedding tokenRoleEmbedding;
  private final StrategicBlock[] blocks;

  EpsilonStrategicContextEncoder(int hiddenSize) {
    if (hiddenSize <= 0 || hiddenSize % ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be positive and divisible by " + ATTENTION_HEADS);
    }
    this.hiddenSize = hiddenSize;
    handAttentionWidth = Math.min(hiddenSize, MAXIMUM_HAND_ATTENTION_WIDTH);
    handAttentionHeadSize = handAttentionWidth / ATTENTION_HEADS;
    handQueryProjection =
        addChildBlock("handQueryProjection", Linear.builder().setUnits(handAttentionWidth).build());
    handKeyValueProjection =
        addChildBlock(
            "handKeyValueProjection",
            Linear.builder().setUnits(handAttentionWidth * 2L).optBias(false).build());
    handContextProjection =
        addChildBlock("handContextProjection", Linear.builder().setUnits(hiddenSize).build());
    handSummaryLayerNorm =
        addChildBlock("handSummaryLayerNorm", LayerNorm.builder().axis(1).build());
    tokenRoleEmbedding =
        addChildBlock(
            "tokenRoleEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(STRATEGIC_TOKEN_COUNT)
                .setEmbeddingSize(hiddenSize)
                .build());
    blocks = new StrategicBlock[BLOCK_COUNT];
    for (int index = 0; index < blocks.length; index++) {
      blocks[index] =
          addChildBlock("block" + index, new StrategicBlock(hiddenSize, ATTENTION_HEADS));
    }
  }

  /** 34牌種を自家手牌トークンへ集約し、局・手牌・4席の6 トークンを2層で文脈化する。 */
  StrategicContext encode(
      ParameterStore parameterStore,
      NDArray roundEmbedding,
      NDArray tileEmbeddings,
      NDArray playerSummaries,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return encode(
        parameterStore,
        roundEmbedding,
        tileEmbeddings,
        tileEmbeddings,
        playerSummaries,
        training,
        null,
        runtimeParameters);
  }

  /** 34牌種と4席の要約を作り、推論時だけ永続戦略文脈層の積み重ね実行計画を利用する。 */
  StrategicContext encode(
      ParameterStore parameterStore,
      NDArray roundEmbedding,
      NDArray tileEmbeddings,
      NDArray tileProjectionEmbeddings,
      NDArray playerSummaries,
      boolean training,
      EpsilonStrategicContextFusionExecution.Forward inferenceForward,
      PairList<String, Object> runtimeParameters) {
    validateInputs(roundEmbedding, tileEmbeddings, tileProjectionEmbeddings, playerSummaries);
    long rowCount = roundEmbedding.getShape().get(0);
    NDArray handSummary =
        summarizeHand(
            parameterStore,
            roundEmbedding,
            playerSummaries.get(":,0,:"),
            tileEmbeddings,
            tileProjectionEmbeddings,
            training,
            runtimeParameters);
    NDArray strategicTokens =
        NDArrays.concat(
            new NDList(roundEmbedding.expandDims(1), handSummary.expandDims(1), playerSummaries),
            1);
    NDArray roleEmbeddings =
        parameterStore
            .getValue(
                tokenRoleEmbedding.getParameters().get("embedding"),
                strategicTokens.getDevice(),
                training)
            .reshape(1, STRATEGIC_TOKEN_COUNT, hiddenSize);
    strategicTokens = strategicTokens.add(roleEmbeddings);
    if (inferenceForward != null) {
      if (training) {
        throw new IllegalArgumentException("strategic inference plan cannot train");
      }
      strategicTokens = inferenceForward.encode(strategicTokens);
    } else {
      for (StrategicBlock block : blocks) {
        strategicTokens =
            block.encode(parameterStore, strategicTokens, training, runtimeParameters);
      }
    }
    NDList strategicSections = strategicTokens.split(new long[] {1, 2}, 1);
    return new StrategicContext(
        strategicSections.get(0).reshape(rowCount, hiddenSize),
        strategicSections.get(2).reshape(rowCount, GameState.NUM_PLAYERS, hiddenSize));
  }

  /**
   * Fusion 実行処理へ渡す凍結済みパラメーター参照を組み立てる。
   *
   * <p>ブロック自体は公開せず、パラメーターを保持するNDArray 参照だけを名前付きレコードへ写す。重みを複製しないため、 返却値の寿命は指定管理元とパラメーター保存先に従う。
   */
  EpsilonStrategicContextFusionExecution.Parameters fusionParameters(
      ParameterStore parameterStore, NDManager manager) {
    List<EpsilonStrategicContextFusionExecution.BlockParameters> blockParameters =
        new ArrayList<>(blocks.length);
    for (StrategicBlock block : blocks) {
      blockParameters.add(
          new EpsilonStrategicContextFusionExecution.BlockParameters(
              value(parameterStore, block.attentionInputLayerNorm, "gamma", manager),
              value(parameterStore, block.attentionInputLayerNorm, "beta", manager),
              value(parameterStore, block.queryKeyValueProjection, "weight", manager),
              value(parameterStore, block.attentionOutputProjection, "weight", manager),
              value(parameterStore, block.attentionOutputProjection, "bias", manager),
              value(parameterStore, block.feedForwardInputLayerNorm, "gamma", manager),
              value(parameterStore, block.feedForwardInputLayerNorm, "beta", manager),
              value(parameterStore, block.feedForwardExpansion, "weight", manager),
              value(parameterStore, block.feedForwardExpansion, "bias", manager),
              value(parameterStore, block.feedForwardProjection, "weight", manager),
              value(parameterStore, block.feedForwardProjection, "bias", manager),
              value(parameterStore, block.outputLayerNorm, "gamma", manager),
              value(parameterStore, block.outputLayerNorm, "beta", manager)));
    }
    return new EpsilonStrategicContextFusionExecution.Parameters(blockParameters);
  }

  private static NDArray value(
      ParameterStore parameterStore, AbstractBlock block, String name, NDManager manager) {
    return parameterStore.getValue(block.getParameters().get(name), manager.getDevice(), false);
  }

  private NDArray summarizeHand(
      ParameterStore parameterStore,
      NDArray roundEmbedding,
      NDArray selfPlayerSummary,
      NDArray tileEmbeddings,
      NDArray tileProjectionEmbeddings,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long rowCount = roundEmbedding.getShape().get(0);
    NDArray querySeed = roundEmbedding.add(selfPlayerSummary).mul((float) (1.0 / Math.sqrt(2.0)));
    NDArray queries =
        applyLinear(handQueryProjection, parameterStore, querySeed, training, runtimeParameters)
            .reshape(rowCount, ATTENTION_HEADS, 1, handAttentionHeadSize);
    NDArray keyValues =
        applyLinear(
            handKeyValueProjection,
            parameterStore,
            tileProjectionEmbeddings,
            training,
            runtimeParameters);
    NDList keyValueSections = keyValues.split(2, 2);
    NDArray keys =
        keyValueSections
            .get(0)
            .reshape(rowCount, Tile.NUM_TILE_TYPES, ATTENTION_HEADS, handAttentionHeadSize)
            .swapAxes(1, 2);
    NDArray values =
        keyValueSections
            .get(1)
            .reshape(rowCount, Tile.NUM_TILE_TYPES, ATTENTION_HEADS, handAttentionHeadSize)
            .swapAxes(1, 2);
    NDArray context =
        queries
            .getNDArrayInternal()
            .scaledDotProductAttention(keys, values, null, 0.0, false)
            .reshape(rowCount, handAttentionWidth);
    NDArray tileMean = tileEmbeddings.mean(new int[] {1});
    NDArray projectedContext =
        applyLinear(handContextProjection, parameterStore, context, training, runtimeParameters);
    if (!training) {
      return addToOwnedResidualAndNormalize(
          parameterStore, tileMean, projectedContext, handSummaryLayerNorm);
    }
    return EpsilonResidualLayerNorm.applyTraining(
            parameterStore, tileMean, projectedContext, handSummaryLayerNorm)
        .get(0);
  }

  private void validateInputs(
      NDArray roundEmbedding,
      NDArray tileEmbeddings,
      NDArray tileProjectionEmbeddings,
      NDArray playerSummaries) {
    long rows = roundEmbedding.getShape().get(0);
    Shape expectedRound = new Shape(rows, hiddenSize);
    Shape expectedTiles = new Shape(rows, Tile.NUM_TILE_TYPES, hiddenSize);
    Shape expectedPlayers = new Shape(rows, GameState.NUM_PLAYERS, hiddenSize);
    if (!roundEmbedding.getShape().equals(expectedRound)
        || !tileEmbeddings.getShape().equals(expectedTiles)
        || !tileProjectionEmbeddings.getShape().equals(expectedTiles)
        || !playerSummaries.getShape().equals(expectedPlayers)) {
      throw new IllegalArgumentException(
          "strategic context shape mismatch: "
              + roundEmbedding.getShape()
              + " / "
              + tileEmbeddings.getShape()
              + " / "
              + tileProjectionEmbeddings.getShape()
              + " / "
              + playerSummaries.getShape());
    }
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    handQueryProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    handKeyValueProjection.initialize(
        manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize));
    handContextProjection.initialize(manager, dataType, new Shape(-1, handAttentionWidth));
    handSummaryLayerNorm.initialize(manager, dataType, new Shape(-1, hiddenSize));
    tokenRoleEmbedding.initialize(manager, dataType, new Shape(STRATEGIC_TOKEN_COUNT));
    Shape strategicShape = new Shape(-1, STRATEGIC_TOKEN_COUNT, hiddenSize);
    for (StrategicBlock block : blocks) {
      block.initialize(manager, dataType, strategicShape);
    }
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    throw new UnsupportedOperationException("Use encode with typed strategic inputs");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {
      new Shape(-1, hiddenSize), new Shape(-1, GameState.NUM_PLAYERS, hiddenSize)
    };
  }

  private static NDArray applyLinear(
      Linear block,
      ParameterStore parameterStore,
      NDArray input,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return block
        .forward(parameterStore, new NDList(input), training, runtimeParameters)
        .singletonOrThrow();
  }

  private static NDArray addToOwnedResidualAndNormalize(
      ParameterStore parameterStore, NDArray residual, NDArray update, LayerNorm layerNorm) {
    NDArray gamma =
        parameterStore.getValue(
            layerNorm.getParameters().get("gamma"), residual.getDevice(), false);
    NDArray beta =
        parameterStore.getValue(layerNorm.getParameters().get("beta"), residual.getDevice(), false);
    return NDArrays.addToOwnedResidualAndLayerNorm(residual, update, gamma, beta, 1.0e-5f);
  }

  record StrategicContext(NDArray roundEmbedding, NDArray playerSummaries) {}

  /** 6 トークンだけを対象に、128幅注意機構と512幅FFNで残差更新するブロック。 */
  private static final class StrategicBlock extends AbstractBlock {

    private final int hiddenSize;
    private final int attentionHeads;
    private final int attentionWidth;
    private final int attentionHeadSize;
    private final int feedForwardWidth;
    private final LayerNorm attentionInputLayerNorm;
    private final Linear queryKeyValueProjection;
    private final Linear attentionOutputProjection;
    private final LayerNorm feedForwardInputLayerNorm;
    private final Linear feedForwardExpansion;
    private final Linear feedForwardProjection;
    private final LayerNorm outputLayerNorm;

    StrategicBlock(int hiddenSize, int attentionHeads) {
      this.hiddenSize = hiddenSize;
      this.attentionHeads = attentionHeads;
      attentionWidth = Math.min(hiddenSize, MAXIMUM_ATTENTION_WIDTH);
      attentionHeadSize = attentionWidth / attentionHeads;
      feedForwardWidth = Math.min(hiddenSize * 2, MAXIMUM_FEED_FORWARD_WIDTH);
      attentionInputLayerNorm =
          addChildBlock("attentionInputLayerNorm", LayerNorm.builder().axis(2).build());
      queryKeyValueProjection =
          addChildBlock(
              "queryKeyValueProjection",
              Linear.builder().setUnits(attentionWidth * 3L).optBias(false).build());
      attentionOutputProjection =
          addChildBlock("attentionOutputProjection", Linear.builder().setUnits(hiddenSize).build());
      feedForwardInputLayerNorm =
          addChildBlock("feedForwardInputLayerNorm", LayerNorm.builder().axis(2).build());
      feedForwardExpansion =
          addChildBlock(
              "feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
      feedForwardProjection =
          addChildBlock("feedForwardProjection", Linear.builder().setUnits(hiddenSize).build());
      outputLayerNorm = addChildBlock("outputLayerNorm", LayerNorm.builder().axis(2).build());
    }

    NDArray encode(
        ParameterStore parameterStore,
        NDArray tokens,
        boolean training,
        PairList<String, Object> runtimeParameters) {
      long rowCount = tokens.getShape().get(0);
      NDArray normalized =
          attentionInputLayerNorm
              .forward(parameterStore, new NDList(tokens), training, runtimeParameters)
              .singletonOrThrow();
      NDArray queryKeyValues =
          applyLinear(
              queryKeyValueProjection, parameterStore, normalized, training, runtimeParameters);
      NDList queryKeyValueSections = queryKeyValues.split(3, 2);
      NDArray queries = attentionHeads(queryKeyValueSections.get(0), rowCount);
      NDArray keys = attentionHeads(queryKeyValueSections.get(1), rowCount);
      NDArray values = attentionHeads(queryKeyValueSections.get(2), rowCount);
      NDArray context =
          queries
              .getNDArrayInternal()
              .scaledDotProductAttention(keys, values, null, 0.0, false)
              .swapAxes(1, 2)
              .reshape(rowCount, STRATEGIC_TOKEN_COUNT, attentionWidth);
      NDArray projectedAttention =
          applyLinear(
              attentionOutputProjection, parameterStore, context, training, runtimeParameters);
      NDArray residual;
      if (training) {
        NDList attentionOutput =
            EpsilonResidualLayerNorm.applyTraining(
                parameterStore, tokens, projectedAttention, feedForwardInputLayerNorm);
        normalized = attentionOutput.get(0);
        residual = attentionOutput.get(1);
      } else {
        residual = tokens;
        normalized =
            addToOwnedResidualAndNormalize(
                parameterStore, residual, projectedAttention, feedForwardInputLayerNorm);
      }
      NDArray expanded =
          EpsilonMahjongStateEncoder.silu(
              applyLinear(
                  feedForwardExpansion, parameterStore, normalized, training, runtimeParameters));
      NDArray projectedFeedForward =
          applyLinear(feedForwardProjection, parameterStore, expanded, training, runtimeParameters);
      if (training) {
        return EpsilonResidualLayerNorm.applyTraining(
                parameterStore, residual, projectedFeedForward, outputLayerNorm)
            .get(0);
      }
      return addToOwnedResidualAndNormalize(
          parameterStore, residual, projectedFeedForward, outputLayerNorm);
    }

    private NDArray attentionHeads(NDArray values, long rowCount) {
      return values
          .reshape(rowCount, STRATEGIC_TOKEN_COUNT, attentionHeads, attentionHeadSize)
          .swapAxes(1, 2);
    }

    @Override
    protected void initializeChildBlocks(
        NDManager manager, DataType dataType, Shape... inputShapes) {
      Shape tokenShape = new Shape(-1, STRATEGIC_TOKEN_COUNT, hiddenSize);
      attentionInputLayerNorm.initialize(manager, dataType, tokenShape);
      queryKeyValueProjection.initialize(manager, dataType, tokenShape);
      attentionOutputProjection.initialize(
          manager, dataType, new Shape(-1, STRATEGIC_TOKEN_COUNT, attentionWidth));
      feedForwardInputLayerNorm.initialize(manager, dataType, tokenShape);
      feedForwardExpansion.initialize(manager, dataType, tokenShape);
      feedForwardProjection.initialize(
          manager, dataType, new Shape(-1, STRATEGIC_TOKEN_COUNT, feedForwardWidth));
      outputLayerNorm.initialize(manager, dataType, tokenShape);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> runtimeParameters) {
      throw new UnsupportedOperationException("Use encode with strategic tokens");
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      return new Shape[] {new Shape(-1, STRATEGIC_TOKEN_COUNT, hiddenSize)};
    }
  }
}

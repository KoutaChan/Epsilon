package com.epsilon.pico.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.ai.model.EpsilonResidualLayerNorm;
import com.epsilon.core.GameState;
import com.epsilon.pico.ai.model.fusion.EpsilonPlayerMemoryFusionExecution;

/**
 * 各プレイヤーの公開履歴を、そのプレイヤー内だけで文脈化する共有エンコーダー。
 *
 * <p>入力順はプレイヤー要約、河の時系列、面子の順で固定する。一人分29 トークンの自己注意を4人へ同じ重みで適用するため、 全構成要素
 * 自己注意より小さい計算量で、リーチ宣言前後、手出し・ツモ切り列、河と副露の対応を各メモリトークンへ保持できる。 パディング トークンはキーと出力の両方から除外する。
 */
final class EpsilonPlayerMemoryEncoder extends AbstractBlock {

  private static final int ATTENTION_HEADS = EpsilonMahjongStateEncoder.ATTENTION_HEADS;
  private static final int MAXIMUM_ATTENTION_WIDTH = 64;
  private static final int MAXIMUM_FEED_FORWARD_WIDTH = 128;
  private static final String DENSE_GRAPH_CAPTURE = "denseGraphCapture";

  private final int hiddenSize;
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

  EpsilonPlayerMemoryEncoder(int hiddenSize) {
    if (hiddenSize <= 0 || hiddenSize % ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be positive and divisible by " + ATTENTION_HEADS);
    }
    this.hiddenSize = hiddenSize;
    attentionWidth = Math.min(hiddenSize, MAXIMUM_ATTENTION_WIDTH);
    attentionHeadSize = attentionWidth / ATTENTION_HEADS;
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
        addChildBlock("feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
    feedForwardProjection = addChildBlock("feedForwardProjection", hiddenLinear());
    outputLayerNorm = addChildBlock("outputLayerNorm", LayerNorm.builder().axis(2).build());
  }

  /**
   * プレイヤーごとの公開メモリを文脈化する。
   *
   * @param playerTokens {@code [batch,4,29,hidden]} のプレイヤー-局所的なトークン列
   * @param playerTokenMask {@code [batch,4,29]} の存在マスク。各プレイヤーの先頭トークンは必ず存在する
   * @return 入力と同形状の文脈化済みメモリ
   */
  NDArray encode(
      ParameterStore parameterStore,
      NDArray playerTokens,
      NDArray playerTokenMask,
      LayerNorm entityTokenLayerNorm,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return encode(
        parameterStore,
        playerTokens,
        playerTokenMask,
        null,
        entityTokenLayerNorm,
        training,
        runtimeParameters);
  }

  NDArray encode(
      ParameterStore parameterStore,
      NDArray playerTokens,
      NDArray playerTokenMask,
      NDArray presentTokenIndices,
      LayerNorm entityTokenLayerNorm,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      return encodeInternal(
          parameterStore,
          playerTokens,
          playerTokenMask,
          presentTokenIndices,
          entityTokenLayerNorm,
          true,
          runtimeParameters);
    }
    NDManager outputManager = playerTokens.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(playerTokens, playerTokenMask);
      if (presentTokenIndices != null) {
        scope.tempAttachAll(presentTokenIndices);
      }
      NDArray encoded =
          encodeInternal(
              parameterStore,
              playerTokens,
              playerTokenMask,
              presentTokenIndices,
              entityTokenLayerNorm,
              false,
              runtimeParameters);
      outputManager.attachAll(encoded);
      return encoded;
    }
  }

  private NDArray encodeInternal(
      ParameterStore parameterStore,
      NDArray playerTokens,
      NDArray playerTokenMask,
      NDArray suppliedPresentTokenIndices,
      LayerNorm entityTokenLayerNorm,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    Shape shape = playerTokens.getShape();
    Shape maskShape = playerTokenMask.getShape();
    if (shape.dimension() != 4
        || shape.get(1) != GameState.NUM_PLAYERS
        || shape.get(2) != EpsilonMahjongStateEncoder.PLAYER_MEMORY_TOKEN_COUNT
        || shape.get(3) != hiddenSize
        || !maskShape.equals(new Shape(shape.get(0), shape.get(1), shape.get(2)))) {
      throw new IllegalArgumentException(
          "player memory shape mismatch: " + shape + " / " + maskShape);
    }

    long flattenedPlayers = shape.get(0) * GameState.NUM_PLAYERS;
    int tokenCount = EpsilonMahjongStateEncoder.PLAYER_MEMORY_TOKEN_COUNT;
    NDArray tokens = playerTokens.reshape(flattenedPlayers, tokenCount, hiddenSize);
    if (!training && Boolean.TRUE.equals(runtimeParameters.get(DENSE_GRAPH_CAPTURE))) {
      NDArray mask =
          playerTokenMask
              .reshape(flattenedPlayers, tokenCount)
              .toType(tokens.getDataType(), false)
              .stopGradient();
      tokens =
          entityTokenLayerNorm
              .forward(parameterStore, new NDList(tokens), false, runtimeParameters)
              .singletonOrThrow()
              .mul(mask.expandDims(2));
      return encodeDense(parameterStore, tokens, mask, false, runtimeParameters).reshape(shape);
    }
    long flattenedTokenCount = flattenedPlayers * tokenCount;
    NDArray presentTokenIndices =
        suppliedPresentTokenIndices == null
            ? EpsilonMaskedRows.indices(playerTokenMask)
            : suppliedPresentTokenIndices;
    NDArray presentTokens =
        EpsilonMaskedRows.gather(
                tokens.reshape(flattenedTokenCount, hiddenSize), presentTokenIndices)
            .expandDims(1);
    presentTokens =
        entityTokenLayerNorm
            .forward(parameterStore, new NDList(presentTokens), training, runtimeParameters)
            .singletonOrThrow();
    long presentTokenCount = presentTokens.getShape().get(0);
    NDArray normalized =
        attentionInputLayerNorm
            .forward(parameterStore, new NDList(presentTokens), training, runtimeParameters)
            .singletonOrThrow();
    NDArray presentQueryKeyValues =
        applyLinear(
            queryKeyValueProjection, parameterStore, normalized, training, runtimeParameters);
    NDArray queryKeyValues =
        EpsilonMaskedRows.scatter(
                presentQueryKeyValues.reshape(presentTokenCount, attentionWidth * 3L),
                presentTokenIndices,
                flattenedTokenCount)
            .reshape(flattenedPlayers, tokenCount, attentionWidth * 3L);
    NDArray queries = attentionHeads(queryKeyValues.get("...,0:{}", attentionWidth));
    NDArray keys =
        attentionHeads(queryKeyValues.get("...,{}:{}", attentionWidth, attentionWidth * 2));
    NDArray values =
        attentionHeads(queryKeyValues.get("...,{}:{}", attentionWidth * 2, attentionWidth * 3));
    // 存在マスクは0/1なので、対数を取りパディングだけを厳密な負の無限大へ変換する。
    NDArray attentionMask =
        playerTokenMask
            .reshape(flattenedPlayers, 1, 1, tokenCount)
            .toType(queries.getDataType(), false)
            .log()
            .stopGradient();
    NDArray context =
        queries
            .getNDArrayInternal()
            .scaledDotProductAttention(keys, values, attentionMask, 0.0, false)
            .swapAxes(1, 2)
            .reshape(flattenedPlayers, tokenCount, attentionWidth);
    NDArray presentContext =
        EpsilonMaskedRows.gather(
                context.reshape(flattenedTokenCount, attentionWidth), presentTokenIndices)
            .expandDims(1);
    NDArray projectedAttention =
        applyLinear(
            attentionOutputProjection, parameterStore, presentContext, training, runtimeParameters);
    if (training) {
      NDList attentionOutput =
          EpsilonResidualLayerNorm.applyTraining(
              parameterStore, presentTokens, projectedAttention, feedForwardInputLayerNorm);
      normalized = attentionOutput.get(0);
      presentTokens = attentionOutput.get(1);
    } else {
      normalized =
          addToOwnedResidualAndNormalize(
              parameterStore, presentTokens, projectedAttention, feedForwardInputLayerNorm);
    }
    NDArray expanded =
        EpsilonMahjongStateEncoder.silu(
            applyLinear(
                feedForwardExpansion, parameterStore, normalized, training, runtimeParameters));
    NDArray projectedFeedForward =
        applyLinear(feedForwardProjection, parameterStore, expanded, training, runtimeParameters);
    NDArray encodedPresentTokens;
    if (training) {
      NDList feedForwardOutput =
          EpsilonResidualLayerNorm.applyTraining(
              parameterStore, presentTokens, projectedFeedForward, outputLayerNorm);
      encodedPresentTokens = feedForwardOutput.get(0);
    } else {
      encodedPresentTokens =
          addToOwnedResidualAndNormalize(
              parameterStore, presentTokens, projectedFeedForward, outputLayerNorm);
    }
    encodedPresentTokens = encodedPresentTokens.reshape(presentTokenCount, hiddenSize);
    return EpsilonMaskedRows.scatter(encodedPresentTokens, presentTokenIndices, flattenedTokenCount)
        .reshape(shape);
  }

  /**
   * 固定形状グラフ記録用のデータ-independent経路。
   *
   * <p>パディングも同じ密な演算へ通すが、キーマスクと各残差接続後のマスクにより存在トークンの意味は通常経路と同じになる。 {@code
   * nonzero}由来の動的形状を排除することだけが目的で、パラメーターと出力スキーマは変えない。
   */
  private NDArray encodeDense(
      ParameterStore parameterStore,
      NDArray tokens,
      NDArray playerTokenMask,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long flattenedPlayers = tokens.getShape().get(0);
    int tokenCount = EpsilonMahjongStateEncoder.PLAYER_MEMORY_TOKEN_COUNT;
    NDArray normalized =
        attentionInputLayerNorm
            .forward(parameterStore, new NDList(tokens), training, runtimeParameters)
            .singletonOrThrow();
    NDArray queryKeyValues =
        applyLinear(
            queryKeyValueProjection, parameterStore, normalized, training, runtimeParameters);
    NDArray queries = attentionHeads(queryKeyValues.get("...,0:{}", attentionWidth));
    NDArray keys =
        attentionHeads(queryKeyValues.get("...,{}:{}", attentionWidth, attentionWidth * 2));
    NDArray values =
        attentionHeads(queryKeyValues.get("...,{}:{}", attentionWidth * 2, attentionWidth * 3));
    NDArray mask = playerTokenMask.toType(queries.getDataType(), false).stopGradient();
    NDArray attentionMask = mask.reshape(flattenedPlayers, 1, 1, tokenCount).log();
    NDArray context =
        queries
            .getNDArrayInternal()
            .scaledDotProductAttention(keys, values, attentionMask, 0.0, false)
            .swapAxes(1, 2)
            .reshape(flattenedPlayers, tokenCount, attentionWidth);
    NDArray expandedMask = mask.expandDims(2);
    tokens =
        tokens
            .add(
                applyLinear(
                    attentionOutputProjection,
                    parameterStore,
                    context,
                    training,
                    runtimeParameters))
            .mul(expandedMask);
    normalized =
        feedForwardInputLayerNorm
            .forward(parameterStore, new NDList(tokens), training, runtimeParameters)
            .singletonOrThrow();
    NDArray expanded =
        EpsilonMahjongStateEncoder.silu(
            applyLinear(
                feedForwardExpansion, parameterStore, normalized, training, runtimeParameters));
    tokens =
        tokens
            .add(
                applyLinear(
                    feedForwardProjection, parameterStore, expanded, training, runtimeParameters))
            .mul(expandedMask);
    return outputLayerNorm
        .forward(parameterStore, new NDList(tokens), training, runtimeParameters)
        .singletonOrThrow()
        .mul(expandedMask);
  }

  private NDArray attentionHeads(NDArray tokens) {
    long flattenedPlayers = tokens.getShape().get(0);
    long tokenCount = tokens.getShape().get(1);
    return tokens
        .reshape(flattenedPlayers, tokenCount, ATTENTION_HEADS, attentionHeadSize)
        .swapAxes(1, 2);
  }

  /** 重みを固定したプレイヤー-メモリ重みを、Fusion実装へ渡す名前付きビューとして解決する。 */
  EpsilonPlayerMemoryFusionExecution.Parameters fusionParameters(
      NDManager manager, ParameterStore parameterStore, LayerNorm entityTokenLayerNorm) {
    return new EpsilonPlayerMemoryFusionExecution.Parameters(
        hiddenSize,
        ATTENTION_HEADS,
        attentionWidth,
        feedForwardWidth,
        parameterStore.getValue(
            entityTokenLayerNorm.getParameters().get("gamma"), manager.getDevice(), false),
        parameterStore.getValue(
            entityTokenLayerNorm.getParameters().get("beta"), manager.getDevice(), false),
        parameterStore.getValue(
            attentionInputLayerNorm.getParameters().get("gamma"), manager.getDevice(), false),
        parameterStore.getValue(
            attentionInputLayerNorm.getParameters().get("beta"), manager.getDevice(), false),
        parameterStore.getValue(
            queryKeyValueProjection.getParameters().get("weight"), manager.getDevice(), false),
        parameterStore.getValue(
            attentionOutputProjection.getParameters().get("weight"), manager.getDevice(), false),
        parameterStore.getValue(
            attentionOutputProjection.getParameters().get("bias"), manager.getDevice(), false),
        parameterStore.getValue(
            feedForwardInputLayerNorm.getParameters().get("gamma"), manager.getDevice(), false),
        parameterStore.getValue(
            feedForwardInputLayerNorm.getParameters().get("beta"), manager.getDevice(), false),
        parameterStore.getValue(
            feedForwardExpansion.getParameters().get("weight"), manager.getDevice(), false),
        parameterStore.getValue(
            feedForwardExpansion.getParameters().get("bias"), manager.getDevice(), false),
        parameterStore.getValue(
            feedForwardProjection.getParameters().get("weight"), manager.getDevice(), false),
        parameterStore.getValue(
            feedForwardProjection.getParameters().get("bias"), manager.getDevice(), false),
        parameterStore.getValue(
            outputLayerNorm.getParameters().get("gamma"), manager.getDevice(), false),
        parameterStore.getValue(
            outputLayerNorm.getParameters().get("beta"), manager.getDevice(), false));
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape tokenShape =
        new Shape(-1, EpsilonMahjongStateEncoder.PLAYER_MEMORY_TOKEN_COUNT, hiddenSize);
    attentionInputLayerNorm.initialize(manager, dataType, tokenShape);
    queryKeyValueProjection.initialize(manager, dataType, tokenShape);
    attentionOutputProjection.initialize(
        manager,
        dataType,
        new Shape(-1, EpsilonMahjongStateEncoder.PLAYER_MEMORY_TOKEN_COUNT, attentionWidth));
    feedForwardInputLayerNorm.initialize(manager, dataType, tokenShape);
    feedForwardExpansion.initialize(manager, dataType, tokenShape);
    feedForwardProjection.initialize(
        manager,
        dataType,
        new Shape(-1, EpsilonMahjongStateEncoder.PLAYER_MEMORY_TOKEN_COUNT, feedForwardWidth));
    outputLayerNorm.initialize(manager, dataType, tokenShape);
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    throw new UnsupportedOperationException("Use encode with shared entity token normalization");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {inputShapes[0]};
  }

  private Linear hiddenLinear() {
    return Linear.builder().setUnits(hiddenSize).build();
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
}

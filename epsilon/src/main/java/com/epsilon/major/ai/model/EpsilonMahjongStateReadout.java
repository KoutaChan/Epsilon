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
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonResidualLayerNorm;

/**
 * 共有された麻雀局面の特徴表現を、方策または価値に使う一つの局面表現へ集約する。
 *
 * <p>方策と価値は牌・河・副露の高価な局所文脈化を共有する一方、このブロックのパラメーターは共有しない。したがって、方策は
 * 行動選択に必要な構成要素を、価値は着順予測に必要な構成要素を、それぞれ別のクエリと射影で集約できる。
 *
 * <p>トークンと出力は{@code hiddenSize}次元を保ち、注意機構通信だけを最大64次元へ制限する。残差接続の恒等写像は圧縮されないため、
 * 低ランクなのは周囲から受け取る更新量だけである。
 */
public final class EpsilonMahjongStateReadout extends AbstractBlock {

  private static final int ATTENTION_HEADS = EpsilonMahjongStateEncoder.ATTENTION_HEADS;
  private static final int MAXIMUM_ATTENTION_WIDTH = 64;
  private static final int DEFAULT_MAXIMUM_FEED_FORWARD_WIDTH = 128;

  private final int hiddenSize;
  private final int attentionWidth;
  private final int attentionHeadSize;
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
   * 指定した構成要素隠れ層幅の目的固有特徴量の集約を構築する。
   *
   * @param hiddenSize 構成要素トークンと出力状態の幅
   */
  public EpsilonMahjongStateReadout(int hiddenSize) {
    this(hiddenSize, DEFAULT_MAXIMUM_FEED_FORWARD_WIDTH);
  }

  /**
   * 指定した上限までFFNを拡張する目的固有特徴量の集約を構築する。
   *
   * @param hiddenSize 構成要素トークンと出力状態の幅
   * @param maximumFeedForwardWidth 特徴量の集約 FFN幅の上限
   */
  EpsilonMahjongStateReadout(int hiddenSize, int maximumFeedForwardWidth) {
    if (hiddenSize <= 0 || hiddenSize % ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be positive and divisible by " + ATTENTION_HEADS);
    }
    if (maximumFeedForwardWidth <= 0) {
      throw new IllegalArgumentException("maximumFeedForwardWidth must be positive");
    }
    this.hiddenSize = hiddenSize;
    attentionWidth = Math.min(hiddenSize, MAXIMUM_ATTENTION_WIDTH);
    attentionHeadSize = attentionWidth / ATTENTION_HEADS;
    feedForwardWidth = Math.min(hiddenSize * 2, maximumFeedForwardWidth);
    querySeedProjection =
        addChildBlock("querySeedProjection", Linear.builder().setUnits(hiddenSize).build());
    queryLayerNorm = addChildBlock("queryLayerNorm", LayerNorm.builder().axis(1).build());
    queryProjection =
        addChildBlock("queryProjection", Linear.builder().setUnits(attentionWidth).build());
    keyValueProjection =
        addChildBlock(
            "keyValueProjection",
            Linear.builder().setUnits(attentionWidth * 2L).optBias(false).build());
    contextProjection =
        addChildBlock("contextProjection", Linear.builder().setUnits(hiddenSize).build());
    feedForwardInputLayerNorm =
        addChildBlock("feedForwardInputLayerNorm", LayerNorm.builder().axis(1).build());
    feedForwardExpansion =
        addChildBlock("feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
    feedForwardProjection =
        addChildBlock("feedForwardProjection", Linear.builder().setUnits(hiddenSize).build());
    outputLayerNorm = addChildBlock("outputLayerNorm", LayerNorm.builder().axis(1).build());
  }

  /**
   * 共有メモリの全有効構成要素を一つの {@code [batch, hidden]} 表現へ集約する。
   *
   * @param parameterStore パラメーター取得先
   * @param memory 状態エンコーダーが構築した共有局・プレイヤー・牌などの特徴表現
   * @param training 学習時の順伝播なら {@code true}
   * @param runtimeParameters DJL ブロックへ渡す実行時パラメーター
   * @return 目的固有の状態埋め込み
   */
  public NDArray read(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedMemory memory,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      return readInternal(parameterStore, memory, summarize(memory), true, runtimeParameters);
    }
    NDManager outputManager = memory.entityEmbeddings().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(memory.roundEmbedding(), memory.entityEmbeddings(), memory.entityMask());
      NDArray stateEmbedding =
          readInternal(parameterStore, memory, summarize(memory), false, runtimeParameters);
      outputManager.attachAll(stateEmbedding);
      return stateEmbedding;
    }
  }

  /** 方策/価値の同時順伝播で共有済みの構成要素要約を使ってメモリを読み出す。 */
  NDArray read(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedMemory memory,
      ReadoutContext readoutContext,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      return readInternal(parameterStore, memory, readoutContext, true, runtimeParameters);
    }
    NDManager outputManager = memory.entityEmbeddings().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          memory.roundEmbedding(),
          memory.entityEmbeddings(),
          memory.entityMask(),
          readoutContext.meanEmbedding(),
          readoutContext.attentionMask());
      NDArray stateEmbedding =
          readInternal(parameterStore, memory, readoutContext, false, runtimeParameters);
      outputManager.attachAll(stateEmbedding);
      return stateEmbedding;
    }
  }

  /** パラメーターを持たない構成要素平均と注意機構マスクを一度だけ構築する。 */
  static ReadoutContext summarize(EpsilonMahjongStateEncoder.EncodedMemory memory) {
    NDArray entityEmbeddings = memory.entityEmbeddings();
    NDArray entityMask = memory.entityMask();
    long rowCount = entityEmbeddings.getShape().get(0);
    NDArray meanEmbedding =
        entityEmbeddings
            .sum(new int[] {1})
            .div(entityMask.sum(new int[] {1}).reshape(rowCount, 1).maximum(1.0f));
    // 存在マスクは0/1なので、内容スコアの大きさによらずパディングを除外する。
    NDArray attentionMask =
        entityMask
            .reshape(rowCount, 1, 1, EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT)
            .log()
            .stopGradient();
    return new ReadoutContext(meanEmbedding, attentionMask);
  }

  private NDArray readInternal(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedMemory memory,
      ReadoutContext readoutContext,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray entityEmbeddings = memory.entityEmbeddings();
    long rowCount = entityEmbeddings.getShape().get(0);
    NDArray querySeed =
        applyLinear(
            querySeedProjection,
            parameterStore,
            memory.roundEmbedding().concat(readoutContext.meanEmbedding(), 1),
            training,
            runtimeParameters);
    NDArray queries =
        applyLinear(queryProjection, parameterStore, querySeed, training, runtimeParameters)
            .reshape(rowCount, ATTENTION_HEADS, 1, attentionHeadSize);
    NDArray keyValues =
        applyLinear(
            keyValueProjection, parameterStore, entityEmbeddings, training, runtimeParameters);
    NDArray keys =
        keyValues
            .get("...,0:{}", attentionWidth)
            .reshape(
                rowCount,
                EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT,
                ATTENTION_HEADS,
                attentionHeadSize)
            .swapAxes(1, 2);
    NDArray values =
        keyValues
            .get("...,{}:{}", attentionWidth, attentionWidth * 2)
            .reshape(
                rowCount,
                EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT,
                ATTENTION_HEADS,
                attentionHeadSize)
            .swapAxes(1, 2);
    NDArray attentionContext =
        queries
            .getNDArrayInternal()
            .scaledDotProductAttention(keys, values, readoutContext.attentionMask(), 0.0, false)
            .reshape(rowCount, attentionWidth);
    NDArray projectedContext =
        applyLinear(
            contextProjection, parameterStore, attentionContext, training, runtimeParameters);
    NDArray stateEmbedding;
    if (training) {
      stateEmbedding =
          EpsilonResidualLayerNorm.applyTraining(
                  parameterStore, querySeed, projectedContext, queryLayerNorm)
              .get(0);
    } else {
      stateEmbedding =
          addToOwnedResidualAndNormalize(
              parameterStore, querySeed, projectedContext, queryLayerNorm);
    }
    NDArray normalized =
        feedForwardInputLayerNorm
            .forward(parameterStore, new NDList(stateEmbedding), training, runtimeParameters)
            .singletonOrThrow();
    NDArray feedForward =
        EpsilonMahjongStateEncoder.silu(
            applyLinear(
                feedForwardExpansion, parameterStore, normalized, training, runtimeParameters));
    NDArray projectedFeedForward =
        applyLinear(
            feedForwardProjection, parameterStore, feedForward, training, runtimeParameters);
    if (training) {
      return EpsilonResidualLayerNorm.applyTraining(
              parameterStore, stateEmbedding, projectedFeedForward, outputLayerNorm)
          .get(0);
    }
    return addToOwnedResidualAndNormalize(
        parameterStore, stateEmbedding, projectedFeedForward, outputLayerNorm);
  }

  /** 2つの特徴量の集約が共有できるパラメーターを持たない構成要素要約。 */
  record ReadoutContext(NDArray meanEmbedding, NDArray attentionMask) {

    ReadoutContext stopGradient() {
      return new ReadoutContext(meanEmbedding.stopGradient(), attentionMask);
    }
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    querySeedProjection.initialize(manager, dataType, new Shape(-1, hiddenSize * 2L));
    queryLayerNorm.initialize(manager, dataType, new Shape(-1, hiddenSize));
    queryProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    keyValueProjection.initialize(
        manager,
        dataType,
        new Shape(-1, EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT, hiddenSize));
    contextProjection.initialize(manager, dataType, new Shape(-1, attentionWidth));
    feedForwardInputLayerNorm.initialize(manager, dataType, new Shape(-1, hiddenSize));
    feedForwardExpansion.initialize(manager, dataType, new Shape(-1, hiddenSize));
    feedForwardProjection.initialize(manager, dataType, new Shape(-1, feedForwardWidth));
    outputLayerNorm.initialize(manager, dataType, new Shape(-1, hiddenSize));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    throw new UnsupportedOperationException("Use read with EncodedMemory");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, hiddenSize)};
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

  /**
   * このブロックが所有する凍結パラメーターを、名前付きの非所有ビューとして返す。
   *
   * <p>返すレコードは現在のデバイス上の{@link NDArray}参照だけを保持し、テンソルを複製しない。パラメーターと配列の寿命は引き続きこのブロックと {@link
   * ParameterStore}の所有者が管理するため、呼び出し側は各配列を閉じてはならない。
   *
   * @param parameterStore パラメーター取得先
   * @param manager パラメーターを参照するデバイスを持つ管理元
   * @return Fusion実行境界へ渡す特徴量の集約パラメーターの変更不可ビュー
   */
  FrozenParameterView frozenParameterView(ParameterStore parameterStore, NDManager manager) {
    return new FrozenParameterView(
        hiddenSize,
        parameterStore.getValue(
            querySeedProjection.getParameters().get("weight"), manager.getDevice(), false),
        parameterStore.getValue(
            querySeedProjection.getParameters().get("bias"), manager.getDevice(), false),
        parameterStore.getValue(
            queryProjection.getParameters().get("weight"), manager.getDevice(), false),
        parameterStore.getValue(
            queryProjection.getParameters().get("bias"), manager.getDevice(), false),
        parameterStore.getValue(
            keyValueProjection.getParameters().get("weight"), manager.getDevice(), false),
        parameterStore.getValue(
            contextProjection.getParameters().get("weight"), manager.getDevice(), false),
        parameterStore.getValue(
            contextProjection.getParameters().get("bias"), manager.getDevice(), false),
        parameterStore.getValue(
            queryLayerNorm.getParameters().get("gamma"), manager.getDevice(), false),
        parameterStore.getValue(
            queryLayerNorm.getParameters().get("beta"), manager.getDevice(), false),
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

  /**
   * 特徴量の集約ブロックが所有する凍結パラメーターへの名前付き非所有ビュー。
   *
   * <p>レコード自体は変更不可だが、各{@link NDArray}の所有権は元のブロック側に残る。Fusion側は参照を定数へ結び付けるだけで、複製や解放を行わない。
   */
  public record FrozenParameterView(
      int hiddenSize,
      NDArray querySeedWeight,
      NDArray querySeedBias,
      NDArray queryWeight,
      NDArray queryBias,
      NDArray keyValueWeight,
      NDArray contextWeight,
      NDArray contextBias,
      NDArray queryNormWeight,
      NDArray queryNormBias,
      NDArray feedForwardNormWeight,
      NDArray feedForwardNormBias,
      NDArray feedForwardExpansionWeight,
      NDArray feedForwardExpansionBias,
      NDArray feedForwardProjectionWeight,
      NDArray feedForwardProjectionBias,
      NDArray outputNormWeight,
      NDArray outputNormBias) {}
}

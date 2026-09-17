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
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonResidualLayerNorm;
import com.epsilon.ai.model.MahjongTileRelation;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.model.fusion.EpsilonTileRelationFusionExecution;
import java.nio.ShortBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;

/**
 * 34牌種トークンを、牌種間の固定関係を持つ自己注意で文脈化するエンコーダー。
 *
 * <p>各注意ヘッドのクエリ {@code Q_i}、キー {@code K_j} と関係種別 {@code r(i,j)} に対し、注意機構ロジットは概念的に次式である。
 *
 * <pre>{@code
 * score(i,j,h) = (Q_i,h · K_j,h + Q_i,h · R_key(h,r(i,j))) / sqrt(headSize)
 *                + R_bias(h,r(i,j))
 * }</pre>
 *
 * <p>スカラーバイアスは関係ごとの固定的な重みを、関係キーはクエリ内容に依存した関係選択を表す。関係パラメーターはゼロ初期化し、
 * 初期状態では通常の自己注意と同じ挙動から学習を開始する。各ブロックは事前の層正規化、注意機構の残差接続、SiLU
 * フィードフォワード残差で構成し、層の積み重ね末尾の層正規化で候補スコア計算処理へ渡す牌トークンの尺度を固定する。
 */
public final class EpsilonTileRelationEncoder extends AbstractBlock {

  /** 積み重ねる牌種間の関係を考慮する注意機構ブロック数。 */
  public static final int BLOCK_COUNT = 2;

  /** 各ブロックの注意ヘッド数。 */
  public static final int ATTENTION_HEADS = 4;

  private static final int MAXIMUM_ATTENTION_WIDTH = 64;
  private static final int MAXIMUM_FEED_FORWARD_WIDTH = 128;
  private static final int RELATION_COUNT = MahjongTileRelation.values().length;
  private static final long[] RELATION_IDS = MahjongTileRelation.relationIds();
  private static final long[] RELATION_TYPE_IDS = createRelationTypeIds();
  private static final String RUNTIME_RELATION_IDS = "tileRelationIds";
  private static final String RUNTIME_RELATION_TYPE_IDS = "tileRelationTypeIds";

  /** 関係集合とエンコーダー構成を要約したチェックポイント互換互換性識別子。 */
  public static final String FINGERPRINT = fingerprint();

  private final int hiddenSize;
  private final TileRelationBlock[] blocks;
  private final LayerNorm outputLayerNorm;

  /**
   * 指定幅の牌関係エンコーダーを構築する。
   *
   * @param hiddenSize 牌トークン幅。注意ヘッド数で割り切れる正数
   */
  public EpsilonTileRelationEncoder(int hiddenSize) {
    if (hiddenSize <= 0 || hiddenSize % ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be positive and divisible by " + ATTENTION_HEADS);
    }
    this.hiddenSize = hiddenSize;
    blocks = new TileRelationBlock[BLOCK_COUNT];
    for (int blockIndex = 0; blockIndex < BLOCK_COUNT; blockIndex++) {
      blocks[blockIndex] = addChildBlock("block" + blockIndex, new TileRelationBlock(hiddenSize));
    }
    outputLayerNorm = addChildBlock("outputLayerNorm", LayerNorm.builder().axis(2).build());
  }

  /**
   * Fusion 実行処理へ渡す凍結済みパラメーター参照を組み立てる。
   *
   * <p>ブロック自体は公開せず、パラメーターを保持するNDArray 参照だけを名前付きレコードへ写す。重みを複製しないため、 返却値の寿命は指定管理元とパラメーター保存先に従う。
   */
  EpsilonTileRelationFusionExecution.Parameters fusionParameters(
      ParameterStore parameterStore, NDManager manager) {
    short[] compactRelationIds = new short[RELATION_IDS.length];
    for (int index = 0; index < RELATION_IDS.length; index++) {
      compactRelationIds[index] = (short) RELATION_IDS[index];
    }
    NDArray relationIds =
        manager.create(
            ShortBuffer.wrap(compactRelationIds),
            new Shape(Tile.NUM_TILE_TYPES, Tile.NUM_TILE_TYPES),
            DataType.INT16);
    List<EpsilonTileRelationFusionExecution.BlockParameters> blockParameters =
        new ArrayList<>(blocks.length);
    for (TileRelationBlock block : blocks) {
      blockParameters.add(
          new EpsilonTileRelationFusionExecution.BlockParameters(
              value(parameterStore, block.attentionInputLayerNorm, "gamma", manager),
              value(parameterStore, block.attentionInputLayerNorm, "beta", manager),
              value(parameterStore, block.queryKeyValueProjection, "weight", manager),
              value(parameterStore, block.relationKeyEmbedding, "embedding", manager),
              value(parameterStore, block.relationBiasEmbedding, "embedding", manager),
              value(parameterStore, block.attentionOutputProjection, "weight", manager),
              value(parameterStore, block.attentionOutputProjection, "bias", manager),
              value(parameterStore, block.feedForwardInputLayerNorm, "gamma", manager),
              value(parameterStore, block.feedForwardInputLayerNorm, "beta", manager),
              value(parameterStore, block.feedForwardExpansion, "weight", manager),
              value(parameterStore, block.feedForwardExpansion, "bias", manager),
              value(parameterStore, block.feedForwardProjection, "weight", manager),
              value(parameterStore, block.feedForwardProjection, "bias", manager)));
    }
    return new EpsilonTileRelationFusionExecution.Parameters(
        relationIds,
        value(parameterStore, outputLayerNorm, "gamma", manager),
        value(parameterStore, outputLayerNorm, "beta", manager),
        blockParameters);
  }

  private static NDArray value(
      ParameterStore parameterStore, AbstractBlock block, String name, NDManager manager) {
    return parameterStore.getValue(block.getParameters().get(name), manager.getDevice(), false);
  }

  /**
   * 推論サーバーが全バッチで共有する固定関係テンソルをデバイス上へ一度だけ作る。
   *
   * <p>通常順伝播の定数ホストからデバイスへの転送を除き、HIP 計算グラフ記録中にもページ固定されていないホストメモリを参照しない。
   *
   * @param manager 固定テンソルを所有する長寿命デバイス管理元
   * @return {@link #encode} へ渡す関係実行時パラメーター
   */
  public static PairList<String, Object> createRuntimeParameters(NDManager manager) {
    PairList<String, Object> parameters = new PairList<>(2);
    parameters.add(
        RUNTIME_RELATION_IDS,
        manager.create(RELATION_IDS, new Shape(Tile.NUM_TILE_TYPES, Tile.NUM_TILE_TYPES)));
    parameters.add(
        RUNTIME_RELATION_TYPE_IDS, manager.create(RELATION_TYPE_IDS, new Shape(RELATION_COUNT)));
    return parameters;
  }

  /**
   * 34牌種トークンへ固定関係付き自己注意を適用する。
   *
   * @param parameterStore パラメーター取得先
   * @param tileTokens {@code [batch,34,hiddenSize]} の入力牌トークン
   * @param training 学習挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 同形状の文脈化・正規化済み牌トークン
   */
  public NDArray encode(
      ParameterStore parameterStore,
      NDArray tileTokens,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      return encodeTraining(parameterStore, tileTokens, runtimeParameters);
    }
    NDManager outputManager = tileTokens.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(tileTokens);
      NDArray contextualTileTokens =
          encodeInference(parameterStore, tileTokens, false, runtimeParameters);
      outputManager.attachAll(contextualTileTokens);
      return contextualTileTokens;
    }
  }

  /**
   * 呼び出し元が所有権を譲った牌トークンバッファを再利用して推論する。
   *
   * <p>入力はこの呼び出し後に再利用できない。学習経路と共有入力には {@link #encode} を使う。
   */
  NDArray encodeOwnedInference(
      ParameterStore parameterStore,
      NDArray ownedTileTokens,
      PairList<String, Object> runtimeParameters) {
    return encodeOwnedInference(parameterStore, ownedTileTokens, null, runtimeParameters);
  }

  NDArray encodeOwnedInference(
      ParameterStore parameterStore,
      NDArray ownedTileTokens,
      EpsilonTileRelationFusionExecution.Forward inferenceForward,
      PairList<String, Object> runtimeParameters) {
    if (inferenceForward != null) {
      requireTileShape(ownedTileTokens);
      return inferenceForward.encodeBorrowed(ownedTileTokens);
    }
    NDManager outputManager = ownedTileTokens.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(ownedTileTokens);
      NDArray contextualTileTokens =
          encodeInference(parameterStore, ownedTileTokens, true, runtimeParameters);
      outputManager.attachAll(contextualTileTokens);
      return contextualTileTokens;
    }
  }

  /** バックエンドが利用可能な融合演算を選び、一つの牌トークンバッファを全ブロックで再利用する。 */
  private NDArray encodeInference(
      ParameterStore parameterStore,
      NDArray tileTokens,
      boolean reuseOwnedInput,
      PairList<String, Object> runtimeParameters) {
    requireTileShape(tileTokens);
    NDManager manager = tileTokens.getManager();
    NDArray relationIds = relationIds(manager, runtimeParameters);
    NDArray relationTypeIds = relationTypeIds(manager, runtimeParameters);
    // 通常登録情報では共有入力を守り、所有する登録情報だけが呼び出し元から譲られたバッファを再利用する。
    NDArray residual = reuseOwnedInput ? tileTokens : tileTokens.duplicate();
    NDArray normalizedInput = null;
    for (int blockIndex = 0; blockIndex < blocks.length; blockIndex++) {
      LayerNorm followingLayerNorm =
          blockIndex + 1 < blocks.length
              ? blocks[blockIndex + 1].attentionInputLayerNorm
              : outputLayerNorm;
      InferenceBlockOutput output =
          encodeInferenceBlock(
              blocks[blockIndex],
              followingLayerNorm,
              parameterStore,
              residual,
              normalizedInput,
              relationIds,
              relationTypeIds,
              runtimeParameters);
      residual = output.residual();
      normalizedInput = output.normalizedForNextBlock();
    }
    return normalizedInput;
  }

  private InferenceBlockOutput encodeInferenceBlock(
      TileRelationBlock block,
      LayerNorm followingLayerNorm,
      ParameterStore parameterStore,
      NDArray residual,
      NDArray normalizedInput,
      NDArray relationIds,
      NDArray relationTypeIds,
      PairList<String, Object> runtimeParameters) {
    NDManager outputManager = residual.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(residual);
      scope.tempAttachAll(relationIds, relationTypeIds);
      if (normalizedInput != null) {
        scope.tempAttachAll(normalizedInput);
      }
      InferenceBlockOutput output =
          block.encodeResidualLayerNorm(
              parameterStore,
              residual,
              normalizedInput,
              relationIds,
              relationTypeIds,
              followingLayerNorm,
              runtimeParameters);
      outputManager.attachAll(output.residual(), output.normalizedForNextBlock());
      return output;
    }
  }

  private record InferenceBlockOutput(NDArray residual, NDArray normalizedForNextBlock) {}

  private NDArray encodeTraining(
      ParameterStore parameterStore,
      NDArray tileTokens,
      PairList<String, Object> runtimeParameters) {
    requireTileShape(tileTokens);
    NDManager manager = tileTokens.getManager();
    NDArray relationIds = relationIds(manager, runtimeParameters);
    NDArray relationTypeIds = relationTypeIds(manager, runtimeParameters);
    NDArray residual = tileTokens;
    NDArray normalizedInput = null;
    for (int blockIndex = 0; blockIndex < blocks.length; blockIndex++) {
      TileRelationBlock block = blocks[blockIndex];
      LayerNorm followingLayerNorm =
          blockIndex + 1 < blocks.length
              ? blocks[blockIndex + 1].attentionInputLayerNorm
              : outputLayerNorm;
      NDList output =
          block.encodeTrainingResidualLayerNorm(
              parameterStore,
              residual,
              normalizedInput,
              relationIds,
              relationTypeIds,
              followingLayerNorm,
              runtimeParameters);
      normalizedInput = output.get(0);
      residual = output.get(1);
    }
    return normalizedInput;
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape tileShape = new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize);
    for (TileRelationBlock block : blocks) {
      block.initialize(manager, dataType, tileShape);
    }
    outputLayerNorm.initialize(manager, dataType, tileShape);
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
    return new Shape[] {new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize)};
  }

  private void requireTileShape(NDArray tileTokens) {
    Shape shape = tileTokens.getShape();
    if (shape.dimension() != 3
        || shape.get(1) != Tile.NUM_TILE_TYPES
        || shape.get(2) != hiddenSize) {
      throw new IllegalArgumentException(
          "tileTokens shape must be [batch,"
              + Tile.NUM_TILE_TYPES
              + ','
              + hiddenSize
              + "], got "
              + shape);
    }
  }

  private static String fingerprint() {
    String descriptor =
        "relations="
            + Arrays.toString(MahjongTileRelation.values())
            + ";ids="
            + Arrays.toString(MahjongTileRelation.relationIds())
            + ";blocks="
            + BLOCK_COUNT
            + ";heads="
            + ATTENTION_HEADS
            + ";maximumAttentionWidth="
            + MAXIMUM_ATTENTION_WIDTH
            + ";maximumFeedForwardWidth="
            + MAXIMUM_FEED_FORWARD_WIDTH
            + ";layout=preln-silu-low-rank-packed-qkv+relation-head-bias+key-zero-init+final-ln";
    try {
      byte[] digest =
          MessageDigest.getInstance("SHA-256").digest(descriptor.getBytes(StandardCharsets.UTF_8));
      return HexFormat.of().formatHex(digest, 0, 8);
    } catch (NoSuchAlgorithmException impossible) {
      throw new ExceptionInInitializerError(impossible);
    }
  }

  private static long[] createRelationTypeIds() {
    long[] relationTypeIds = new long[RELATION_COUNT];
    for (int relation = 0; relation < relationTypeIds.length; relation++) {
      relationTypeIds[relation] = relation;
    }
    return relationTypeIds;
  }

  private static NDArray relationIds(
      NDManager manager, PairList<String, Object> runtimeParameters) {
    NDArray relationIds = (NDArray) runtimeParameters.get(RUNTIME_RELATION_IDS);
    return relationIds != null
        ? relationIds
        : manager.create(RELATION_IDS, new Shape(Tile.NUM_TILE_TYPES, Tile.NUM_TILE_TYPES));
  }

  private static NDArray relationTypeIds(
      NDManager manager, PairList<String, Object> runtimeParameters) {
    NDArray relationTypeIds = (NDArray) runtimeParameters.get(RUNTIME_RELATION_TYPE_IDS);
    return relationTypeIds != null
        ? relationTypeIds
        : manager.create(RELATION_TYPE_IDS, new Shape(RELATION_COUNT));
  }

  private static final class TileRelationBlock extends AbstractBlock {

    private final int hiddenSize;
    private final int attentionWidth;
    private final int attentionHeadSize;
    private final int feedForwardWidth;
    private final LayerNorm attentionInputLayerNorm;
    private final Linear queryKeyValueProjection;
    private final IdEmbedding relationBiasEmbedding;
    private final IdEmbedding relationKeyEmbedding;
    private final Linear attentionOutputProjection;
    private final LayerNorm feedForwardInputLayerNorm;
    private final Linear feedForwardExpansion;
    private final Linear feedForwardProjection;

    private TileRelationBlock(int hiddenSize) {
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
      IdEmbedding relationBias =
          new IdEmbedding.Builder()
              .setDictionarySize(RELATION_COUNT)
              .setEmbeddingSize(ATTENTION_HEADS)
              .build();
      relationBias.getDirectParameters().valueAt(0).setInitializer(new ConstantInitializer(0.0f));
      relationBiasEmbedding = addChildBlock("relationBiasEmbedding", relationBias);
      IdEmbedding relationKey =
          new IdEmbedding.Builder()
              .setDictionarySize(RELATION_COUNT)
              .setEmbeddingSize(attentionWidth)
              .build();
      relationKey.getDirectParameters().valueAt(0).setInitializer(new ConstantInitializer(0.0f));
      relationKeyEmbedding = addChildBlock("relationKeyEmbedding", relationKey);
      attentionOutputProjection =
          addChildBlock("attentionOutputProjection", Linear.builder().setUnits(hiddenSize).build());
      feedForwardInputLayerNorm =
          addChildBlock("feedForwardInputLayerNorm", LayerNorm.builder().axis(2).build());
      feedForwardExpansion =
          addChildBlock(
              "feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
      feedForwardProjection = addChildBlock("feedForwardProjection", hiddenLinear());
    }

    private NDList encodeTrainingResidualLayerNorm(
        ParameterStore parameterStore,
        NDArray residual,
        NDArray normalizedInput,
        NDArray relationIds,
        NDArray relationTypeIds,
        LayerNorm followingLayerNorm,
        PairList<String, Object> runtimeParameters) {
      NDArray attentionInput =
          normalizedInput != null
              ? normalizedInput
              : attentionInputLayerNorm
                  .forward(parameterStore, new NDList(residual), true, runtimeParameters)
                  .singletonOrThrow();
      NDArray projectedAttention =
          projectAttention(
              parameterStore,
              attentionInput,
              relationIds,
              relationTypeIds,
              true,
              runtimeParameters);
      NDList attentionOutput =
          EpsilonResidualLayerNorm.applyTraining(
              parameterStore, residual, projectedAttention, feedForwardInputLayerNorm);
      NDArray normalizedContext = attentionOutput.get(0);
      NDArray contextualResidual = attentionOutput.get(1);
      NDArray feedForwardHidden =
          EpsilonMahjongStateEncoder.silu(
              applyLinear(
                  feedForwardExpansion,
                  parameterStore,
                  normalizedContext,
                  true,
                  runtimeParameters));
      NDArray projectedFeedForward =
          applyLinear(
              feedForwardProjection, parameterStore, feedForwardHidden, true, runtimeParameters);
      return EpsilonResidualLayerNorm.applyTraining(
          parameterStore, contextualResidual, projectedFeedForward, followingLayerNorm);
    }

    private InferenceBlockOutput encodeResidualLayerNorm(
        ParameterStore parameterStore,
        NDArray residual,
        NDArray normalizedInput,
        NDArray relationIds,
        NDArray relationTypeIds,
        LayerNorm followingLayerNorm,
        PairList<String, Object> runtimeParameters) {
      NDArray attentionInput =
          normalizedInput != null
              ? normalizedInput
              : attentionInputLayerNorm
                  .forward(parameterStore, new NDList(residual), false, runtimeParameters)
                  .singletonOrThrow();
      NDArray projectedAttention =
          projectAttention(
              parameterStore,
              attentionInput,
              relationIds,
              relationTypeIds,
              false,
              runtimeParameters);
      NDArray normalizedContext =
          addToOwnedResidualAndNormalize(
              parameterStore, residual, projectedAttention, feedForwardInputLayerNorm);
      NDArray feedForwardHidden =
          EpsilonMahjongStateEncoder.silu(
              applyLinear(
                  feedForwardExpansion,
                  parameterStore,
                  normalizedContext,
                  false,
                  runtimeParameters));
      NDArray projectedFeedForward =
          applyLinear(
              feedForwardProjection, parameterStore, feedForwardHidden, false, runtimeParameters);
      NDArray normalizedForNextBlock =
          addToOwnedResidualAndNormalize(
              parameterStore, residual, projectedFeedForward, followingLayerNorm);
      return new InferenceBlockOutput(residual, normalizedForNextBlock);
    }

    private NDArray projectAttention(
        ParameterStore parameterStore,
        NDArray normalizedTileTokens,
        NDArray relationIds,
        NDArray relationTypeIds,
        boolean training,
        PairList<String, Object> runtimeParameters) {
      long batchSize = normalizedTileTokens.getShape().get(0);
      NDArray queryKeyValues =
          applyLinear(
              queryKeyValueProjection,
              parameterStore,
              normalizedTileTokens,
              training,
              runtimeParameters);
      NDArray queries =
          queryKeyValues
              .get("...,0:{}", attentionWidth)
              .reshape(batchSize, Tile.NUM_TILE_TYPES, ATTENTION_HEADS, attentionHeadSize)
              .swapAxes(1, 2);
      NDArray keys =
          queryKeyValues
              .get("...,{}:{}", attentionWidth, attentionWidth * 2)
              .reshape(batchSize, Tile.NUM_TILE_TYPES, ATTENTION_HEADS, attentionHeadSize)
              .swapAxes(1, 2);
      NDArray values =
          queryKeyValues
              .get("...,{}:{}", attentionWidth * 2, attentionWidth * 3)
              .reshape(batchSize, Tile.NUM_TILE_TYPES, ATTENTION_HEADS, attentionHeadSize)
              .swapAxes(1, 2);
      NDArray relationBias =
          relationBiasEmbedding
              .forward(parameterStore, new NDList(relationIds), training, runtimeParameters)
              .singletonOrThrow();
      relationBias.attach(normalizedTileTokens.getManager());
      relationBias = relationBias.transpose(2, 0, 1).expandDims(0);
      NDArray relationKeys =
          relationKeys(
              parameterStore,
              relationTypeIds,
              normalizedTileTokens.getManager(),
              training,
              runtimeParameters);
      float attentionScale = (float) (1.0 / Math.sqrt(attentionHeadSize));
      NDArray attentionContext =
          relationAttention(
                  queries, keys, values, relationKeys, relationBias, relationIds, attentionScale)
              .swapAxes(1, 2)
              .reshape(batchSize, Tile.NUM_TILE_TYPES, attentionWidth);
      return applyLinear(
          attentionOutputProjection, parameterStore, attentionContext, training, runtimeParameters);
    }

    private NDArray relationKeys(
        ParameterStore parameterStore,
        NDArray relationTypeIds,
        NDManager workingManager,
        boolean training,
        PairList<String, Object> runtimeParameters) {
      NDArray relationKeys =
          relationKeyEmbedding
              .forward(parameterStore, new NDList(relationTypeIds), training, runtimeParameters)
              .singletonOrThrow();
      relationKeys.attach(workingManager);
      relationKeys =
          relationKeys
              .reshape(RELATION_COUNT, ATTENTION_HEADS, attentionHeadSize)
              .transpose(1, 2, 0)
              .expandDims(0);
      return relationKeys;
    }

    private NDArray relationAttention(
        NDArray queries,
        NDArray keys,
        NDArray values,
        NDArray relationKeys,
        NDArray relationBias,
        NDArray relationIds,
        float attentionScale) {
      return NDArrays.relationBiasedScaledDotProductAttention(
          queries, keys, values, relationKeys, relationBias, relationIds, attentionScale);
    }

    @Override
    protected void initializeChildBlocks(
        NDManager manager, DataType dataType, Shape... inputShapes) {
      Shape tileShape = new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize);
      attentionInputLayerNorm.initialize(manager, dataType, tileShape);
      queryKeyValueProjection.initialize(manager, dataType, tileShape);
      relationBiasEmbedding.initialize(
          manager, dataType, new Shape(Tile.NUM_TILE_TYPES, Tile.NUM_TILE_TYPES));
      relationKeyEmbedding.initialize(manager, dataType, new Shape(RELATION_COUNT));
      attentionOutputProjection.initialize(
          manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES, attentionWidth));
      feedForwardInputLayerNorm.initialize(manager, dataType, tileShape);
      feedForwardExpansion.initialize(manager, dataType, tileShape);
      feedForwardProjection.initialize(
          manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES, feedForwardWidth));
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> runtimeParameters) {
      throw new UnsupportedOperationException("Use relation-aware encode");
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      return new Shape[] {new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize)};
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

package com.epsilon.nano.ai.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.EmbeddingReduction;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.input.DecisionCategoryLayout;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.model.fusion.EpsilonPlayerMemoryFusionExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonStrategicContextFusionExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonTileRelationFusionExecution;

/**
 * 固定長の麻雀構成要素集合を、34牌種と各プレイヤーの公開情報を表す共有表現へ変換するエンコーダー。
 *
 * <p>局、4人、34牌種、河イベント、面子をそれぞれトークンへ投影する。牌種・河・面子には同じ共通形式の牌識別情報を加え、 牌種トークンには {@link
 * EpsilonTileRelationEncoder} を適用する。さらに各プレイヤーの要約・河・面子を {@link EpsilonPlayerMemoryEncoder}
 * で局所文脈化し、河の順序、リーチ宣言前後、手出し列、副露との対応を保持する。 パディングされた河・面子トークンは注意機構キーと出力の両方でマスクする。
 *
 * <p>このクラスは高価な局所文脈化だけを所有する。方策と価値の局面集約は、それぞれ独立した {@link EpsilonMahjongStateReadout}
 * が同じメモリを読む。価値側がこの共有メモリを勾配を切り離して使うことで、価値損失は共有エンコーダーを更新しない。
 */
public final class EpsilonMahjongStateEncoder extends AbstractBlock {

  /** 状態エンコーダーと特徴量の集約が共有する注意ヘッド数。 */
  public static final int ATTENTION_HEADS = 4;

  /** Decision専用の戦略文脈ブロックをチェックポイント識別子へ固定する文字列。 */
  public static final String STRATEGIC_CONTEXT_FINGERPRINT =
      "handpool-c64-strategic6x2h4-c128-f512";

  /** 1人分のプレイヤートークン、河、面子を含むメモリ枠数。 */
  public static final int PLAYER_MEMORY_TOKEN_COUNT =
      1
          + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
          + DecisionInputSchema.MAX_MELDS_PER_PLAYER;

  private static final int CATEGORY_EMBEDDING_SIZE = 8;
  private static final float LAYER_NORM_EPSILON = 1.0e-5f;
  private static final int RIVER_TOKEN_COUNT =
      GameState.NUM_PLAYERS * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER;
  private static final int MELD_TOKEN_COUNT =
      GameState.NUM_PLAYERS * DecisionInputSchema.MAX_MELDS_PER_PLAYER;
  public static final int ENTITY_TOKEN_COUNT =
      1 + GameState.NUM_PLAYERS + Tile.NUM_TILE_TYPES + RIVER_TOKEN_COUNT + MELD_TOKEN_COUNT;

  private final int hiddenSize;
  private final IdEmbedding stateFeatureEmbedding;
  private final IdEmbedding canonicalTileEmbedding;
  private final Linear roundFeatureProjection;
  private final Linear playerFeatureProjection;
  private final Linear tileFeatureProjection;
  private final EpsilonTileRelationEncoder tileRelationEncoder;
  private final Linear riverEventProjection;
  private final Linear meldFeatureProjection;
  private final LayerNorm entityTokenLayerNorm;
  private final EpsilonPlayerMemoryEncoder playerMemoryEncoder;
  private final EpsilonStrategicContextEncoder strategicContextEncoder;
  private NDArray stateCategoryOffsets;

  /**
   * 指定幅の構成要素エンコーダーを構築する。
   *
   * @param hiddenSize 全トークンと集約状態の埋め込み幅。注意ヘッド数で割り切れる正数
   */
  public EpsilonMahjongStateEncoder(int hiddenSize) {
    if (hiddenSize <= 0 || hiddenSize % ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be positive and divisible by " + ATTENTION_HEADS);
    }
    this.hiddenSize = hiddenSize;
    stateFeatureEmbedding =
        addChildBlock(
            "stateFeatureEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(DecisionCategoryLayout.STATE_DICTIONARY_SIZE)
                .setEmbeddingSize(CATEGORY_EMBEDDING_SIZE)
                .build());
    canonicalTileEmbedding =
        addChildBlock(
            "canonicalTileEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(Tile.NUM_TILE_TYPES + 1)
                .setEmbeddingSize(hiddenSize)
                .build());
    roundFeatureProjection = addChildBlock("roundFeatureProjection", hiddenLinear());
    playerFeatureProjection = addChildBlock("playerFeatureProjection", hiddenLinear());
    tileFeatureProjection = addChildBlock("tileFeatureProjection", hiddenLinear());
    tileRelationEncoder =
        addChildBlock("tileRelationEncoder", new EpsilonTileRelationEncoder(hiddenSize));
    riverEventProjection = addChildBlock("riverEventProjection", hiddenLinear());
    meldFeatureProjection = addChildBlock("meldFeatureProjection", hiddenLinear());
    entityTokenLayerNorm =
        addChildBlock("entityTokenLayerNorm", LayerNorm.builder().axis(2).build());
    playerMemoryEncoder =
        addChildBlock("playerMemoryEncoder", new EpsilonPlayerMemoryEncoder(hiddenSize));
    strategicContextEncoder =
        addChildBlock("strategicContextEncoder", new EpsilonStrategicContextEncoder(hiddenSize));
  }

  /** 凍結済み戦略文脈層の積み重ねを、このデバイスパイプライン専用の永続Fusion 実行計画へ関連付ける。 */
  EpsilonStrategicContextFusionExecution newStrategicInferenceExecution(
      NDManager manager,
      ParameterStore parameterStore,
      DataType expectedDataType,
      int maximumBatch,
      int executionSlots) {
    return EpsilonStrategicContextFusionExecution.create(
        manager,
        strategicContextEncoder.fusionParameters(parameterStore, manager),
        expectedDataType,
        maximumBatch,
        executionSlots);
  }

  /** 凍結済みプレイヤーごとの履歴表現エンコーダーを、このデバイスパイプライン専用の疎な永続Fusion 実行計画へ関連付ける。 */
  EpsilonPlayerMemoryFusionExecution newPlayerMemoryInferenceExecution(
      NDManager manager,
      ParameterStore parameterStore,
      DataType expectedDataType,
      DataType inputDataType,
      int maximumBatch,
      int executionSlots) {
    return EpsilonPlayerMemoryFusionExecution.create(
        manager,
        playerMemoryEncoder.fusionParameters(manager, parameterStore, entityTokenLayerNorm),
        expectedDataType,
        inputDataType,
        maximumBatch,
        executionSlots);
  }

  /** 凍結済み牌種間の関係層の積み重ねを、このデバイスパイプライン専用の永続Fusion 実行計画へ関連付ける。 */
  EpsilonTileRelationFusionExecution newTileRelationInferenceExecution(
      NDManager manager,
      ParameterStore parameterStore,
      DataType expectedDataType,
      int maximumBatch,
      int executionSlots) {
    return EpsilonTileRelationFusionExecution.create(
        manager,
        tileRelationEncoder.fusionParameters(parameterStore, manager),
        expectedDataType,
        maximumBatch,
        executionSlots);
  }

  /**
   * 標準形式の状態連続バッファを構成要素トークンへ展開し、共有メモリを返す。
   *
   * @param parameterStore パラメーター取得先
   * @param stateCategories {@code [batch, STATE_INT_COUNT]} カテゴリ値状態テンソル
   * @param stateNumerics {@code [batch, STATE_FLOAT_COUNT]} 数値状態テンソル
   * @param training 子ブロックへ伝播する学習モード。学習固有の層 を追加した場合にもこの値を使う
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 局トークン、全構成要素、34牌種、4人別公開メモリとマスク
   */
  public EncodedMemory encodeMemory(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return encodeMemory(
        parameterStore, stateCategories, stateNumerics, null, training, runtimeParameters);
  }

  /**
   * 標準形式の状態連続バッファと有効要素のみのプレイヤーごとの履歴表現インデックスから共有局・プレイヤー・牌などの特徴表現を作る。
   *
   * @param parameterStore パラメーター取得先
   * @param stateCategories {@code [batch, STATE_INT_COUNT]} カテゴリ値状態テンソル
   * @param stateNumerics {@code [batch, STATE_FLOAT_COUNT]} 数値状態テンソル
   * @param playerMemoryPresentIndices 存在するプレイヤーごとの履歴表現行の有効要素のみのインデックス。未指定ならマスクから求める
   * @param training 子ブロックへ伝播する学習モード
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 局トークン、全構成要素、34牌種、4人別公開メモリとマスク
   */
  public EncodedMemory encodeMemory(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      NDArray playerMemoryPresentIndices,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return encodeMemory(
        parameterStore,
        stateCategories,
        stateNumerics,
        playerMemoryPresentIndices,
        null,
        null,
        null,
        training,
        runtimeParameters);
  }

  EncodedMemory encodeMemory(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      NDArray playerMemoryPresentIndices,
      EpsilonPlayerMemoryFusionExecution.Forward playerMemoryForward,
      EpsilonTileRelationFusionExecution.Forward tileRelationForward,
      EpsilonStrategicContextFusionExecution.Forward strategicForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      return encodeMemoryInternal(
          parameterStore,
          stateCategories,
          stateNumerics,
          playerMemoryPresentIndices,
          playerMemoryForward,
          tileRelationForward,
          strategicForward,
          true,
          runtimeParameters);
    }
    NDManager outputManager = stateCategories.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(stateCategories, stateNumerics);
      if (playerMemoryPresentIndices != null) {
        scope.tempAttachAll(playerMemoryPresentIndices);
      }
      EncodedMemory memory =
          encodeMemoryInternal(
              parameterStore,
              stateCategories,
              stateNumerics,
              playerMemoryPresentIndices,
              playerMemoryForward,
              tileRelationForward,
              strategicForward,
              false,
              runtimeParameters);
      outputManager.attachAll(
          memory.roundEmbedding(),
          memory.entityEmbeddings(),
          memory.entityMask(),
          memory.tileEmbeddings(),
          memory.playerMemory(),
          memory.playerMemoryMask());
      if (memory.tileProjectionEmbeddings() != memory.tileEmbeddings()) {
        outputManager.attachAll(memory.tileProjectionEmbeddings());
      }
      return memory;
    }
  }

  private EncodedMemory encodeMemoryInternal(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      NDArray playerMemoryPresentIndices,
      EpsilonPlayerMemoryFusionExecution.Forward playerMemoryForward,
      EpsilonTileRelationFusionExecution.Forward tileRelationForward,
      EpsilonStrategicContextFusionExecution.Forward strategicForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long rowCount = stateCategories.getShape().get(0);
    requireStateShape(
        stateCategories, "stateCategories", rowCount, DecisionInputSchema.STATE_INT_COUNT);
    requireStateShape(
        stateNumerics, "stateNumerics", rowCount, DecisionInputSchema.STATE_FLOAT_COUNT);
    NDArray categoricalFeatureValues = stateCategories.stopGradient();
    NDArray stateEmbeddingTable =
        stateFeatureEmbedding.getValue(parameterStore, stateCategories.getDevice(), training);
    boolean useStateEntityFeaturePack =
        usesStateEntityFeaturePack(
            stateCategories.getDevice(),
            stateNumerics.getDataType(),
            stateEmbeddingTable.getDataType());
    NDArray categoricalFeatureEmbeddings =
        useStateEntityFeaturePack
            ? null
            : embedStateCategories(
                categoricalFeatureValues, stateCategoryOffsets, stateEmbeddingTable);
    NDArray numericFeatures = stateNumerics;

    int categoricalFeatureOffset = 0;
    int numericFeatureOffset = 0;
    NDArray roundEmbedding =
        projectEntityFeatures(
                roundFeatureProjection,
                parameterStore,
                categoricalFeatureValues,
                categoricalFeatureEmbeddings,
                stateCategoryOffsets,
                stateEmbeddingTable,
                useStateEntityFeaturePack,
                categoricalFeatureOffset,
                1,
                DecisionInputSchema.ROUND_INT_COUNT,
                numericFeatures,
                numericFeatureOffset,
                DecisionInputSchema.ROUND_FLOAT_COUNT,
                rowCount,
                training,
                runtimeParameters)
            .reshape(rowCount, hiddenSize);
    categoricalFeatureOffset += DecisionInputSchema.ROUND_INT_COUNT;
    numericFeatureOffset += DecisionInputSchema.ROUND_FLOAT_COUNT;

    NDArray playerEmbeddings =
        projectEntityFeatures(
            playerFeatureProjection,
            parameterStore,
            categoricalFeatureValues,
            categoricalFeatureEmbeddings,
            stateCategoryOffsets,
            stateEmbeddingTable,
            useStateEntityFeaturePack,
            categoricalFeatureOffset,
            GameState.NUM_PLAYERS,
            DecisionInputSchema.PLAYER_INT_STRIDE,
            numericFeatures,
            numericFeatureOffset,
            DecisionInputSchema.PLAYER_FLOAT_STRIDE,
            rowCount,
            training,
            runtimeParameters);
    categoricalFeatureOffset += GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_INT_STRIDE;
    numericFeatureOffset += GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_FLOAT_STRIDE;

    NDArray tileEmbeddings =
        projectEntityFeatures(
            tileFeatureProjection,
            parameterStore,
            categoricalFeatureValues,
            categoricalFeatureEmbeddings,
            stateCategoryOffsets,
            stateEmbeddingTable,
            useStateEntityFeaturePack,
            categoricalFeatureOffset,
            Tile.NUM_TILE_TYPES,
            DecisionInputSchema.TILE_INT_STRIDE,
            numericFeatures,
            numericFeatureOffset,
            DecisionInputSchema.TILE_FLOAT_STRIDE,
            rowCount,
            training,
            runtimeParameters);
    tileEmbeddings =
        addFixedCanonicalTileIdentities(parameterStore, tileEmbeddings, rowCount, training);
    tileEmbeddings =
        training
            ? tileRelationEncoder.encode(parameterStore, tileEmbeddings, true, runtimeParameters)
            : tileRelationEncoder.encodeOwnedInference(
                parameterStore, tileEmbeddings, tileRelationForward, runtimeParameters);
    categoricalFeatureOffset += Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_INT_STRIDE;
    numericFeatureOffset += Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_FLOAT_STRIDE;

    NDArray riverCategoryValues =
        categoricalFeatureValues
            .get(
                ":, {}:{}",
                categoricalFeatureOffset,
                categoricalFeatureOffset + RIVER_TOKEN_COUNT * DecisionInputSchema.RIVER_INT_STRIDE)
            .reshape(rowCount, RIVER_TOKEN_COUNT, DecisionInputSchema.RIVER_INT_STRIDE);
    NDArray riverEventEmbeddings =
        projectEntityFeatures(
            riverEventProjection,
            parameterStore,
            categoricalFeatureValues,
            categoricalFeatureEmbeddings,
            stateCategoryOffsets,
            stateEmbeddingTable,
            useStateEntityFeaturePack,
            categoricalFeatureOffset,
            RIVER_TOKEN_COUNT,
            DecisionInputSchema.RIVER_INT_STRIDE,
            numericFeatures,
            numericFeatureOffset,
            DecisionInputSchema.RIVER_FLOAT_STRIDE,
            rowCount,
            training,
            runtimeParameters);
    NDArray riverEventPresentValues =
        riverCategoryValues.get("...,{}", DecisionInputSchema.RiverInt.PRESENT.ordinal());
    NDArray riverTileIds =
        riverCategoryValues.get("...,{}", DecisionInputSchema.RiverInt.TILE.ordinal());
    NDArray inferenceCanonicalEmbeddingTable =
        training || !riverEventEmbeddings.getDevice().isGpu()
            ? null
            : canonicalTileEmbedding.getValue(
                parameterStore, riverEventEmbeddings.getDevice(), false);
    NDArray riverEventPresentMask;
    if (!usesOwnedCanonicalTileEpilogue(
        riverEventEmbeddings, inferenceCanonicalEmbeddingTable, numericFeatures.getDataType())) {
      riverEventPresentMask =
          riverEventPresentValues.toType(numericFeatures.getDataType(), false).stopGradient();
      riverEventEmbeddings =
          riverEventEmbeddings
              .add(embedCanonicalTiles(parameterStore, riverTileIds, training, runtimeParameters))
              .mul(riverEventPresentMask.expandDims(2));
    } else {
      riverEventPresentMask =
          addCanonicalTileResidualToOwnedTokens(
              riverEventEmbeddings,
              new NDList(riverTileIds),
              inferenceCanonicalEmbeddingTable,
              riverEventPresentValues,
              EmbeddingReduction.SUM);
    }
    categoricalFeatureOffset += RIVER_TOKEN_COUNT * DecisionInputSchema.RIVER_INT_STRIDE;
    numericFeatureOffset += RIVER_TOKEN_COUNT * DecisionInputSchema.RIVER_FLOAT_STRIDE;

    NDArray meldCategoryValues =
        categoricalFeatureValues
            .get(
                ":, {}:{}",
                categoricalFeatureOffset,
                categoricalFeatureOffset + MELD_TOKEN_COUNT * DecisionInputSchema.MELD_INT_STRIDE)
            .reshape(rowCount, MELD_TOKEN_COUNT, DecisionInputSchema.MELD_INT_STRIDE);
    NDArray meldEmbeddings =
        projectEntityFeatures(
            meldFeatureProjection,
            parameterStore,
            categoricalFeatureValues,
            categoricalFeatureEmbeddings,
            stateCategoryOffsets,
            stateEmbeddingTable,
            useStateEntityFeaturePack,
            categoricalFeatureOffset,
            MELD_TOKEN_COUNT,
            DecisionInputSchema.MELD_INT_STRIDE,
            numericFeatures,
            numericFeatureOffset,
            DecisionInputSchema.MELD_FLOAT_STRIDE,
            rowCount,
            training,
            runtimeParameters);
    NDArray meldPresentValues =
        meldCategoryValues.get("...,{}", DecisionInputSchema.MeldInt.PRESENT.ordinal());
    NDArray meldBaseTileIds =
        meldCategoryValues.get("...,{}", DecisionInputSchema.MeldInt.BASE_TILE.ordinal());
    NDArray meldCalledTileIds =
        meldCategoryValues.get("...,{}", DecisionInputSchema.MeldInt.CALLED_TILE.ordinal());
    NDArray meldPresentMask;
    if (!usesOwnedCanonicalTileEpilogue(
        meldEmbeddings, inferenceCanonicalEmbeddingTable, numericFeatures.getDataType())) {
      meldPresentMask =
          meldPresentValues.toType(numericFeatures.getDataType(), false).stopGradient();
      NDArray meldIdentityCount =
          meldBaseTileIds
              .neq(DecisionInputSchema.PAD_ID)
              .toType(meldEmbeddings.getDataType(), false)
              .add(
                  meldCalledTileIds
                      .neq(DecisionInputSchema.PAD_ID)
                      .toType(meldEmbeddings.getDataType(), false))
              .maximum(1.0f)
              .expandDims(2)
              .stopGradient();
      NDArray meldTileIdentities =
          embedCanonicalTiles(parameterStore, meldBaseTileIds, training, runtimeParameters)
              .add(
                  embedCanonicalTiles(
                      parameterStore, meldCalledTileIds, training, runtimeParameters))
              .div(meldIdentityCount);
      meldEmbeddings = meldEmbeddings.add(meldTileIdentities).mul(meldPresentMask.expandDims(2));
    } else {
      meldPresentMask =
          addCanonicalTileResidualToOwnedTokens(
              meldEmbeddings,
              new NDList(meldBaseTileIds, meldCalledTileIds),
              inferenceCanonicalEmbeddingTable,
              meldPresentValues,
              EmbeddingReduction.MEAN_VALID);
    }

    // 局と牌をバッチ方向へ並べ、正規化後の34牌を連続したビューのままLinearへ渡す。
    NDArray normalizedRoundAndTileEmbeddings =
        NDArrays.concat(
            new NDList(
                roundEmbedding.reshape(rowCount, 1, hiddenSize),
                tileEmbeddings.reshape(rowCount * Tile.NUM_TILE_TYPES, 1, hiddenSize)),
            0);
    NDArray projectionRoundAndTileEmbeddings;
    if (normalizedRoundAndTileEmbeddings.getDevice().isGpu()
        && normalizedRoundAndTileEmbeddings.getDataType() != DataType.FLOAT32) {
      NDArray gamma =
          parameterStore.getValue(
              entityTokenLayerNorm.getParameters().get("gamma"),
              normalizedRoundAndTileEmbeddings.getDevice(),
              training);
      NDArray beta =
          parameterStore.getValue(
              entityTokenLayerNorm.getParameters().get("beta"),
              normalizedRoundAndTileEmbeddings.getDevice(),
              training);
      NDList normalizedOutputs =
          LayerNorm.layerNormAndCast(
              normalizedRoundAndTileEmbeddings,
              new Shape(hiddenSize),
              gamma,
              beta,
              LAYER_NORM_EPSILON,
              normalizedRoundAndTileEmbeddings.getDataType());
      normalizedRoundAndTileEmbeddings = normalizedOutputs.get(0);
      projectionRoundAndTileEmbeddings = normalizedOutputs.get(1);
    } else {
      normalizedRoundAndTileEmbeddings =
          entityTokenLayerNorm
              .forward(
                  parameterStore,
                  new NDList(normalizedRoundAndTileEmbeddings),
                  training,
                  runtimeParameters)
              .singletonOrThrow();
      projectionRoundAndTileEmbeddings = normalizedRoundAndTileEmbeddings;
    }
    NDArray normalizedRoundEmbedding = normalizedRoundAndTileEmbeddings.get("0:{},0,:", rowCount);
    tileEmbeddings =
        normalizedRoundAndTileEmbeddings
            .get("{}:,0,:", rowCount)
            .reshape(rowCount, Tile.NUM_TILE_TYPES, hiddenSize);
    NDArray tileProjectionEmbeddings =
        projectionRoundAndTileEmbeddings == normalizedRoundAndTileEmbeddings
            ? tileEmbeddings
            : projectionRoundAndTileEmbeddings
                .get("{}:,0,:", rowCount)
                .reshape(rowCount, Tile.NUM_TILE_TYPES, hiddenSize);
    NDArray alwaysPresentMask =
        numericFeatures
            .getManager()
            .ones(
                new Shape(1 + GameState.NUM_PLAYERS + Tile.NUM_TILE_TYPES),
                normalizedRoundAndTileEmbeddings.getDataType())
            .reshape(1, -1)
            .broadcast(rowCount, 1 + GameState.NUM_PLAYERS + Tile.NUM_TILE_TYPES);
    NDArray activeTokenMask =
        NDArrays.concat(new NDList(alwaysPresentMask, riverEventPresentMask, meldPresentMask), 1)
            .stopGradient();
    NDArray playerMemoryMask =
        buildPlayerMemoryMask(riverEventPresentMask, meldPresentMask, rowCount);
    NDArray playerTokens = playerEmbeddings.reshape(rowCount, GameState.NUM_PLAYERS, 1, hiddenSize);
    NDArray riverTokens =
        riverEventEmbeddings.reshape(
            rowCount,
            GameState.NUM_PLAYERS,
            DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER,
            hiddenSize);
    NDArray meldTokens =
        meldEmbeddings.reshape(
            rowCount, GameState.NUM_PLAYERS, DecisionInputSchema.MAX_MELDS_PER_PLAYER, hiddenSize);
    NDArray playerMemory;
    if (playerMemoryForward == null) {
      playerMemory = NDArrays.concat(new NDList(playerTokens, riverTokens, meldTokens), 2);
      playerMemory =
          playerMemoryEncoder.encode(
              parameterStore,
              playerMemory,
              playerMemoryMask,
              playerMemoryPresentIndices,
              entityTokenLayerNorm,
              training,
              runtimeParameters);
    } else {
      if (training || playerMemoryPresentIndices == null) {
        throw new IllegalArgumentException(
            "player-memory Fusion requires frozen inference and supplied indices");
      }
      playerMemory =
          playerMemoryForward.encode(
              playerTokens, riverTokens, meldTokens, playerMemoryPresentIndices);
    }
    NDList memorySections =
        playerMemory.split(new long[] {1, 1 + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER}, 2);
    EpsilonStrategicContextEncoder.StrategicContext strategicContext =
        strategicContextEncoder.encode(
            parameterStore,
            normalizedRoundEmbedding,
            tileEmbeddings,
            tileProjectionEmbeddings,
            memorySections.get(0).reshape(rowCount, GameState.NUM_PLAYERS, hiddenSize),
            training,
            strategicForward,
            runtimeParameters);
    normalizedRoundEmbedding = strategicContext.roundEmbedding();
    NDArray entityTokenEmbeddings =
        NDArrays.concat(
            new NDList(
                normalizedRoundEmbedding.reshape(rowCount, 1, hiddenSize),
                strategicContext.playerSummaries(),
                tileEmbeddings,
                memorySections.get(1).reshape(rowCount, RIVER_TOKEN_COUNT, hiddenSize),
                memorySections.get(2).reshape(rowCount, MELD_TOKEN_COUNT, hiddenSize)),
            1);
    // 方策は席別メモリも使う。学習時だけ新しい要約を連結し、公開トークンの分割結果を再利用する。
    if (training) {
      playerMemory =
          NDArrays.concat(
              new NDList(
                  strategicContext.playerSummaries().expandDims(2),
                  memorySections.get(1),
                  memorySections.get(2)),
              2);
    } else {
      playerMemory.set(new NDIndex(":,:,0,:"), strategicContext.playerSummaries());
    }

    return new EncodedMemory(
        normalizedRoundEmbedding,
        entityTokenEmbeddings,
        activeTokenMask,
        tileEmbeddings,
        tileProjectionEmbeddings,
        playerMemory,
        playerMemoryMask);
  }

  /** 各プレイヤーの要約トークンを必ず有効にし、その後ろへ河・面子の存在マスクを並べる。 */
  static NDArray buildPlayerMemoryMask(
      NDArray riverEventPresentMask, NDArray meldPresentMask, long rowCount) {
    NDManager manager = riverEventPresentMask.getManager();
    return NDArrays.concat(
            new NDList(
                manager.ones(
                    new Shape(rowCount, GameState.NUM_PLAYERS, 1),
                    riverEventPresentMask.getDataType()),
                riverEventPresentMask.reshape(
                    rowCount,
                    GameState.NUM_PLAYERS,
                    DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER),
                meldPresentMask.reshape(
                    rowCount, GameState.NUM_PLAYERS, DecisionInputSchema.MAX_MELDS_PER_PLAYER)),
            2)
        .stopGradient();
  }

  /**
   * 元の状態カテゴリへフィールド名前空間オフセットを加え、共有表を中間ID テンソルなしで参照する。
   *
   * <p>未対応エンジンではDJLの可搬な加算後埋め込みへ代替処理する。返却データ型は{@code table}と同じで、学習時は通常の密な埋め込み 勾配を保持する。
   *
   * @param rawIds オフセット適用前のカテゴリ ID
   * @param offsets {@code rawIds}へブロードキャストするフィールド名前空間オフセット
   * @param table 共有状態埋め込み表
   * @return {@code rawIds + offsets}が参照する埋め込み
   */
  static NDArray embedStateCategories(NDArray rawIds, NDArray offsets, NDArray table) {
    return NDArrays.embeddingWithOffsets(rawIds, offsets, table);
  }

  /** GPUで埋め込み表と数値入力のデータ型が一致するときは構成要素特徴量を直接連結する。 */
  static boolean usesStateEntityFeaturePack(
      Device device, DataType numericDataType, DataType embeddingDataType) {
    return device.isGpu() && numericDataType == embeddingDataType;
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    stateCategoryOffsets =
        manager
            .create(DecisionCategoryLayout.stateOffsets())
            .reshape(1, DecisionInputSchema.STATE_INT_COUNT);
    stateFeatureEmbedding.initialize(
        manager, dataType, new Shape(-1, DecisionInputSchema.STATE_INT_COUNT));
    canonicalTileEmbedding.initialize(manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES));
    roundFeatureProjection.initialize(
        manager,
        dataType,
        new Shape(
            -1,
            DecisionInputSchema.ROUND_INT_COUNT * CATEGORY_EMBEDDING_SIZE
                + DecisionInputSchema.ROUND_FLOAT_COUNT));
    playerFeatureProjection.initialize(
        manager,
        dataType,
        entityShape(
            DecisionInputSchema.PLAYER_INT_STRIDE, DecisionInputSchema.PLAYER_FLOAT_STRIDE));
    tileFeatureProjection.initialize(
        manager,
        dataType,
        entityShape(DecisionInputSchema.TILE_INT_STRIDE, DecisionInputSchema.TILE_FLOAT_STRIDE));
    tileRelationEncoder.initialize(
        manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize));
    riverEventProjection.initialize(
        manager,
        dataType,
        entityShape(DecisionInputSchema.RIVER_INT_STRIDE, DecisionInputSchema.RIVER_FLOAT_STRIDE));
    meldFeatureProjection.initialize(
        manager,
        dataType,
        entityShape(DecisionInputSchema.MELD_INT_STRIDE, DecisionInputSchema.MELD_FLOAT_STRIDE));
    entityTokenLayerNorm.initialize(
        manager, dataType, new Shape(-1, ENTITY_TOKEN_COUNT, hiddenSize));
    playerMemoryEncoder.initialize(
        manager,
        dataType,
        new Shape(-1, GameState.NUM_PLAYERS, PLAYER_MEMORY_TOKEN_COUNT, hiddenSize),
        new Shape(-1, GameState.NUM_PLAYERS, PLAYER_MEMORY_TOKEN_COUNT));
    strategicContextEncoder.initialize(
        manager,
        dataType,
        new Shape(-1, hiddenSize),
        new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize),
        new Shape(-1, GameState.NUM_PLAYERS, PLAYER_MEMORY_TOKEN_COUNT, hiddenSize));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    throw new UnsupportedOperationException("Use encodeMemory with typed state tensors");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {
      new Shape(-1, hiddenSize),
      new Shape(-1, ENTITY_TOKEN_COUNT, hiddenSize),
      new Shape(-1, ENTITY_TOKEN_COUNT),
      new Shape(-1, Tile.NUM_TILE_TYPES, hiddenSize),
      new Shape(-1, GameState.NUM_PLAYERS, PLAYER_MEMORY_TOKEN_COUNT, hiddenSize),
      new Shape(-1, GameState.NUM_PLAYERS, PLAYER_MEMORY_TOKEN_COUNT)
    };
  }

  private Linear hiddenLinear() {
    return Linear.builder().setUnits(hiddenSize).build();
  }

  private NDArray embedCanonicalTiles(
      ParameterStore parameterStore,
      NDArray storedTileIds,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray storedIds = storedTileIds.stopGradient();
    NDArray tileIds = storedIds.toType(DataType.INT32, false);
    NDArray embeddings =
        canonicalTileEmbedding
            .forward(parameterStore, new NDList(tileIds), training, runtimeParameters)
            .singletonOrThrow();
    NDArray present =
        storedIds
            .neq(DecisionInputSchema.PAD_ID)
            .toType(embeddings.getDataType(), false)
            .expandDims(tileIds.getShape().dimension())
            .stopGradient();
    return embeddings.mul(present);
  }

  /**
   * 推論用に所有している射影出力へ共通形式の牌識別情報を直接加え、パディングトークンを同時に0化する。
   *
   * <p>参照、PAD除外、面子の有効牌平均、存在するマスクを一つのエンジン演算へ渡す。学習時は自動微分境界を変えないため、 {@link
   * #embedCanonicalTiles(ParameterStore, NDArray, boolean, PairList)} の逐次実行の式を使う。
   */
  private NDArray addCanonicalTileResidualToOwnedTokens(
      NDArray tokenEmbeddings,
      NDList storedTileIds,
      NDArray canonicalEmbeddingTable,
      NDArray presentValues,
      EmbeddingReduction reduction) {
    return NDArrays.addMaskedEmbeddingResidualToOwnedTokens(
        tokenEmbeddings,
        storedTileIds,
        canonicalEmbeddingTable,
        presentValues,
        DecisionInputSchema.PAD_ID,
        reduction);
  }

  /** GPU推論かつ数値入力、射影出力、埋め込みパラメーターが同じデータ型の場合だけ使用する。 */
  private static boolean usesOwnedCanonicalTileEpilogue(
      NDArray tokenEmbeddings, NDArray canonicalEmbeddingTable, DataType numericDataType) {
    return canonicalEmbeddingTable != null
        && tokenEmbeddings.getDataType() == numericDataType
        && tokenEmbeddings.getDataType() == canonicalEmbeddingTable.getDataType();
  }

  /** 固定34牌種トークンへ、パディング行を除く共通形式の埋め込み表をビューのまま加える。 */
  private NDArray addFixedCanonicalTileIdentities(
      ParameterStore parameterStore, NDArray tileEmbeddings, long rowCount, boolean training) {
    NDManager outputManager = tileEmbeddings.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(tileEmbeddings);
      NDArray canonicalEmbeddingTable =
          canonicalTileEmbedding.getValue(parameterStore, tileEmbeddings.getDevice(), training);
      scope.tempAttachAll(canonicalEmbeddingTable);
      NDArray fixedTileIdentities =
          canonicalEmbeddingTable
              .get("1:{},:", Tile.NUM_TILE_TYPES + 1)
              .expandDims(0)
              .broadcast(rowCount, Tile.NUM_TILE_TYPES, hiddenSize);
      NDArray enrichedTileEmbeddings =
          training
              ? tileEmbeddings.add(fixedTileIdentities)
              : tileEmbeddings.addi(fixedTileIdentities);
      outputManager.attachAll(enrichedTileEmbeddings);
      return enrichedTileEmbeddings;
    }
  }

  private static Shape entityShape(int intStride, int floatStride) {
    return new Shape(-1, intStride * CATEGORY_EMBEDDING_SIZE + floatStride);
  }

  private static NDArray projectEntityFeatures(
      Linear projection,
      ParameterStore parameterStore,
      NDArray categoricalFeatureValues,
      NDArray categoricalFeatureEmbeddings,
      NDArray categoryOffsets,
      NDArray embeddingTable,
      boolean useStateEntityFeaturePack,
      int categoricalFeatureOffset,
      int entityCount,
      int categoricalFeatureStride,
      NDArray numericFeatures,
      int numericFeatureOffset,
      int floatStride,
      long rowCount,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray entityFeatures;
    if (useStateEntityFeaturePack) {
      entityFeatures =
          packStateEntityFeatures(
              categoricalFeatureValues,
              categoryOffsets,
              embeddingTable,
              categoricalFeatureOffset,
              entityCount,
              categoricalFeatureStride,
              numericFeatures,
              numericFeatureOffset,
              floatStride,
              rowCount);
    } else {
      NDArray categoricalEntityFeatures =
          categoricalFeatureEmbeddings
              .get(
                  ":, {}:{}, :",
                  categoricalFeatureOffset,
                  categoricalFeatureOffset + entityCount * categoricalFeatureStride)
              .reshape(rowCount, entityCount, categoricalFeatureStride * CATEGORY_EMBEDDING_SIZE);
      NDArray numericEntityFeatures =
          numericFeatures
              .get(
                  ":, {}:{}",
                  numericFeatureOffset,
                  numericFeatureOffset + entityCount * floatStride)
              .reshape(rowCount, entityCount, floatStride);
      entityFeatures = categoricalEntityFeatures.concat(numericEntityFeatures, 2);
    }
    return applyLinear(projection, parameterStore, entityFeatures, training, runtimeParameters);
  }

  /** 状態の1 構成要素区間を先頭軸に間隔を持つビューのままオフセット埋め込みと数値末尾へ連結する。 */
  static NDArray packStateEntityFeatures(
      NDArray categoricalFeatureValues,
      NDArray categoryOffsets,
      NDArray embeddingTable,
      int categoricalFeatureOffset,
      int entityCount,
      int categoricalFeatureStride,
      NDArray numericFeatures,
      int numericFeatureOffset,
      int floatStride,
      long rowCount) {
    int categoricalFeatureEnd = categoricalFeatureOffset + entityCount * categoricalFeatureStride;
    int numericFeatureEnd = numericFeatureOffset + entityCount * floatStride;
    NDArray rawEntityFeatures =
        categoricalFeatureValues
            .get(":, {}:{}", categoricalFeatureOffset, categoricalFeatureEnd)
            .reshape(rowCount, entityCount, categoricalFeatureStride);
    NDArray entityOffsets =
        categoryOffsets
            .get(":, {}:{}", categoricalFeatureOffset, categoricalFeatureEnd)
            .reshape(1, entityCount, categoricalFeatureStride);
    NDArray numericEntityFeatures =
        numericFeatures
            .get(":, {}:{}", numericFeatureOffset, numericFeatureEnd)
            .reshape(rowCount, entityCount, floatStride);
    return NDArrays.embeddingFeaturePack(
        rawEntityFeatures, entityOffsets, embeddingTable, numericEntityFeatures);
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

  /**
   * 入力と同じ形状で {@code x * sigmoid(x)} を返す。
   *
   * @param value 活性化前テンソル
   * @return SiLU を適用したテンソル
   */
  public static NDArray silu(NDArray value) {
    return Activation.swish(value, 1.0f);
  }

  private static void requireStateShape(NDArray input, String label, long batch, int width) {
    Shape expected = new Shape(batch, width);
    if (!input.getShape().equals(expected)) {
      throw new IllegalArgumentException(
          label + " shape must be " + expected + ", got " + input.getShape());
    }
  }

  /**
   * 方策/価値の目的固有特徴量の集約へ渡す共有局・プレイヤー・牌などの特徴表現。
   *
   * @param roundEmbedding 局全体の固定長表現。形状は {@code [batch, hidden]}
   * @param entityEmbeddings プレイヤー・牌・河・面子を席順と時系列を保って並べた表現。形状は {@code [batch, entity, hidden]}
   * @param entityMask {@code entityEmbeddings} の有効トークンマスク。形状は {@code [batch, entity]}
   * @param tileEmbeddings 34牌種それぞれの周囲の情報を反映した表現。形状は {@code [batch, 34, hidden]}
   * @param tileProjectionEmbeddings 牌種表現をLinearへ渡す低精度ビュー。CPUでは {@code tileEmbeddings} と同じ参照
   * @param playerMemory プレイヤートークンと各プレイヤーの河・面子を席別に保持したメモリ。形状は {@code [batch, 4, playerMemory,
   *     hidden]}
   * @param playerMemoryMask {@code playerMemory} の有効トークンマスク。形状は {@code [batch, 4, playerMemory]}
   */
  public record EncodedMemory(
      NDArray roundEmbedding,
      NDArray entityEmbeddings,
      NDArray entityMask,
      NDArray tileEmbeddings,
      NDArray tileProjectionEmbeddings,
      NDArray playerMemory,
      NDArray playerMemoryMask) {

    /** 低精度ビューを分けない既存の呼び出し箇所用の同じ結果となるコンストラクター。 */
    public EncodedMemory(
        NDArray roundEmbedding,
        NDArray entityEmbeddings,
        NDArray entityMask,
        NDArray tileEmbeddings,
        NDArray playerMemory,
        NDArray playerMemoryMask) {
      this(
          roundEmbedding,
          entityEmbeddings,
          entityMask,
          tileEmbeddings,
          tileEmbeddings,
          playerMemory,
          playerMemoryMask);
    }

    /** 全学習目的から切り離した同値メモリを返す。 */
    EncodedMemory stopGradient() {
      NDArray detachedTiles = tileEmbeddings.stopGradient();
      NDArray detachedProjectionTiles =
          tileProjectionEmbeddings == tileEmbeddings
              ? detachedTiles
              : tileProjectionEmbeddings.stopGradient();
      return new EncodedMemory(
          roundEmbedding.stopGradient(),
          entityEmbeddings.stopGradient(),
          entityMask,
          detachedTiles,
          detachedProjectionTiles,
          playerMemory.stopGradient(),
          playerMemoryMask);
    }

    /**
     * 目的固有の集約表現を付与して方策出力層入力を作る。
     *
     * @param stateEmbedding 方策特徴量の集約が作った {@code [batch, hidden]} 表現
     * @return 牌・プレイヤーごとの履歴表現を共有する方策入力
     */
    public EncodedState withStateEmbedding(NDArray stateEmbedding) {
      return new EncodedState(
          stateEmbedding, tileEmbeddings, tileProjectionEmbeddings, playerMemory, playerMemoryMask);
    }
  }

  /**
   * 方策出力層へ渡す局面表現と共有メモリ。
   *
   * @param stateEmbedding 方策専用特徴量の集約が集約した局面表現。形状は {@code [batch, hidden]}
   * @param tileEmbeddings 34牌種それぞれの周囲の情報を反映した表現。形状は {@code [batch, 34, hidden]}
   * @param tileProjectionEmbeddings 牌種表現をLinearへ渡す低精度ビュー。CPUでは {@code tileEmbeddings} と同じ参照
   * @param playerMemory プレイヤートークンと各プレイヤーの河・面子を席別に保持したメモリ。形状は {@code [batch, 4, playerMemory,
   *     hidden]}
   * @param playerMemoryMask {@code playerMemory} の有効トークンマスク。形状は {@code [batch, 4, playerMemory]}
   */
  public record EncodedState(
      NDArray stateEmbedding,
      NDArray tileEmbeddings,
      NDArray tileProjectionEmbeddings,
      NDArray playerMemory,
      NDArray playerMemoryMask) {

    /** 低精度ビューを分けない既存の呼び出し箇所用の同じ結果となるコンストラクター。 */
    public EncodedState(
        NDArray stateEmbedding,
        NDArray tileEmbeddings,
        NDArray playerMemory,
        NDArray playerMemoryMask) {
      this(stateEmbedding, tileEmbeddings, tileEmbeddings, playerMemory, playerMemoryMask);
    }
  }
}

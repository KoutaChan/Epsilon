package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonLinear;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.config.settings.DecisionInferenceFusionMode;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.input.DecisionCategoryLayout;
import com.epsilon.nano.ai.decision.input.DecisionGroupedInferenceBatch;
import com.epsilon.nano.ai.decision.input.DecisionInferenceInputs;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.nano.ai.decision.input.DecisionPolicyInputs;
import com.epsilon.nano.ai.decision.input.DecisionTransitionInputs;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionCandidatePrefixExecution;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionPlayerTileContextExecution;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import java.util.LinkedHashMap;
import java.util.Objects;
import java.util.function.BiConsumer;

/**
 * Nanoが使う状態・行動・遷移の候補表現。方策分岐や出力の目的は呼出元が所有する。
 *
 * <p>所有者のBlockへ重みを登録するため、既存Nano checkpointの平坦な名前も維持できる。
 */
public final class EpsilonDecisionCandidateEncoder {
  private static final int CATEGORY_EMBEDDING = 8;
  private static final int WAIT_YAKU_EMBEDDING = 16;
  private static final int TRANSITION_TILE_ATTENTION_HEADS = 4;
  private static final int TRANSITION_TILE_KEY_SIZE = 8;
  private static final int TRANSITION_TILE_KEY_WIDTH =
      TRANSITION_TILE_ATTENTION_HEADS * TRANSITION_TILE_KEY_SIZE;
  private static final int WAIT_YAKU_CODE_COUNT = 1 << DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK;
  private static final int MAX_CONTEXT_WIDTH = 64;
  private final int hiddenSize;
  private final TransitionPooling transitionPooling;
  private final int contextWidth;
  private final IdEmbedding actionFeatureEmbedding;
  private final IdEmbedding transitionTileRelationEmbedding;
  private final Linear transitionWaitYakuProjection;
  private final Linear actionFeatureProjection;
  private final Linear transitionFeatureProjection;
  private final Linear transitionTileKeyValueProjection;
  private final Linear transitionTileQueryProjection;
  private final Linear transitionWaitKeyValueProjection;
  private final Linear playerTileQueryProjection;
  private final Linear playerMemoryKeyValueProjection;
  final EpsilonPlayerTileContextMixer playerTileContextMixer;
  final EpsilonFeatureFusion transitionFusion;
  private final LayerNorm transitionLayerNorm;
  private final EpsilonDiscardCandidateScorer discardCandidateScorer;
  final EpsilonFeatureFusion candidateFusion;
  private final LayerNorm candidateLayerNorm;
  private NDArray actionCategoryOffsets;
  private NDArray transitionCategoryOffsets;
  private NDArray waitYakuBitDivisors;
  private NDArray inferenceWaitYakuLookup;
  private final LinkedHashMap<String, Block> components = new LinkedHashMap<>();

  public EpsilonDecisionCandidateEncoder(int hiddenSize) {
    this(hiddenSize, TransitionPooling.SELECTED_DISCARD_POLICY);
  }

  public EpsilonDecisionCandidateEncoder(int hiddenSize, TransitionPooling transitionPooling) {
    if (hiddenSize <= 0 || hiddenSize % TRANSITION_TILE_ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be positive and divisible by attention heads");
    }
    this.hiddenSize = hiddenSize;
    this.transitionPooling = Objects.requireNonNull(transitionPooling);
    contextWidth = Math.min(hiddenSize, MAX_CONTEXT_WIDTH);
    actionFeatureEmbedding =
        component(
            "actionFeatureEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(DecisionCategoryLayout.ACTION_DICTIONARY_SIZE)
                .setEmbeddingSize(CATEGORY_EMBEDDING)
                .build());
    transitionTileRelationEmbedding =
        component(
            "transitionTileRelationEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(DecisionInputSchema.ACTION_TRANSITION_TILE_DICTIONARY_SIZE)
                .setEmbeddingSize(TRANSITION_TILE_KEY_WIDTH + contextWidth)
                .build());
    transitionWaitYakuProjection =
        component(
            "transitionWaitYakuProjection",
            Linear.builder().setUnits(WAIT_YAKU_EMBEDDING).optBias(false).build());
    actionFeatureProjection = component("actionFeatureProjection", hiddenLinear());
    transitionFeatureProjection = component("transitionFeatureProjection", hiddenLinear());
    transitionTileKeyValueProjection =
        component(
            "transitionTileKeyValueProjection",
            Linear.builder().setUnits(TRANSITION_TILE_KEY_WIDTH + contextWidth).build());
    transitionTileQueryProjection =
        component(
            "transitionTileQueryProjection",
            Linear.builder().setUnits(TRANSITION_TILE_KEY_WIDTH).build());
    transitionWaitKeyValueProjection =
        component(
            "transitionWaitKeyValueProjection",
            Linear.builder().setUnits(TRANSITION_TILE_KEY_WIDTH + contextWidth).build());
    playerTileQueryProjection =
        component(
            "playerTileQueryProjection",
            Linear.builder().setUnits(contextWidth).optBias(false).build());
    playerMemoryKeyValueProjection =
        component(
            "playerMemoryKeyValueProjection",
            Linear.builder().setUnits(contextWidth * 2L).optBias(false).build());
    playerTileContextMixer =
        component("playerTileContextMixer", new EpsilonPlayerTileContextMixer(contextWidth));
    transitionFusion =
        component(
            "transitionFusion",
            new EpsilonFeatureFusion(
                hiddenSize, 4, hiddenSize, contextWidth, hiddenSize, contextWidth, hiddenSize));
    transitionLayerNorm = component("transitionLayerNorm", LayerNorm.builder().axis(3).build());
    discardCandidateScorer =
        component("discardCandidateScorer", new EpsilonDiscardCandidateScorer(hiddenSize));
    candidateFusion =
        component(
            "candidateFusion",
            new EpsilonFeatureFusion(
                hiddenSize, 4, hiddenSize, hiddenSize, hiddenSize, contextWidth, hiddenSize));
    candidateLayerNorm = component("candidateLayerNorm", LayerNorm.builder().axis(2).build());
  }

  public int hiddenSize() {
    return hiddenSize;
  }

  /** 方策ヘッドを作らず、候補表現が使う凍結推論計画だけを所有する。 */
  public DecisionCandidateExecution newInferenceExecution(
      NDManager manager,
      ParameterStore parameters,
      DecisionInferenceFusionSettings settings,
      DataType dataType,
      int maximumBatch,
      int executionSlots) {
    if (maximumBatch <= 0 || executionSlots <= 0)
      throw new IllegalArgumentException("Candidate inference capacity must be positive");
    DecisionPolicyCandidatePrefixExecution prefix = null;
    DecisionPolicyPlayerTileContextExecution playerTile = null;
    try {
      if (settings.candidatePrefix() == DecisionInferenceFusionMode.FUSION)
        prefix =
            new DecisionPolicyFusionCandidatePrefixExecution(
                manager, parameters, candidateFusion, dataType, maximumBatch, executionSlots);
      playerTile =
          settings.playerTileContext() == DecisionInferenceFusionMode.FUSION
              ? new DecisionPolicyFusionPlayerTileContextExecution(
                  manager,
                  parameters,
                  playerTileContextMixer,
                  dataType,
                  maximumBatch,
                  executionSlots)
              : DecisionPolicyPlayerTileContextExecution.eager(
                  playerTileContextMixer, executionSlots);
      return new DecisionCandidateExecution(
          candidateFusion,
          transitionFusion,
          prefix,
          playerTile,
          manager.getDevice().isGpu()
                  && (dataType == DataType.BFLOAT16 || dataType == DataType.FLOAT16)
              ? () ->
                  new DecisionPolicyAffineExecution.PackedTransitionProjection(
                      manager, parameters, transitionFusion, dataType, false)
              : null);
    } catch (RuntimeException | Error failure) {
      for (AutoCloseable resource : new AutoCloseable[] {playerTile, prefix}) {
        if (resource == null) continue;
        try {
          resource.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      throw failure;
    }
  }

  private <T extends Block> T component(String name, T block) {
    components.put(name, block);
    return block;
  }

  /** 呼出元のパラメーター階層へ表現の全ブロックを一度だけ登録する。 */
  public void registerWith(BiConsumer<String, Block> registrar) {
    components.forEach(registrar);
  }

  // Nanoの公開checkpointで方策埋め込みがこの二組の間にあるため、登録順序を保つ。
  void registerEmbeddings(BiConsumer<String, Block> registrar) {
    components.entrySet().stream()
        .limit(3)
        .forEach(e -> registrar.accept(e.getKey(), e.getValue()));
  }

  void registerContexts(BiConsumer<String, Block> registrar) {
    components.entrySet().stream().skip(3).forEach(e -> registrar.accept(e.getKey(), e.getValue()));
  }

  public void initialize(NDManager manager, DataType type) {
    initializeEmbeddings(manager, type);
    initializeContexts(manager, type);
  }

  void initializeEmbeddings(NDManager manager, DataType dataType) {
    actionCategoryOffsets =
        manager
            .create(DecisionCategoryLayout.actionOffsets())
            .reshape(1, 1, DecisionInputSchema.ACTION_INT_STRIDE);
    transitionCategoryOffsets =
        manager
            .create(DecisionCategoryLayout.transitionOffsets())
            .reshape(1, 1, 1, DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE);
    float[] bitDivisors = new float[DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK];
    for (int bit = 0; bit < bitDivisors.length; bit++) {
      bitDivisors[bit] = 1 << bit;
    }
    waitYakuBitDivisors = manager.create(bitDivisors);
    actionFeatureEmbedding.initialize(
        manager, dataType, new Shape(-1, -1, DecisionInputSchema.ACTION_INT_STRIDE));
    transitionTileRelationEmbedding.initialize(
        manager, dataType, new Shape(-1, -1, -1, DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT));
    transitionWaitYakuProjection.initialize(
        manager,
        dataType,
        new Shape(
            -1,
            DecisionInputSchema.WAIT_YAKU_CHUNKS_PER_WIN_TYPE
                * DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK));
  }

  void initializeContexts(NDManager manager, DataType dataType) {
    actionFeatureProjection.initialize(
        manager,
        dataType,
        new Shape(
            -1,
            DecisionInputSchema.ACTION_INT_STRIDE * CATEGORY_EMBEDDING
                + DecisionInputSchema.ACTION_FLOAT_STRIDE));
    transitionFeatureProjection.initialize(
        manager,
        dataType,
        new Shape(
            -1,
            DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE * CATEGORY_EMBEDDING
                + DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE));
    transitionTileKeyValueProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    transitionTileQueryProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    transitionWaitKeyValueProjection.initialize(
        manager,
        dataType,
        new Shape(
            -1,
            DecisionInputSchema.WaitWinType.values().length * WAIT_YAKU_EMBEDDING
                + DecisionInputSchema.ACTION_TRANSITION_WAIT_FLOAT_STRIDE));
    playerTileQueryProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    playerMemoryKeyValueProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    playerTileContextMixer.initialize(
        manager, dataType, new Shape(-1, GameState.NUM_PLAYERS, Tile.NUM_TILE_TYPES, contextWidth));
    transitionFusion.initialize(manager, dataType, new Shape(-1, hiddenSize));
    transitionLayerNorm.initialize(manager, dataType, new Shape(-1, -1, -1, hiddenSize));
    discardCandidateScorer.initialize(manager, dataType, new Shape(-1, hiddenSize));
    candidateFusion.initialize(manager, dataType, new Shape(-1, hiddenSize));
    candidateLayerNorm.initialize(manager, dataType, new Shape(-1, -1, hiddenSize));
  }

  public NDArray encode(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionNetworkInputs inputs,
      NDArray presentTransitions,
      boolean training,
      PairList<String, Object> parameters) {
    return encodeRepresentation(
        store,
        state,
        inputs,
        DecisionTransitionInputs.dense(inputs),
        presentTransitions,
        null,
        training,
        parameters,
        true);
  }

  /**
   * Encodes candidate representations without producing separate discard policy scores. A single
   * transition needs no pooling score; multi-transition aggregation retains its complete gradient.
   */
  public NDArray encodeRepresentation(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionNetworkInputs inputs,
      NDArray presentTransitions,
      boolean training,
      PairList<String, Object> parameters) {
    return encodeRepresentation(
        store,
        state,
        inputs,
        DecisionTransitionInputs.dense(inputs),
        presentTransitions,
        null,
        training,
        parameters,
        false);
  }

  /** 共有候補計画を借用して表現を推論する。出力利用権は呼出側が最終D2Hまで保持する。 */
  public NDArray encodeInferenceRepresentation(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionNetworkInputs inputs,
      NDArray presentTransitions,
      DecisionCandidateForward forward,
      PairList<String, Object> parameters) {
    return encodeRepresentation(
        store,
        state,
        inputs,
        DecisionTransitionInputs.dense(inputs),
        presentTransitions,
        forward,
        false,
        parameters,
        false);
  }

  /** 集約済みの推論入力から、捨て牌方策のスコアを作らず候補表現だけを返す。 */
  public NDArray encodeInferenceRepresentation(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionInferenceInputs inputs,
      NDArray presentTransitions,
      DecisionCandidateForward forward,
      PairList<String, Object> parameters) {
    return encodeRepresentation(
        store,
        state,
        inputs,
        DecisionTransitionInputs.compact(inputs),
        presentTransitions,
        forward,
        false,
        parameters,
        false);
  }

  /** 盤面と実在遷移を共有し、合法候補だけを元の行動順で一度に計算する。 */
  public NDArray encodeGroupedInferenceRepresentation(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionGroupedInferenceBatch batch,
      DecisionCandidateForward forward,
      PairList<String, Object> parameters) {
    var contexts = encodePlayerTileContexts(store, state, forward, false, parameters);
    NDArray sharedTransitions =
        encodeSharedTransitions(store, state, contexts.allPlayers(), batch, forward, parameters);
    NDList categories = new NDList(), numerics = new NDList(), pooled = new NDList();
    for (var group : batch.groups()) {
      var inputs = group.inputs();
      int rows = inputs.rowCount();
      int count = inputs.transitionCount();
      int actions = inputs.bucket().legalActionCapacity();
      int capacity = inputs.bucket().actionTransitionCapacity();
      NDArray present = group.transitionPresentIndices();
      NDArray presentMask =
          capacity == 1
              ? null
              : EpsilonMaskedRows.scatter(
                      present
                          .getManager()
                          .ones(new Shape(count, 1), state.stateEmbedding().getDataType()),
                      present,
                      (long) rows * actions * capacity)
                  .reshape(rows, actions, capacity)
                  .stopGradient();
      var transitions =
          poolCompactedTransitions(
              store,
              rows(sharedTransitions, group.transitionOffset(), count),
              inputs.transitionCategories(),
              present,
              presentMask,
              rows,
              actions,
              capacity,
              false,
              parameters,
              false);
      categories.add(gatherActions(inputs.actionCategories(), group.actionPresentIndices()));
      numerics.add(gatherActions(inputs.actionNumerics(), group.actionPresentIndices()));
      pooled.add(gatherActions(transitions.aggregatedEmbeddings(), group.actionPresentIndices()));
    }
    int count = batch.actionCount();
    NDArray actionCategories = concatenateGroups(categories);
    NDArray actionNumerics = concatenateGroups(numerics);
    NDArray mask =
        state
            .stateEmbedding()
            .getManager()
            .ones(new Shape(count, 1), state.stateEmbedding().getDataType());
    var masks = DecisionPolicyCandidates.candidateMasks(actionCategories, mask);
    NDArray features =
        encodeActionFeatures(store, actionCategories, actionNumerics, false, parameters, count, 1);
    NDArray projectedFeatures =
        EpsilonMahjongStateEncoder.silu(
            applyLinear(actionFeatureProjection, store, features, false, parameters));
    NDArray primaryTiles =
        DecisionPolicyTensorOps.gatherCompactedActionPrimaryTileEmbeddings(
            state.tileEmbeddings(), actionCategories, batch.actionBatchIds());
    NDArray rootContexts =
        DecisionPolicyTensorOps.composeCompactedRootPlayerContexts(
            contexts.byPlayer(),
            contexts.allPlayers(),
            actionCategories,
            masks,
            batch
                .stateCategories()
                .get(":,{}", DecisionInputSchema.RoundInt.EVENT_PLAYER_RELATIVE_SEAT.ordinal()),
            batch.stateCategories().get(":,{}", DecisionInputSchema.RoundInt.EVENT_TILE.ordinal()),
            batch.actionBatchIds());
    NDArray states = state.stateEmbedding().reshape(batch.rowCount(), 1, hiddenSize);
    NDArray transitions = concatenateGroups(pooled);
    NDArray candidates =
        forward == null
            ? EpsilonMahjongStateEncoder.silu(
                candidateFusion.fuseIndexedState(
                    store,
                    parameters,
                    states,
                    batch.actionBatchIds(),
                    projectedFeatures,
                    primaryTiles,
                    transitions,
                    rootContexts))
            : forward.compactCandidate(
                store,
                parameters,
                projectedFeatures,
                primaryTiles,
                transitions,
                rootContexts,
                states,
                batch.actionBatchIds());
    return candidateLayerNorm
        .forward(store, new NDList(candidates), false, parameters)
        .singletonOrThrow();
  }

  private static NDArray concatenateGroups(NDList groups) {
    return groups.size() == 1 ? groups.getFirst() : NDArrays.concat(groups, 0);
  }

  private static NDArray gatherActions(NDArray dense, NDArray indices) {
    long width = dense.getShape().get(2);
    return EpsilonMaskedRows.gather(dense.reshape(-1, width), indices)
        .reshape(indices.size(), 1, width);
  }

  /** 共有遷移の出力だけを残し、集約前の大きな中間テンソルはその場で解放する。 */
  private NDArray encodeSharedTransitions(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      NDArray playerContexts,
      DecisionGroupedInferenceBatch batch,
      DecisionCandidateForward forward,
      PairList<String, Object> parameters) {
    NDManager outputManager = state.stateEmbedding().getManager();
    var inputs = batch.transitions();
    try (var scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          state.stateEmbedding(),
          state.tileEmbeddings(),
          playerContexts,
          inputs.categories(),
          inputs.numerics(),
          inputs.tileCodes(),
          inputs.waitTileIds(),
          inputs.waitYakus(),
          inputs.waitScores(),
          batch.transitionBatchIds());
      tempAttachTileProjection(scope, state);
      NDArray output =
          encodeCompactTransitionEmbeddings(
              store,
              state,
              playerContexts,
              inputs,
              batch.transitionBatchIds(),
              forward,
              false,
              parameters);
      outputManager.attachAll(output);
      return output;
    }
  }

  private static NDArray rows(NDArray array, int offset, int count) {
    return array.get("{}:{}", offset, offset + count);
  }

  private NDArray encodeRepresentation(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionPolicyInputs inputs,
      DecisionTransitionInputs transitionInputs,
      NDArray presentTransitions,
      DecisionCandidateForward forward,
      boolean training,
      PairList<String, Object> parameters,
      boolean includeDiscardScores) {
    NDArray mask =
        inputs.legalActionMask().toType(state.stateEmbedding().getDataType(), false).stopGradient();
    var masks = DecisionPolicyCandidates.candidateMasks(inputs.actionCategories(), mask);
    return encode(
            store,
            state,
            inputs,
            transitionInputs,
            presentTransitions,
            mask,
            masks,
            forward,
            training,
            parameters,
            includeDiscardScores)
        .embeddings();
  }

  Features encode(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionPolicyInputs inputs,
      DecisionTransitionInputs transitionInputs,
      NDArray presentTransitions,
      NDArray mask,
      DecisionPolicyCandidates.CandidateMasks masks,
      DecisionCandidateForward forward,
      boolean training,
      PairList<String, Object> parameters) {
    return encode(
        store,
        state,
        inputs,
        transitionInputs,
        presentTransitions,
        mask,
        masks,
        forward,
        training,
        parameters,
        true);
  }

  private Features encode(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionPolicyInputs inputs,
      DecisionTransitionInputs transitionInputs,
      NDArray presentTransitions,
      NDArray mask,
      DecisionPolicyCandidates.CandidateMasks masks,
      DecisionCandidateForward forward,
      boolean training,
      PairList<String, Object> parameters,
      boolean includeDiscardScores) {
    var contexts = encodePlayerTileContexts(store, state, forward, training, parameters);
    var transitions =
        encodeTransitions(
            store,
            state,
            inputs,
            transitionInputs,
            contexts.allPlayers(),
            presentTransitions,
            forward,
            training,
            parameters,
            inputs.rowCount(),
            includeDiscardScores);
    NDArray embeddings =
        encodeDenseCandidates(
            store,
            state,
            inputs,
            contexts,
            transitions,
            mask,
            masks,
            forward,
            training,
            parameters,
            inputs.rowCount(),
            inputs.bucket().legalActionCapacity());
    return new Features(embeddings, transitions.discardScores());
  }

  record Features(NDArray embeddings, NDArray discardScores) {}

  private NDArray encodeDenseCandidates(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyInputs inputs,
      PlayerTileContexts playerTileContexts,
      TransitionSet transitions,
      NDArray legalActionMask,
      DecisionPolicyCandidates.CandidateMasks candidateMasks,
      DecisionCandidateForward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount,
      int actionCapacity) {
    NDArray actionFeatures =
        encodeActionFeatures(
            parameterStore,
            inputs.actionCategories(),
            inputs.actionNumerics(),
            training,
            runtimeParameters,
            rowCount,
            actionCapacity);
    NDArray actionFeatureEmbeddings =
        EpsilonMahjongStateEncoder.silu(
            applyLinear(
                actionFeatureProjection,
                parameterStore,
                actionFeatures,
                training,
                runtimeParameters));
    NDArray primaryTileEmbeddings =
        DecisionPolicyTensorOps.gatherActionPrimaryTileEmbeddings(
            encodedState.tileEmbeddings(), inputs.actionCategories(), rowCount, actionCapacity);
    NDArray rootPlayerContexts =
        DecisionPolicyTensorOps.composeRootPlayerContexts(
            playerTileContexts.byPlayer(),
            playerTileContexts.allPlayers(),
            inputs.actionCategories(),
            candidateMasks,
            inputs.roundCategory(DecisionInputSchema.RoundInt.EVENT_PLAYER_RELATIVE_SEAT),
            inputs.roundCategory(DecisionInputSchema.RoundInt.EVENT_TILE));
    NDArray stateContext = encodedState.stateEmbedding().reshape(rowCount, 1, hiddenSize);
    NDArray candidateEmbeddings =
        policyForward == null
            ? EpsilonMahjongStateEncoder.silu(
                candidateFusion.fuse(
                    parameterStore,
                    training,
                    runtimeParameters,
                    actionFeatureEmbeddings,
                    primaryTileEmbeddings,
                    transitions.aggregatedEmbeddings(),
                    rootPlayerContexts,
                    stateContext))
            : policyForward.candidate(
                parameterStore,
                runtimeParameters,
                actionFeatureEmbeddings,
                primaryTileEmbeddings,
                transitions.aggregatedEmbeddings(),
                rootPlayerContexts,
                stateContext);
    candidateEmbeddings =
        candidateLayerNorm
            .forward(parameterStore, new NDList(candidateEmbeddings), training, runtimeParameters)
            .singletonOrThrow()
            .mul(legalActionMask.expandDims(2));
    return candidateEmbeddings;
  }

  private NDArray encodeActionFeatures(
      ParameterStore parameterStore,
      NDArray actionCategories,
      NDArray actionNumerics,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount,
      int actionCapacity) {
    if (!training && actionCategories.getDevice().isGpu()) {
      NDArray embeddingTable =
          actionFeatureEmbedding.getValue(parameterStore, actionCategories.getDevice(), false);
      if (embeddingTable.getDataType() == actionNumerics.getDataType()) {
        return packActionFeatures(
            actionCategories,
            actionCategoryOffsets,
            embeddingTable,
            actionNumerics,
            rowCount,
            actionCapacity);
      }
    }
    NDArray actionCategoryIds = actionCategories.stopGradient().add(actionCategoryOffsets);
    NDArray actionCategoryEmbeddings =
        actionFeatureEmbedding
            .forward(parameterStore, new NDList(actionCategoryIds), training, runtimeParameters)
            .singletonOrThrow();
    return actionCategoryEmbeddings
        .reshape(
            rowCount, actionCapacity, DecisionInputSchema.ACTION_INT_STRIDE * CATEGORY_EMBEDDING)
        .concat(actionNumerics, 2);
  }

  static NDArray packActionFeatures(
      NDArray actionCategories,
      NDArray categoryOffsets,
      NDArray embeddingTable,
      NDArray actionNumerics,
      long rowCount,
      int actionCapacity) {
    long flatRows = Math.multiplyExact(rowCount, actionCapacity);
    return NDArrays.embeddingFeaturePack(
            actionCategories
                .stopGradient()
                .reshape(flatRows, DecisionInputSchema.ACTION_INT_STRIDE),
            categoryOffsets.reshape(1, DecisionInputSchema.ACTION_INT_STRIDE),
            embeddingTable,
            actionNumerics.reshape(flatRows, DecisionInputSchema.ACTION_FLOAT_STRIDE))
        .reshape(
            rowCount,
            actionCapacity,
            DecisionInputSchema.ACTION_INT_STRIDE * CATEGORY_EMBEDDING
                + DecisionInputSchema.ACTION_FLOAT_STRIDE);
  }

  /** 凍結GPUでは埋め込み参照と数値の連結をまとめ、実在遷移ごとの中間配列を省く。 */
  NDArray encodeCompactedTransitionFeatures(
      ParameterStore parameterStore,
      NDArray categories,
      NDArray numerics,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (!training
        && categories.getDevice().isGpu()
        && !actionFeatureEmbedding.getDirectParameters().get("embedding").requiresGradient()) {
      NDArray table =
          actionFeatureEmbedding.getValue(parameterStore, categories.getDevice(), false);
      if (table.getDataType() == numerics.getDataType()) {
        long rows = categories.getShape().get(0);
        // 融合処理が受け取る2次元ビューに揃え、実在遷移の配置へ戻す。
        return NDArrays.embeddingFeaturePack(
                categories
                    .stopGradient()
                    .reshape(rows, DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE),
                transitionCategoryOffsets.reshape(
                    1, DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE),
                table,
                numerics.reshape(rows, DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE))
            .reshape(
                rows,
                1,
                1,
                DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE * CATEGORY_EMBEDDING
                    + DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE);
      }
    }
    NDArray ids = categories.add(transitionCategoryOffsets);
    return actionFeatureEmbedding
        .forward(parameterStore, new NDList(ids), training, runtimeParameters)
        .singletonOrThrow()
        .reshape(
            categories.getShape().get(0),
            1,
            1,
            DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE * CATEGORY_EMBEDDING)
        .concat(numerics, 3);
  }

  private TransitionSet encodeTransitions(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyInputs inputs,
      DecisionTransitionInputs transitionInputs,
      NDArray allPlayerTileContexts,
      NDArray transitionPresentIndices,
      DecisionCandidateForward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount,
      boolean includeDiscardScores) {
    if (training) {
      return encodeTransitionsInternal(
          parameterStore,
          encodedState,
          inputs,
          transitionInputs,
          allPlayerTileContexts,
          transitionPresentIndices,
          policyForward,
          true,
          runtimeParameters,
          rowCount,
          includeDiscardScores);
    }
    NDManager outputManager = encodedState.stateEmbedding().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          encodedState.stateEmbedding(),
          encodedState.tileEmbeddings(),
          allPlayerTileContexts,
          transitionInputs.categories(),
          transitionInputs.numerics(),
          transitionInputs.tileCodes(),
          transitionInputs.waitTileIds(),
          transitionInputs.waitYakus(),
          transitionInputs.waitScores());
      tempAttachTileProjection(scope, encodedState);
      TransitionSet transitions =
          encodeTransitionsInternal(
              parameterStore,
              encodedState,
              inputs,
              transitionInputs,
              allPlayerTileContexts,
              transitionPresentIndices,
              policyForward,
              false,
              runtimeParameters,
              rowCount,
              includeDiscardScores);
      if (transitions.discardScores() != null) outputManager.attachAll(transitions.discardScores());
      outputManager.attachAll(transitions.aggregatedEmbeddings());
      return transitions;
    }
  }

  private TransitionSet encodeTransitionsInternal(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyInputs inputs,
      DecisionTransitionInputs transitionInputs,
      NDArray allPlayerTileContexts,
      NDArray transitionPresentIndices,
      DecisionCandidateForward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount,
      boolean includeDiscardScores) {
    int actionCapacity = inputs.bucket().legalActionCapacity();
    int transitionCapacity = inputs.bucket().actionTransitionCapacity();
    if (transitionInputs.hostCompacted()
        || transitionCapacity > 1
        || (!training
            && encodedState.stateEmbedding().getDevice().isGpu()
            && transitionPresentIndices != null
            && transitionPresentIndices.getShape().size() > 0)) {
      return encodeCompactedTransitions(
          parameterStore,
          encodedState,
          transitionInputs,
          allPlayerTileContexts,
          transitionPresentIndices,
          policyForward,
          training,
          runtimeParameters,
          rowCount,
          actionCapacity,
          transitionCapacity,
          includeDiscardScores);
    }
    NDArray transitionCategoryValues = transitionInputs.categories().stopGradient();
    NDArray transitionCategoryIds = transitionCategoryValues.add(transitionCategoryOffsets);
    NDArray transitionCategoryEmbeddings =
        actionFeatureEmbedding
            .forward(parameterStore, new NDList(transitionCategoryIds), training, runtimeParameters)
            .singletonOrThrow();
    NDArray transitionFeatures =
        transitionCategoryEmbeddings
            .reshape(
                rowCount,
                actionCapacity,
                transitionCapacity,
                DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE * CATEGORY_EMBEDDING)
            .concat(transitionInputs.numerics(), 3);
    NDArray transitionFeatureEmbeddings =
        EpsilonMahjongStateEncoder.silu(
            applyLinear(
                transitionFeatureProjection,
                parameterStore,
                transitionFeatures,
                training,
                runtimeParameters));
    NDArray projectedTileKeyValues =
        EpsilonLinear.projectBatchSlices(
            transitionTileKeyValueProjection,
            parameterStore,
            encodedState.tileProjectionEmbeddings(),
            training,
            runtimeParameters);
    NDArray transitionTileContexts =
        attendTransitionTiles(
            parameterStore,
            projectedTileKeyValues,
            null,
            transitionFeatureEmbeddings,
            transitionInputs.tileCodes(),
            transitionInputs.waitTileIds(),
            transitionInputs.waitYakus(),
            transitionInputs.waitScores(),
            rowCount,
            actionCapacity,
            transitionCapacity,
            training,
            runtimeParameters);
    NDArray discardTileEmbeddings =
        DecisionPolicyTensorOps.gatherTransitionDiscardTileEmbeddings(
            encodedState.tileEmbeddings(),
            transitionInputs.categories(),
            rowCount,
            actionCapacity,
            transitionCapacity);
    NDArray discardSafetyContexts =
        DecisionPolicyTensorOps.gatherTransitionDiscardContexts(
            allPlayerTileContexts,
            transitionInputs.categories(),
            rowCount,
            actionCapacity,
            transitionCapacity);
    NDArray transitionStateContexts =
        encodedState.stateEmbedding().reshape(rowCount, 1, 1, hiddenSize);
    NDArray transitionEmbeddings =
        EpsilonMahjongStateEncoder.silu(
            transitionFusion.fuse(
                parameterStore,
                training,
                runtimeParameters,
                transitionFeatureEmbeddings,
                transitionTileContexts,
                discardTileEmbeddings,
                discardSafetyContexts,
                transitionStateContexts));
    transitionEmbeddings =
        transitionLayerNorm
            .forward(parameterStore, new NDList(transitionEmbeddings), training, runtimeParameters)
            .singletonOrThrow();
    NDArray presentMask =
        transitionInputs
            .categories()
            .get("...,{}", DecisionInputSchema.ActionTransitionInt.PRESENT.ordinal())
            .neq(0)
            .toType(encodedState.stateEmbedding().getDataType(), false)
            .stopGradient();
    transitionEmbeddings = transitionEmbeddings.mul(presentMask.expandDims(3));
    if (!includeDiscardScores && transitionCapacity == 1)
      return new TransitionSet(null, transitionEmbeddings.get("...,:,0,:"));
    NDArray discardMask =
        transitionInputs
            .categories()
            .get("...,{}", DecisionInputSchema.ActionTransitionInt.DISCARD_ACTION_ID.ordinal())
            .neq(0)
            .toType(transitionEmbeddings.getDataType(), false)
            .mul(presentMask)
            .stopGradient();
    NDArray discardScores =
        discardCandidateScorer
            .score(parameterStore, transitionEmbeddings, training, runtimeParameters)
            .reshape(rowCount, actionCapacity, transitionCapacity)
            .mul(discardMask);
    NDArray aggregatedEmbeddings;
    if (transitionCapacity == 1) {
      aggregatedEmbeddings = transitionEmbeddings.get("...,:,0,:");
    } else {
      NDArray transitionWeights = transitionPooling.weights(discardScores, presentMask);
      aggregatedEmbeddings =
          transitionEmbeddings.mul(transitionWeights.expandDims(3)).sum(new int[] {2});
    }
    return new TransitionSet(discardScores, aggregatedEmbeddings);
  }

  private TransitionSet encodeCompactedTransitions(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionTransitionInputs transitionInputs,
      NDArray allPlayerTileContexts,
      NDArray suppliedPresentIndices,
      DecisionCandidateForward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount,
      int actionCapacity,
      int transitionCapacity,
      boolean includeDiscardScores) {
    long denseCount =
        Math.multiplyExact(rowCount, Math.multiplyExact(actionCapacity, transitionCapacity));
    boolean directSingleTransitionInference =
        !training && transitionCapacity == 1 && encodedState.stateEmbedding().getDevice().isGpu();
    NDArray presentMask;
    NDArray presentIndices;
    if (directSingleTransitionInference) {
      presentIndices = suppliedPresentIndices;
      presentMask = null;
    } else if (transitionInputs.hostCompacted()) {
      presentIndices = suppliedPresentIndices;
      long presentCount = presentIndices.getShape().size();
      presentMask =
          EpsilonMaskedRows.scatter(
                  presentIndices
                      .getManager()
                      .ones(
                          new Shape(presentCount, 1), encodedState.stateEmbedding().getDataType()),
                  presentIndices,
                  denseCount)
              .reshape(rowCount, actionCapacity, transitionCapacity)
              .stopGradient();
    } else {
      presentMask =
          transitionInputs
              .categories()
              .get("...,{}", DecisionInputSchema.ActionTransitionInt.PRESENT.ordinal())
              .neq(0)
              .toType(encodedState.stateEmbedding().getDataType(), false)
              .stopGradient();
      presentIndices =
          suppliedPresentIndices == null
              ? EpsilonMaskedRows.indices(presentMask)
              : suppliedPresentIndices;
    }
    long presentCount = presentIndices.getShape().size();

    NDArray compactCategories =
        transitionInputs.hostCompacted()
            ? transitionInputs.categories()
            : DecisionPolicyTensorOps.compactTransitionRows(
                transitionInputs.categories(), presentIndices, denseCount);
    NDArray compactNumerics =
        transitionInputs.hostCompacted()
            ? transitionInputs.numerics()
            : DecisionPolicyTensorOps.compactTransitionRows(
                transitionInputs.numerics(), presentIndices, denseCount);
    NDArray compactTileCodes =
        transitionInputs.hostCompacted()
            ? transitionInputs.tileCodes()
            : DecisionPolicyTensorOps.compactTransitionRows(
                transitionInputs.tileCodes(), presentIndices, denseCount);
    NDArray compactWaitTileIds =
        transitionInputs.hostCompacted()
            ? transitionInputs.waitTileIds()
            : DecisionPolicyTensorOps.compactTransitionRows(
                transitionInputs.waitTileIds(), presentIndices, denseCount);
    NDArray compactWaitYakus =
        transitionInputs.hostCompacted()
            ? transitionInputs.waitYakus()
            : DecisionPolicyTensorOps.compactTransitionRows(
                transitionInputs.waitYakus(), presentIndices, denseCount);
    NDArray compactWaitScores =
        transitionInputs.hostCompacted()
            ? transitionInputs.waitScores()
            : DecisionPolicyTensorOps.compactTransitionRows(
                transitionInputs.waitScores(), presentIndices, denseCount);

    NDArray compactBatchIds =
        DecisionPolicyTensorOps.compactBatchIds(
            presentIndices, rowCount, actionCapacity, transitionCapacity);
    NDArray compactTransitionEmbeddings =
        encodeCompactTransitionEmbeddings(
            parameterStore,
            encodedState,
            allPlayerTileContexts,
            new DecisionTransitionInputs(
                compactCategories,
                compactNumerics,
                compactTileCodes,
                compactWaitTileIds,
                compactWaitYakus,
                compactWaitScores,
                true),
            compactBatchIds,
            policyForward,
            training,
            runtimeParameters);
    return poolCompactedTransitions(
        parameterStore,
        compactTransitionEmbeddings,
        compactCategories,
        presentIndices,
        presentMask,
        rowCount,
        actionCapacity,
        transitionCapacity,
        training,
        runtimeParameters,
        includeDiscardScores);
  }

  /** 区分に依存しない実在遷移の特徴をまとめて計算する。 */
  private NDArray encodeCompactTransitionEmbeddings(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      NDArray allPlayerTileContexts,
      DecisionTransitionInputs inputs,
      NDArray compactBatchIds,
      DecisionCandidateForward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long presentCount = compactBatchIds.size();
    NDArray compactCategories = inputs.categories();
    NDArray compactNumerics = inputs.numerics();
    NDArray compactTileCodes = inputs.tileCodes();
    NDArray compactWaitTileIds = inputs.waitTileIds();
    NDArray compactWaitYakus = inputs.waitYakus();
    NDArray compactWaitScores = inputs.waitScores();
    NDArray compactFeatures =
        encodeCompactedTransitionFeatures(
            parameterStore, compactCategories, compactNumerics, training, runtimeParameters);
    NDArray compactFeatureEmbeddings =
        EpsilonMahjongStateEncoder.silu(
            applyLinear(
                transitionFeatureProjection,
                parameterStore,
                compactFeatures,
                training,
                runtimeParameters));
    NDArray projectedTileKeyValues =
        EpsilonLinear.projectBatchSlices(
            transitionTileKeyValueProjection,
            parameterStore,
            encodedState.tileProjectionEmbeddings(),
            training,
            runtimeParameters);
    NDArray compactTileContexts =
        attendTransitionTiles(
            parameterStore,
            projectedTileKeyValues,
            compactBatchIds,
            compactFeatureEmbeddings,
            compactTileCodes,
            compactWaitTileIds,
            compactWaitYakus,
            compactWaitScores,
            presentCount,
            1,
            1,
            training,
            runtimeParameters);
    NDArray compactDiscardTileEmbeddings =
        DecisionPolicyTensorOps.gatherCompactedTransitionDiscardTileEmbeddings(
            encodedState.tileEmbeddings(), compactCategories, compactBatchIds);
    NDArray compactDiscardSafetyContexts =
        DecisionPolicyTensorOps.gatherCompactedTransitionDiscardContexts(
            allPlayerTileContexts, compactCategories, compactBatchIds);
    NDArray compactTransitionEmbeddings =
        policyForward == null
            ? EpsilonMahjongStateEncoder.silu(
                transitionFusion.fuse(
                    parameterStore,
                    training,
                    runtimeParameters,
                    compactFeatureEmbeddings,
                    compactTileContexts,
                    compactDiscardTileEmbeddings,
                    compactDiscardSafetyContexts,
                    EpsilonMaskedRows.gather(encodedState.stateEmbedding(), compactBatchIds)
                        .reshape(presentCount, 1, 1, hiddenSize)))
            : policyForward.transition(
                parameterStore,
                runtimeParameters,
                compactFeatureEmbeddings,
                compactTileContexts,
                compactDiscardTileEmbeddings,
                compactDiscardSafetyContexts,
                encodedState.stateEmbedding(),
                compactBatchIds);
    compactTransitionEmbeddings =
        transitionLayerNorm
            .forward(
                parameterStore,
                new NDList(compactTransitionEmbeddings),
                training,
                runtimeParameters)
            .singletonOrThrow();

    return compactTransitionEmbeddings;
  }

  /** 集約だけを元の容量区分で実行し、幅1の遷移へスコア計算を追加しない。 */
  private TransitionSet poolCompactedTransitions(
      ParameterStore parameterStore,
      NDArray compactTransitionEmbeddings,
      NDArray compactCategories,
      NDArray presentIndices,
      NDArray presentMask,
      long rowCount,
      int actionCapacity,
      int transitionCapacity,
      boolean training,
      PairList<String, Object> runtimeParameters,
      boolean includeDiscardScores) {
    long presentCount = presentIndices.size();
    long denseCount = rowCount * actionCapacity * transitionCapacity;
    boolean directSingleTransitionInference =
        !training && transitionCapacity == 1 && compactTransitionEmbeddings.getDevice().isGpu();
    if (!includeDiscardScores && transitionCapacity == 1) {
      NDArray aggregatedEmbeddings =
          directSingleTransitionInference
              ? DecisionPolicyTensorOps.scatterSingleCompactedTransitionRows(
                  compactTransitionEmbeddings, presentIndices, rowCount, actionCapacity, hiddenSize)
              : EpsilonMaskedRows.scatter(
                      compactTransitionEmbeddings.reshape(presentCount, hiddenSize),
                      presentIndices,
                      denseCount)
                  .reshape(rowCount, actionCapacity, hiddenSize);
      return new TransitionSet(null, aggregatedEmbeddings);
    }

    NDArray compactDiscardScores =
        discardCandidateScorer
            .score(parameterStore, compactTransitionEmbeddings, training, runtimeParameters)
            .reshape(presentCount, 1, 1);
    if (directSingleTransitionInference) {
      compactDiscardScores =
          DecisionPolicyTensorOps.alignSingleTransitionScoreType(
              compactDiscardScores, compactTransitionEmbeddings);
      NDArray discardScores =
          DecisionPolicyTensorOps.scatterSingleCompactedTransitionRows(
              compactDiscardScores, presentIndices, rowCount, actionCapacity, 1);
      NDArray aggregatedEmbeddings =
          DecisionPolicyTensorOps.scatterSingleCompactedTransitionRows(
              compactTransitionEmbeddings, presentIndices, rowCount, actionCapacity, hiddenSize);
      return new TransitionSet(discardScores, aggregatedEmbeddings);
    }

    NDArray compactDiscardMask =
        compactCategories
            .get("...,{}", DecisionInputSchema.ActionTransitionInt.DISCARD_ACTION_ID.ordinal())
            .neq(0)
            .toType(compactTransitionEmbeddings.getDataType(), false)
            .stopGradient();
    compactDiscardScores = compactDiscardScores.mul(compactDiscardMask);
    NDArray discardScores =
        EpsilonMaskedRows.scatter(
                compactDiscardScores.reshape(presentCount, 1), presentIndices, denseCount)
            .reshape(rowCount, actionCapacity, transitionCapacity);
    NDArray transitionWeights = transitionPooling.weights(discardScores, presentMask);
    NDArray aggregatedEmbeddings =
        DecisionPolicyTensorOps.aggregateCompactedTransitions(
            compactTransitionEmbeddings,
            transitionWeights,
            presentIndices,
            denseCount,
            rowCount,
            actionCapacity,
            transitionCapacity,
            hiddenSize);
    return new TransitionSet(discardScores, aggregatedEmbeddings);
  }

  private NDArray attendTransitionTiles(
      ParameterStore parameterStore,
      NDArray projectedTileKeyValues,
      NDArray projectedTileGroupIndices,
      NDArray transitionFeatureEmbeddings,
      NDArray transitionTileCodes,
      NDArray waitTileIds,
      NDArray transitionWaitYakuCodes,
      NDArray transitionWaitScores,
      long rowCount,
      int actionCapacity,
      int transitionCapacity,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      return attendTransitionTilesInternal(
          parameterStore,
          projectedTileKeyValues,
          projectedTileGroupIndices,
          transitionFeatureEmbeddings,
          transitionTileCodes,
          waitTileIds,
          transitionWaitYakuCodes,
          transitionWaitScores,
          rowCount,
          actionCapacity,
          transitionCapacity,
          true,
          runtimeParameters);
    }
    NDManager outputManager = transitionFeatureEmbeddings.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          projectedTileKeyValues,
          transitionFeatureEmbeddings,
          transitionTileCodes,
          waitTileIds,
          transitionWaitYakuCodes,
          transitionWaitScores);
      if (projectedTileGroupIndices != null) {
        scope.tempAttachAll(projectedTileGroupIndices);
      }
      NDArray context =
          attendTransitionTilesInternal(
              parameterStore,
              projectedTileKeyValues,
              projectedTileGroupIndices,
              transitionFeatureEmbeddings,
              transitionTileCodes,
              waitTileIds,
              transitionWaitYakuCodes,
              transitionWaitScores,
              rowCount,
              actionCapacity,
              transitionCapacity,
              false,
              runtimeParameters);
      outputManager.attachAll(context);
      return context;
    }
  }

  private NDArray attendTransitionTilesInternal(
      ParameterStore parameterStore,
      NDArray projectedTileKeyValues,
      NDArray projectedTileGroupIndices,
      NDArray transitionFeatureEmbeddings,
      NDArray transitionTileCodes,
      NDArray waitTileIds,
      NDArray transitionWaitYakuCodes,
      NDArray transitionWaitScores,
      long rowCount,
      int actionCapacity,
      int transitionCapacity,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long candidatesPerState = Math.multiplyExact(actionCapacity, transitionCapacity);
    long candidateCount = Math.multiplyExact(rowCount, candidatesPerState);
    NDArray storedRelationCodes = transitionTileCodes.stopGradient().toType(DataType.INT32, false);
    NDArray waitYakuEmbeddings =
        encodeWaitYakus(parameterStore, transitionWaitYakuCodes, training, runtimeParameters);
    NDArray waitFeatures = waitYakuEmbeddings.concat(transitionWaitScores, 4);
    NDArray projectedWaitKeyValues =
        applyLinear(
            transitionWaitKeyValueProjection,
            parameterStore,
            waitFeatures,
            training,
            runtimeParameters);
    NDArray flatTransitionQueries =
        applyLinear(
                transitionTileQueryProjection,
                parameterStore,
                transitionFeatureEmbeddings,
                training,
                runtimeParameters)
            .reshape(candidateCount, TRANSITION_TILE_ATTENTION_HEADS, TRANSITION_TILE_KEY_SIZE);
    float attentionScale = (float) (1.0 / Math.sqrt(TRANSITION_TILE_KEY_SIZE));
    NDArray storedWaitTileIds = waitTileIds.toType(DataType.INT32, false);
    NDArray tileGroupIndices =
        projectedTileGroupIndices == null
            ? DecisionPolicyTensorOps.denseBatchIds(
                flatTransitionQueries.getManager(), rowCount, actionCapacity, transitionCapacity)
            : projectedTileGroupIndices.reshape(candidateCount);
    if (tileGroupIndices.getDataType() != DataType.INT32) {
      tileGroupIndices = tileGroupIndices.toType(DataType.INT32, false);
    }
    tileGroupIndices = tileGroupIndices.stopGradient();
    NDArray relationTable =
        transitionTileRelationTable(parameterStore, flatTransitionQueries, training);
    return NDArrays.mappedGroupedIndexedScaledDotProductAttention(
            flatTransitionQueries,
            projectedTileKeyValues.reshape(
                projectedTileKeyValues.getShape().get(0),
                DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT,
                TRANSITION_TILE_KEY_WIDTH + contextWidth),
            tileGroupIndices.reshape(candidateCount),
            relationTable,
            storedRelationCodes.reshape(
                candidateCount, DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT),
            projectedWaitKeyValues.reshape(
                candidateCount,
                DecisionInputSchema.MAX_WAIT_TILE_TYPES,
                TRANSITION_TILE_KEY_WIDTH + contextWidth),
            storedWaitTileIds.reshape(candidateCount, DecisionInputSchema.MAX_WAIT_TILE_TYPES),
            attentionScale)
        .reshape(rowCount, actionCapacity, transitionCapacity, contextWidth);
  }

  private NDArray transitionTileRelationTable(
      ParameterStore parameterStore, NDArray query, boolean training) {
    NDArray relationTable =
        transitionTileRelationEmbedding.getValue(parameterStore, query.getDevice(), training);
    if (relationTable.getDataType() == query.getDataType()) {
      return relationTable;
    }
    NDArray converted = relationTable.toType(query.getDataType(), false);
    converted.attach(query.getManager());
    return converted;
  }

  private NDArray encodeWaitYakus(
      ParameterStore parameterStore,
      NDArray storedCodes,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      clearInferenceWaitYakuLookup();
      return encodeWaitYakusInternal(parameterStore, storedCodes, true, runtimeParameters);
    }
    NDArray lookup = inferenceWaitYakuLookup(parameterStore, storedCodes);
    NDManager outputManager = storedCodes.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(storedCodes, lookup);
      NDArray embeddings = encodeWaitYakusFromLookup(storedCodes, lookup);
      outputManager.attachAll(embeddings);
      return embeddings;
    }
  }

  private NDArray encodeWaitYakusInternal(
      ParameterStore parameterStore,
      NDArray storedCodes,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    Shape shape = storedCodes.getShape();
    NDArray packedBits =
        storedCodes.toType(DataType.FLOAT32, false).maximum(1.0f).sub(1.0f).stopGradient();
    NDArray yakuBits =
        packedBits
            .expandDims(5)
            .div(waitYakuBitDivisors)
            .floor()
            .mod(2)
            .reshape(
                shape.get(0),
                shape.get(1),
                shape.get(2),
                shape.get(3),
                DecisionInputSchema.WaitWinType.values().length,
                DecisionInputSchema.WAIT_YAKU_CHUNKS_PER_WIN_TYPE
                    * DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK)
            .stopGradient();
    return applyLinear(
            transitionWaitYakuProjection, parameterStore, yakuBits, training, runtimeParameters)
        .reshape(
            shape.get(0),
            shape.get(1),
            shape.get(2),
            shape.get(3),
            DecisionInputSchema.WaitWinType.values().length * WAIT_YAKU_EMBEDDING);
  }

  static NDArray encodeWaitYakusFromLookup(NDArray storedCodes, NDArray lookup) {
    Shape shape = storedCodes.getShape();
    int winTypes = DecisionInputSchema.WaitWinType.values().length;
    int chunks = DecisionInputSchema.WAIT_YAKU_CHUNKS_PER_WIN_TYPE;
    NDArray chunkCodes =
        storedCodes.reshape(
            shape.get(0), shape.get(1), shape.get(2), shape.get(3), winTypes, chunks);
    return NDArrays.segmentedLookupSum(lookup, chunkCodes)
        .reshape(
            shape.get(0), shape.get(1), shape.get(2), shape.get(3), winTypes * WAIT_YAKU_EMBEDDING);
  }

  private NDArray inferenceWaitYakuLookup(ParameterStore parameterStore, NDArray storedCodes) {
    if (inferenceWaitYakuLookup != null) {
      return inferenceWaitYakuLookup;
    }
    NDArray weight =
        parameterStore.getValue(
            transitionWaitYakuProjection.getParameters().get("weight"),
            storedCodes.getDevice(),
            false);
    NDManager outputManager = weight.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(weight, waitYakuBitDivisors);
      NDArray codes =
          scope.arange(WAIT_YAKU_CODE_COUNT).toType(DataType.FLOAT32, false).reshape(-1, 1);
      NDArray bits = codes.div(waitYakuBitDivisors).floor().mod(2);
      NDList chunkTables = new NDList();
      for (int chunk = 0; chunk < DecisionInputSchema.WAIT_YAKU_CHUNKS_PER_WIN_TYPE; chunk++) {
        int start = chunk * DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK;
        int end = start + DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK;
        chunkTables.add(bits.matMul(weight.get(":,{}:{}", start, end).transpose()));
      }
      inferenceWaitYakuLookup =
          NDArrays.stack(chunkTables)
              .reshape(
                  (long) DecisionInputSchema.WAIT_YAKU_CHUNKS_PER_WIN_TYPE * WAIT_YAKU_CODE_COUNT,
                  WAIT_YAKU_EMBEDDING);
      outputManager.attachAll(inferenceWaitYakuLookup);
      return inferenceWaitYakuLookup;
    }
  }

  private void clearInferenceWaitYakuLookup() {
    if (inferenceWaitYakuLookup != null) {
      inferenceWaitYakuLookup.close();
      inferenceWaitYakuLookup = null;
    }
  }

  private PlayerTileContexts encodePlayerTileContexts(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionCandidateForward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      return encodePlayerTileContextsInternal(
          parameterStore, encodedState, null, true, runtimeParameters);
    }
    NDManager outputManager = encodedState.tileEmbeddings().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          encodedState.tileEmbeddings(),
          encodedState.playerMemory(),
          encodedState.playerMemoryMask());
      tempAttachTileProjection(scope, encodedState);
      PlayerTileContexts contexts =
          encodePlayerTileContextsInternal(
              parameterStore, encodedState, policyForward, false, runtimeParameters);
      outputManager.attachAll(contexts.byPlayer(), contexts.allPlayers());
      return contexts;
    }
  }

  private PlayerTileContexts encodePlayerTileContextsInternal(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionCandidateForward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray byPlayer =
        attendPlayerMemories(parameterStore, encodedState, training, runtimeParameters);
    NDArray allPlayers =
        policyForward == null
            ? playerTileContextMixer.mix(parameterStore, byPlayer, training)
            : policyForward.mixPlayerTileContexts(parameterStore, byPlayer);
    return new PlayerTileContexts(byPlayer, allPlayers);
  }

  private NDArray attendPlayerMemories(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray queries =
        applyLinear(
            playerTileQueryProjection,
            parameterStore,
            encodedState.tileProjectionEmbeddings(),
            training,
            runtimeParameters);
    NDArray projectedKeyValues =
        applyLinear(
            playerMemoryKeyValueProjection,
            parameterStore,
            encodedState.playerMemory(),
            training,
            runtimeParameters);
    return attendProjectedPlayerMemories(
        queries, projectedKeyValues, encodedState.playerMemoryMask());
  }

  static NDArray attendProjectedPlayerMemories(
      NDArray queries, NDArray projectedKeyValues, NDArray playerMemoryMask) {
    int queryWidth = Math.toIntExact(queries.getShape().get(queries.getShape().dimension() - 1));
    int headSize = queryWidth / EpsilonMahjongStateEncoder.ATTENTION_HEADS;
    return NDArrays.groupedPackedScaledDotProductAttention(
        queries,
        projectedKeyValues,
        playerMemoryMask,
        EpsilonMahjongStateEncoder.ATTENTION_HEADS,
        1.0 / Math.sqrt(headSize));
  }

  private static void tempAttachTileProjection(
      NDManager scope, EpsilonMahjongStateEncoder.EncodedState encodedState) {
    if (encodedState.tileProjectionEmbeddings() != encodedState.tileEmbeddings()) {
      scope.tempAttachAll(encodedState.tileProjectionEmbeddings());
    }
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

  private Linear hiddenLinear() {
    return Linear.builder().setUnits(hiddenSize).build();
  }

  private record PlayerTileContexts(NDArray byPlayer, NDArray allPlayers) {}

  private record TransitionSet(NDArray discardScores, NDArray aggregatedEmbeddings) {}
}

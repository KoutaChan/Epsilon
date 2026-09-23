package com.epsilon.major.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.model.EpsilonLinear;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.config.settings.DecisionInferenceFusionMode;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.major.ai.decision.input.DecisionCategoryLayout;
import com.epsilon.major.ai.decision.input.DecisionFeatureCodec;
import com.epsilon.major.ai.decision.input.DecisionInferenceDeviceBatch;
import com.epsilon.major.ai.decision.input.DecisionInferenceInputs;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.major.ai.decision.input.DecisionPolicyInputs;
import com.epsilon.major.ai.decision.policy.fusion.DecisionPolicyFusionAffineExecution;
import com.epsilon.major.ai.decision.policy.fusion.DecisionPolicyFusionCandidateContextExecution;
import com.epsilon.major.ai.decision.policy.fusion.DecisionPolicyFusionCandidatePrefixExecution;
import com.epsilon.major.ai.decision.policy.fusion.DecisionPolicyFusionContextExecution;
import com.epsilon.major.ai.decision.policy.fusion.DecisionPolicyFusionIndexedAffineExecution;
import com.epsilon.major.ai.decision.policy.fusion.DecisionPolicyFusionPlayerTileContextExecution;
import com.epsilon.major.ai.model.EpsilonDecisionArchitecture;
import com.epsilon.major.ai.model.EpsilonMahjongStateEncoder;

/**
 * 状態・行動・行動適用後の遷移を採点し、麻雀の条件付き方策グラフへ渡す。
 *
 * <p>全行動候補は最低一件の遷移を持つ。チー・ポンは合法な直後打牌ごとに一件、通常打牌・PASSは一件、槓は嶺上待ち一件、
 * 和了・九種九牌は終端一件である。遷移集合は共有の打牌採点器のマスク付きソフトマックスで重み付けし、同じ重みから一つの 整合した表現を作る。次元別maxや経験則によるな最良打牌は使わない。
 *
 * <p>条件付き打牌ロジットは通常打牌と鳴き直後打牌で同じ{@link EpsilonDiscardCandidateScorer}を通る。softmax重みは
 * 全スコアへの定数加算で変化せずであり、鳴き判断損失からは勾配を遮断する。これにより行動選択の損失は投影表現を学習できる一方、未実行の後続打牌ロジットを
 * 都合よく改変しない。採点器自体は、実際に観測された通常打牌・鳴き直後打牌の選択行動損失から学習する。
 *
 * <p>行動候補、MELD/KAN 種類、CALL/KAN 二択分岐の親表現も、方策グラフが実際に使う条件付き子分布で再帰的に集約する。
 * 各重みは勾配を遮断し、上位損失から子ロジットへの直接勾配だけを遮断する。独立した種類-クエリ注意機構は持たない。
 *
 * <p>各遷移の34牌関係は、小さい関係-キーベクトルとして文脈を反映した牌トークンへのマルチヘッド
 * 注意機構へ入る。形待ちとRON/TSUMO別の公開和了事実は固定Projectionで点棒・順位・半荘遷移へ変換し、対応する牌種の表現へ加える。
 *
 * <p>34牌種はクエリとして各プレイヤー固有のプレイヤートークン・河・副露メモリを一度だけ読む。捨て牌遷移は全プレイヤー文脈を牌種で
 * 指定位置の抽出し、応答行動はイベントの牌種と打牌元のプレイヤーで直接抽出する。KANは主デバイス牌の全プレイヤー文脈を使う。これにより
 * 「この牌を誰へ切るか」「誰の捨て牌へ応答するか」を全体の状態へ潰さず、全行動と全構成要素の密な注意機構も作らない。
 */
public final class EpsilonDecisionPolicyHead extends AbstractBlock {

  private static final int CATEGORY_EMBEDDING = 8;
  private static final int TRANSITION_TILE_ATTENTION_HEADS = 4;
  private static final int TRANSITION_TILE_KEY_SIZE = 8;
  private static final int TRANSITION_TILE_KEY_WIDTH =
      TRANSITION_TILE_ATTENTION_HEADS * TRANSITION_TILE_KEY_SIZE;
  private static final long[] ALTERNATIVE_IDS = alternativeIds();

  private final int hiddenSize;
  private final int contextWidth;
  private final IdEmbedding actionFeatureEmbedding;
  private final IdEmbedding transitionTileRelationEmbedding;
  private final EpsilonPointProjection pointProjection;
  private final EpsilonPointOutcomeEncoder pointOutcomeEncoder;
  private final IdEmbedding alternativeOffset;
  private final Linear actionFeatureProjection;
  private final Linear transitionFeatureProjection;
  private final Linear transitionTileKeyValueProjection;
  private final Linear transitionPublicContextProjection;
  private final Linear transitionTileQueryProjection;
  private final Linear transitionWaitKeyValueProjection;
  private final Linear playerTileQueryProjection;
  private final EpsilonPlayerTileContextMixer playerTileContextMixer;
  private final EpsilonFeatureFusion transitionFusion;
  private final LayerNorm transitionLayerNorm;
  private final EpsilonDiscardCandidateScorer discardCandidateScorer;
  private final EpsilonFeatureFusion candidateFusion;
  private final LayerNorm candidateLayerNorm;
  private final Linear actionCandidateHidden;
  private final Linear actionCandidateScoreHead;
  private final Linear alternativeHidden;
  private final Linear alternativeScoreHead;
  private final EpsilonBinaryBranchGate ronGate;
  private final EpsilonBinaryBranchGate callGate;
  private final EpsilonBinaryBranchGate tsumoGate;
  private final EpsilonBinaryBranchGate kyushuGate;
  private final EpsilonBinaryBranchGate kanGate;
  private final EpsilonBinaryBranchGate riichiGate;
  private NDArray actionCategoryOffsets;
  private NDArray transitionCategoryOffsets;
  private NDArray alternativeIds;

  /**
   * 指定隠れ層幅で遷移-を考慮した方策採点器を構築する。
   *
   * @param hiddenSize 状態、行動、遷移埋め込みの共通幅
   */
  public EpsilonDecisionPolicyHead(int hiddenSize, EpsilonUtilityProfile utilityProfile) {
    if (hiddenSize % TRANSITION_TILE_ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be divisible by transition tile attention heads");
    }
    this.hiddenSize = hiddenSize;
    contextWidth = Math.min(hiddenSize, EpsilonDecisionArchitecture.POLICY_CONTEXT_WIDTH);
    actionFeatureEmbedding =
        addChildBlock(
            "actionFeatureEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(DecisionCategoryLayout.ACTION_DICTIONARY_SIZE)
                .setEmbeddingSize(CATEGORY_EMBEDDING)
                .build());
    transitionTileRelationEmbedding =
        addChildBlock(
            "transitionTileRelationEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(DecisionInputSchema.ACTION_TRANSITION_TILE_DICTIONARY_SIZE)
                .setEmbeddingSize(TRANSITION_TILE_KEY_WIDTH + contextWidth)
                .build());
    pointProjection = addChildBlock("pointProjection", new EpsilonPointProjection(utilityProfile));
    pointOutcomeEncoder =
        addChildBlock("pointOutcomeEncoder", new EpsilonPointOutcomeEncoder(contextWidth));
    alternativeOffset =
        addChildBlock(
            "alternativeOffset",
            new IdEmbedding.Builder()
                .setDictionarySize(DecisionAlternative.NETWORK_SIZE)
                .setEmbeddingSize(hiddenSize)
                .build());
    actionFeatureProjection = addChildBlock("actionFeatureProjection", hiddenLinear());
    transitionFeatureProjection = addChildBlock("transitionFeatureProjection", hiddenLinear());
    transitionTileKeyValueProjection =
        addChildBlock(
            "transitionTileKeyValueProjection",
            Linear.builder().setUnits(TRANSITION_TILE_KEY_WIDTH + contextWidth).build());
    transitionPublicContextProjection =
        addChildBlock(
            "transitionPublicContextProjection",
            Linear.builder().setUnits(TRANSITION_TILE_KEY_WIDTH + contextWidth).build());
    transitionPublicContextProjection
        .getDirectParameters()
        .get("weight")
        .setInitializer(new ConstantInitializer(0.0f));
    transitionPublicContextProjection
        .getDirectParameters()
        .get("bias")
        .setInitializer(new ConstantInitializer(0.0f));
    transitionTileQueryProjection =
        addChildBlock(
            "transitionTileQueryProjection",
            Linear.builder().setUnits(TRANSITION_TILE_KEY_WIDTH).build());
    transitionWaitKeyValueProjection =
        addChildBlock(
            "transitionWaitKeyValueProjection",
            Linear.builder().setUnits(TRANSITION_TILE_KEY_WIDTH + contextWidth).build());
    playerTileQueryProjection =
        addChildBlock(
            "playerTileQueryProjection",
            Linear.builder().setUnits(contextWidth).optBias(false).build());
    playerTileContextMixer =
        addChildBlock("playerTileContextMixer", new EpsilonPlayerTileContextMixer(contextWidth));
    transitionFusion =
        addChildBlock(
            "transitionFusion",
            new EpsilonFeatureFusion(
                hiddenSize, 4, hiddenSize, contextWidth, hiddenSize, contextWidth, hiddenSize));
    transitionLayerNorm = addChildBlock("transitionLayerNorm", LayerNorm.builder().axis(3).build());
    discardCandidateScorer =
        addChildBlock("discardCandidateScorer", new EpsilonDiscardCandidateScorer(hiddenSize));
    candidateFusion =
        addChildBlock(
            "candidateFusion",
            new EpsilonFeatureFusion(
                hiddenSize, 4, hiddenSize, hiddenSize, hiddenSize, contextWidth, hiddenSize));
    candidateLayerNorm = addChildBlock("candidateLayerNorm", LayerNorm.builder().axis(2).build());
    actionCandidateHidden = addChildBlock("actionCandidateHidden", hiddenLinear());
    actionCandidateScoreHead =
        addChildBlock(
            "actionCandidateScoreHead", Linear.builder().setUnits(1).optBias(false).build());
    alternativeHidden = addChildBlock("alternativeHidden", hiddenLinear());
    alternativeScoreHead =
        addChildBlock("alternativeScoreHead", Linear.builder().setUnits(1).build());
    ronGate =
        addChildBlock(
            "ronGate",
            new EpsilonBinaryBranchGate(hiddenSize, EpsilonPointProjection.GATE_FEATURE_WIDTH));
    callGate = addChildBlock("callGate", new EpsilonBinaryBranchGate(hiddenSize));
    tsumoGate =
        addChildBlock(
            "tsumoGate",
            new EpsilonBinaryBranchGate(hiddenSize, EpsilonPointProjection.GATE_FEATURE_WIDTH));
    kyushuGate = addChildBlock("kyushuGate", new EpsilonBinaryBranchGate(hiddenSize));
    kanGate = addChildBlock("kanGate", new EpsilonBinaryBranchGate(hiddenSize));
    riichiGate = addChildBlock("riichiGate", new EpsilonBinaryBranchGate(hiddenSize));
  }

  /**
   * 凍結推論パイプラインが同時に保持する位置数を指定して方策実行境界を作る。
   *
   * <p>新しいブロックやパラメーターは登録しない。位置数はEAGER 順伝播の所有権上限と、Fusion セッションが確保する再利用する出力 バッファ数の両方に使う。
   *
   * @param manager パイプラインと同じデバイスを持つ長寿命管理元
   * @param parameterStore 凍結パラメーターの取得先
   * @param runtimeParameters 埋め込み初期化へ渡す実行時パラメーター
   * @param fusionSettings 各推論構成部分のFusion実行方式
   * @param expectedDataType 凍結パラメーターに要求する推論データ型
   * @param maxBatch 通常容量区分の最大行数
   * @param multiTransitionMaxBatch 複数遷移容量区分の最大行数
   * @param executionSlots 同時に保持できる未回収順伝播数
   * @param frozenParameters パラメーターが方策実行境界の寿命中に更新されないなら{@code true}
   * @return パイプライン-局所的な方策推論実行境界
   */
  public DecisionPolicyExecution newExecution(
      NDManager manager,
      ParameterStore parameterStore,
      PairList<String, Object> runtimeParameters,
      DecisionInferenceFusionSettings fusionSettings,
      DataType expectedDataType,
      int maxBatch,
      int multiTransitionMaxBatch,
      int executionSlots,
      boolean frozenParameters) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    validateFusionEligibility(fusionSettings, frozenParameters, manager.getDevice().isGpu());
    DecisionPolicyAffineExecution affine =
        switch (fusionSettings.policyAffine()) {
          case EAGER ->
              DecisionPolicyAffineExecution.eager(
                  candidateFusion,
                  transitionFusion,
                  alternativeHidden,
                  alternativeOffset,
                  alternativeIds,
                  hiddenSize,
                  executionSlots,
                  manager,
                  parameterStore,
                  expectedDataType,
                  usePackedTransitionProjection(
                      fusionSettings.policyAffine(),
                      manager.getDevice().isGpu(),
                      frozenParameters,
                      expectedDataType));
          case FUSION ->
              new DecisionPolicyFusionAffineExecution(
                  manager,
                  parameterStore,
                  candidateFusion,
                  transitionFusion,
                  alternativeHidden,
                  alternativeOffset,
                  hiddenSize,
                  expectedDataType,
                  maxBatch,
                  multiTransitionMaxBatch,
                  executionSlots);
        };
    DecisionPolicyIndexedAffineExecution indexedAffine = null;
    DecisionPolicyCandidatePrefixExecution candidatePrefix = null;
    DecisionPolicyPlayerTileContextExecution playerTileContext = null;
    DecisionPolicyCandidateContextExecution candidateContext = null;
    DecisionPolicyContextExecution context = null;
    try {
      indexedAffine =
          switch (fusionSettings.policyIndexedAffine()) {
            case EAGER ->
                DecisionPolicyIndexedAffineExecution.eager(
                    riichiGate, callGate, kanGate, kyushuGate, executionSlots);
            case FUSION ->
                new DecisionPolicyFusionIndexedAffineExecution(
                    manager,
                    parameterStore,
                    riichiGate,
                    callGate,
                    kanGate,
                    kyushuGate,
                    hiddenSize,
                    expectedDataType,
                    maxBatch,
                    executionSlots);
          };
      candidatePrefix =
          fusionSettings.candidatePrefix() == DecisionInferenceFusionMode.FUSION
              ? new DecisionPolicyFusionCandidatePrefixExecution(
                  manager,
                  parameterStore,
                  candidateFusion,
                  expectedDataType,
                  maxBatch,
                  executionSlots)
              : null;
      playerTileContext =
          fusionSettings.playerTileContext() == DecisionInferenceFusionMode.FUSION
              ? new DecisionPolicyFusionPlayerTileContextExecution(
                  manager,
                  parameterStore,
                  playerTileContextMixer,
                  expectedDataType,
                  maxBatch,
                  executionSlots)
              : DecisionPolicyPlayerTileContextExecution.eager(
                  playerTileContextMixer, executionSlots);
      candidateContext =
          fusionSettings.candidateContext() == DecisionInferenceFusionMode.FUSION
              ? new DecisionPolicyFusionCandidateContextExecution(
                  manager, hiddenSize, expectedDataType, maxBatch, executionSlots)
              : DecisionPolicyCandidateContextExecution.eager(executionSlots);
      context =
          fusionSettings.binaryContext() == DecisionInferenceFusionMode.FUSION
              ? new DecisionPolicyFusionContextExecution(
                  manager, hiddenSize, expectedDataType, maxBatch, executionSlots)
              : DecisionPolicyContextExecution.eager(executionSlots);
      return new DecisionPolicyExecution(
          affine,
          indexedAffine,
          candidatePrefix,
          playerTileContext,
          candidateContext,
          context,
          executionSlots);
    } catch (RuntimeException | Error constructionFailure) {
      if (context != null) {
        try {
          context.close();
        } catch (Throwable closeFailure) {
          constructionFailure.addSuppressed(closeFailure);
        }
      }
      if (candidateContext != null) {
        try {
          candidateContext.close();
        } catch (Throwable closeFailure) {
          constructionFailure.addSuppressed(closeFailure);
        }
      }
      if (playerTileContext != null) {
        try {
          playerTileContext.close();
        } catch (Throwable closeFailure) {
          constructionFailure.addSuppressed(closeFailure);
        }
      }
      if (candidatePrefix != null) {
        try {
          candidatePrefix.close();
        } catch (Throwable closeFailure) {
          constructionFailure.addSuppressed(closeFailure);
        }
      }
      if (indexedAffine != null) {
        try {
          indexedAffine.close();
        } catch (Throwable closeFailure) {
          constructionFailure.addSuppressed(closeFailure);
        }
      }
      try {
        affine.close();
      } catch (Throwable closeFailure) {
        constructionFailure.addSuppressed(closeFailure);
      }
      throw constructionFailure;
    }
  }

  /** 方策単体の公開生成処理でも、Fusionが要求するデバイスとパラメーター寿命を検証する。 */
  static void validateFusionEligibility(
      DecisionInferenceFusionSettings fusionSettings, boolean frozenParameters, boolean gpu) {
    boolean parameterBoundFusion =
        fusionSettings.policyAffine() == DecisionInferenceFusionMode.FUSION
            || fusionSettings.policyIndexedAffine() == DecisionInferenceFusionMode.FUSION
            || fusionSettings.candidatePrefix() == DecisionInferenceFusionMode.FUSION
            || fusionSettings.playerTileContext() == DecisionInferenceFusionMode.FUSION;
    boolean policyFusion =
        parameterBoundFusion
            || fusionSettings.candidateContext() == DecisionInferenceFusionMode.FUSION
            || fusionSettings.binaryContext() == DecisionInferenceFusionMode.FUSION;
    if (parameterBoundFusion && !frozenParameters) {
      throw new IllegalArgumentException("Policy FUSION execution requires frozen parameters");
    }
    if (policyFusion && !gpu) {
      throw new UnsupportedOperationException(
          "Policy FUSION execution requires a GPU inference device");
    }
  }

  /** 必要な要素数だけの遷移連結は凍結済みFLOAT16のグラフ外GPU EAGERでだけ利用する。 */
  static boolean usePackedTransitionProjection(
      DecisionInferenceFusionMode affineMode,
      boolean gpu,
      boolean frozenParameters,
      DataType dataType) {
    return affineMode == DecisionInferenceFusionMode.EAGER
        && gpu
        && frozenParameters
        && dataType == DataType.FLOAT16;
  }

  /**
   * 一容量区分内の全行動候補と遷移を一括採点する。
   *
   * @param parameterStore 順伝播に使うパラメーターの取得先
   * @param encodedState 状態埋め込み、牌トークン、プレイヤー-局所的なメモリ
   * @param inputs 行動・遷移・wait の型付きテンソル
   * @param training 学習固有のブロック挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 方策グラフが消費する分岐・候補スコア
   */
  public DecisionPolicyScores score(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionNetworkInputs inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return score(parameterStore, encodedState, inputs, null, training, runtimeParameters);
  }

  /**
   * 存在する遷移のインデックスを利用し、パディング遷移の計算を省いて候補を採点する。
   *
   * @param parameterStore 順伝播に使うパラメーターの取得先
   * @param encodedState 状態埋め込み、牌トークン、プレイヤー-局所的なメモリ
   * @param inputs 行動・遷移・wait の型付きテンソル
   * @param transitionPresentIndices PRESENT 遷移の一次元にしたインデックス。省略時は {@code null}
   * @param training 学習固有のブロック挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 方策グラフが消費する分岐・候補スコア
   */
  public DecisionPolicyScores score(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionNetworkInputs inputs,
      NDArray transitionPresentIndices,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training) {
      return scoreInternal(
          parameterStore,
          encodedState,
          inputs,
          TransitionInputs.dense(inputs),
          transitionPresentIndices,
          true,
          runtimeParameters);
    }
    NDManager outputManager = encodedState.stateEmbedding().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          encodedState.stateEmbedding(),
          encodedState.tileEmbeddings(),
          encodedState.playerMemoryKeyValues(),
          encodedState.playerMemoryMask());
      tempAttachTileProjection(scope, encodedState);
      scope.tempAttachAll(inputs.toNDList().toArray(new NDArray[0]));
      if (transitionPresentIndices != null) {
        scope.tempAttachAll(transitionPresentIndices);
      }
      DecisionPolicyScores scores =
          scoreInternal(
              parameterStore,
              encodedState,
              inputs,
              TransitionInputs.dense(inputs),
              transitionPresentIndices,
              false,
              runtimeParameters);
      outputManager.attachAll(scores.toNDList());
      return scores;
    }
  }

  /** ホストの存在する順にコンパクト済みの遷移テンソルを指定位置の抽出せず採点する。 */
  public DecisionPolicyScores scoreInference(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionInferenceInputs inputs,
      NDArray transitionPresentIndices,
      DecisionInferenceDeviceBatch.PolicyExecutionIndices policyExecutionIndices,
      DecisionPolicyExecution.Forward policyForward,
      PairList<String, Object> runtimeParameters) {
    if (inputs.transitionCount() == 0) {
      throw new IllegalArgumentException("eager Policy input has no present transition");
    }
    if (!encodedState.stateEmbedding().getDevice().isGpu()) {
      policyExecutionIndices = null;
    }
    NDManager outputManager = encodedState.stateEmbedding().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          encodedState.stateEmbedding(),
          encodedState.tileEmbeddings(),
          encodedState.playerMemoryKeyValues(),
          encodedState.playerMemoryMask());
      tempAttachTileProjection(scope, encodedState);
      scope.tempAttachAll(inputs.arrays());
      scope.tempAttachAll(transitionPresentIndices);
      if (policyExecutionIndices != null) {
        scope.tempAttachAll(policyExecutionIndices.packedIndices());
      }
      DecisionPolicyScores scores =
          scoreInternal(
              parameterStore,
              encodedState,
              inputs,
              TransitionInputs.compact(inputs),
              transitionPresentIndices,
              policyExecutionIndices,
              policyForward,
              false,
              runtimeParameters);
      outputManager.attachAll(scores.toNDList());
      return scores;
    }
  }

  private DecisionPolicyScores scoreInternal(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyInputs inputs,
      TransitionInputs transitionInputs,
      NDArray transitionPresentIndices,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return scoreInternal(
        parameterStore,
        encodedState,
        inputs,
        transitionInputs,
        transitionPresentIndices,
        null,
        null,
        training,
        runtimeParameters);
  }

  private DecisionPolicyScores scoreInternal(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyInputs inputs,
      TransitionInputs transitionInputs,
      NDArray transitionPresentIndices,
      DecisionInferenceDeviceBatch.PolicyExecutionIndices policyExecutionIndices,
      DecisionPolicyExecution.Forward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long rowCount = inputs.rowCount();
    int legalActionCapacity = inputs.bucket().legalActionCapacity();
    NDArray legalActionMask =
        inputs
            .legalActionMask()
            .toType(encodedState.stateEmbedding().getDataType(), false)
            .stopGradient();
    DecisionPolicyCandidates.CandidateMasks candidateMasks =
        DecisionPolicyCandidates.candidateMasks(inputs.actionCategories(), legalActionMask);
    PlayerTileContexts playerTileContexts =
        encodePlayerTileContexts(
            parameterStore, encodedState, policyForward, training, runtimeParameters);

    TransitionSet transitions =
        encodeTransitions(
            parameterStore,
            encodedState,
            inputs,
            transitionInputs,
            playerTileContexts.allPlayers(),
            transitionPresentIndices,
            policyForward,
            training,
            runtimeParameters,
            rowCount);
    CandidateSet candidates =
        encodeDenseCandidates(
            parameterStore,
            encodedState,
            inputs,
            playerTileContexts,
            transitions,
            legalActionMask,
            candidateMasks,
            policyForward,
            training,
            runtimeParameters,
            rowCount,
            legalActionCapacity);
    NDArray candidateEmbeddings = candidates.embeddings();
    NDArray rootCandidateScores = candidates.scores();
    NDArray actionTypes =
        inputs.actionCategories().get("...,{}", DecisionInputSchema.ActionInt.TYPE.ordinal());
    NDArray ronPointGateFeatures =
        pointGateFeatures(candidates.gateFeatures(), actionTypes, Action.Type.RON_AGARI);
    NDArray tsumoPointGateFeatures =
        pointGateFeatures(candidates.gateFeatures(), actionTypes, Action.Type.TSUMO_AGARI);
    DecisionPolicyCandidates.DiscardChoices discardChoices =
        DecisionPolicyCandidates.discardChoices(inputs.actionRoutes(), candidateMasks);
    NDArray discardRootMask = discardChoices.discardMask();
    NDArray firstTransitionDiscardScores = transitions.discardScores().get("...,:,0");
    NDArray sharedPhysicalDiscardScores =
        DecisionPolicyContextPool.sharePhysicalDiscardScores(
            firstTransitionDiscardScores, discardChoices);
    NDArray actionCandidateScores =
        rootCandidateScores
            .mul(discardRootMask.neg().add(1.0f))
            .add(sharedPhysicalDiscardScores.mul(discardRootMask))
            .mul(legalActionMask);
    DecisionPolicyContextPool.DiscardBranchContexts discardBranchContexts =
        DecisionPolicyContextPool.gatherDiscardBranches(candidateEmbeddings, discardChoices);

    NDArray riichiGateScores =
        scoreRiichiVersusDama(
            parameterStore,
            encodedState.stateEmbedding(),
            discardBranchContexts,
            discardChoices,
            rowCount,
            legalActionCapacity,
            policyExecutionIndices,
            policyForward,
            training,
            runtimeParameters);
    NDArray alternativeScores =
        scoreAlternativesFromCandidatePolicy(
            parameterStore,
            encodedState.stateEmbedding(),
            candidateEmbeddings,
            candidateMasks,
            actionCandidateScores,
            riichiGateScores,
            discardChoices,
            discardBranchContexts,
            ronPointGateFeatures,
            tsumoPointGateFeatures,
            rowCount,
            policyExecutionIndices,
            policyForward,
            training,
            runtimeParameters);
    return new DecisionPolicyScores(alternativeScores, actionCandidateScores, riichiGateScores);
  }

  private CandidateSet encodeDenseCandidates(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyInputs inputs,
      PlayerTileContexts playerTileContexts,
      TransitionSet transitions,
      NDArray legalActionMask,
      DecisionPolicyCandidates.CandidateMasks candidateMasks,
      DecisionPolicyExecution.Forward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount,
      int actionCapacity) {
    EpsilonPointProjection.Projection actionPoints =
        pointProjection.projectActions(
            inputs.pointLedger100(),
            inputs.stateCategories(),
            inputs.actionCategories(),
            inputs.actionWinFacts(),
            inputs instanceof DecisionInferenceInputs inference
                ? inference.winningActions()
                : null);
    NDArray pointContext =
        pointOutcomeEncoder.encodeSelected(
            parameterStore,
            actionPoints.features().toType(encodedState.stateEmbedding().getDataType(), false),
            actionPoints.validMask(),
            inputs instanceof DecisionInferenceInputs inference ? inference.winningActions() : null,
            training,
            runtimeParameters);
    NDArray actionFeatures =
        encodeActionFeatures(
            parameterStore,
            inputs.actionCategories(),
            pointContext,
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
    NDArray scores =
        scoreRootCandidates(parameterStore, candidateEmbeddings, training, runtimeParameters)
            .reshape(rowCount, actionCapacity);
    return new CandidateSet(
        candidateEmbeddings,
        scores,
        actionPoints.gateFeatures().toType(candidateEmbeddings.getDataType(), false));
  }

  private static NDArray pointGateFeatures(
      NDArray actionGateFeatures, NDArray actionTypes, Action.Type actionType) {
    NDArray typeMask =
        actionTypes
            .eq(DecisionFeatureCodec.actionType(actionType))
            .toType(actionGateFeatures.getDataType(), false)
            .expandDims(2);
    return actionGateFeatures.mul(typeMask).sum(new int[] {1});
  }

  private NDArray encodeActionFeatures(
      ParameterStore parameterStore,
      NDArray actionCategories,
      NDArray pointContext,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount,
      int actionCapacity) {
    NDArray actionCategoryIds = actionCategories.stopGradient().add(actionCategoryOffsets);
    NDArray actionCategoryEmbeddings =
        actionFeatureEmbedding
            .forward(parameterStore, new NDList(actionCategoryIds), training, runtimeParameters)
            .singletonOrThrow();
    return actionCategoryEmbeddings
        .reshape(
            rowCount, actionCapacity, DecisionInputSchema.ACTION_INT_STRIDE * CATEGORY_EMBEDDING)
        .concat(pointContext, 2);
  }

  private NDArray scoreRootCandidates(
      ParameterStore parameterStore,
      NDArray candidateEmbeddings,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return applyLinear(
        actionCandidateScoreHead,
        parameterStore,
        EpsilonMahjongStateEncoder.silu(
            applyLinear(
                actionCandidateHidden,
                parameterStore,
                candidateEmbeddings,
                training,
                runtimeParameters)),
        training,
        runtimeParameters);
  }

  private TransitionSet encodeTransitions(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyInputs inputs,
      TransitionInputs transitionInputs,
      NDArray allPlayerTileContexts,
      NDArray transitionPresentIndices,
      DecisionPolicyExecution.Forward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount) {
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
          rowCount);
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
          transitionInputs.waitWinFacts());
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
              rowCount);
      outputManager.attachAll(transitions.discardScores(), transitions.aggregatedEmbeddings());
      return transitions;
    }
  }

  private TransitionSet encodeTransitionsInternal(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyInputs inputs,
      TransitionInputs transitionInputs,
      NDArray allPlayerTileContexts,
      NDArray transitionPresentIndices,
      DecisionPolicyExecution.Forward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount) {
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
          inputs,
          allPlayerTileContexts,
          transitionPresentIndices,
          policyForward,
          training,
          runtimeParameters,
          rowCount,
          actionCapacity,
          transitionCapacity);
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
        projectTransitionTiles(
            parameterStore,
            encodedState.tileProjectionEmbeddings(),
            allPlayerTileContexts,
            training,
            runtimeParameters);
    NDArray waitPointContexts =
        encodeWaitPointContexts(
            parameterStore,
            inputs.pointLedger100(),
            inputs.stateCategories(),
            inputs.actionCategories().get("...,{}", DecisionInputSchema.ActionInt.TYPE.ordinal()),
            transitionInputs.tileCodes(),
            transitionInputs.waitTileIds(),
            transitionInputs.waitWinFacts(),
            encodedState.stateEmbedding().getDataType(),
            training,
            runtimeParameters);
    NDArray projectedWaitKeyValues =
        applyLinear(
            transitionWaitKeyValueProjection,
            parameterStore,
            waitPointContexts,
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
            projectedWaitKeyValues,
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
      NDArray transitionWeights =
          DecisionPolicyTensorOps.maskedTransitionWeights(discardScores, presentMask);
      aggregatedEmbeddings =
          transitionEmbeddings.mul(transitionWeights.expandDims(3)).sum(new int[] {2});
    }
    return new TransitionSet(discardScores, aggregatedEmbeddings);
  }

  /**
   * 複数遷移容量区分から実在遷移だけを集め、重い牌・待ち注意機構をパディングへ実行せず元配置へ戻す。
   *
   * <p>GPU推論では幅1でもホストで構築済みの存在するインデックスを使い、パディング行動に対する牌・待ち注意機構を省く。CPU推論と
   * 学習では幅1を密なまま保ち、幅が複数になる鳴き候補だけをコンパクト化する。
   */
  private TransitionSet encodeCompactedTransitions(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      TransitionInputs transitionInputs,
      DecisionPolicyInputs inputs,
      NDArray allPlayerTileContexts,
      NDArray suppliedPresentIndices,
      DecisionPolicyExecution.Forward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters,
      long rowCount,
      int actionCapacity,
      int transitionCapacity) {
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
    NDArray compactWaitWinFacts =
        transitionInputs.hostCompacted()
            ? transitionInputs.waitWinFacts()
            : DecisionPolicyTensorOps.compactTransitionRows(
                transitionInputs.waitWinFacts(), presentIndices, denseCount);

    NDArray compactBatchIds =
        DecisionPolicyTensorOps.compactBatchIds(
            presentIndices, rowCount, actionCapacity, transitionCapacity);
    NDArray compactCategoryIds = compactCategories.add(transitionCategoryOffsets);
    NDArray compactActionIndices =
        presentIndices.floorDivide(transitionCapacity).toType(DataType.INT64, false);
    NDArray compactActionTypes =
        EpsilonMaskedRows.gather(
            inputs
                .actionCategories()
                .get("...,{}", DecisionInputSchema.ActionInt.TYPE.ordinal())
                .reshape(Math.multiplyExact(rowCount, actionCapacity), 1),
            compactActionIndices);
    NDArray compactWaitKeyValues =
        !training && inputs instanceof DecisionInferenceInputs inference
            ? encodeSelectedWaitKeyValues(
                parameterStore,
                inputs.pointLedger100(),
                inputs.stateCategories(),
                compactActionTypes,
                compactTileCodes,
                compactWaitTileIds,
                compactWaitWinFacts,
                encodedState.stateEmbedding().getDataType(),
                runtimeParameters,
                inference.waitRows(),
                compactBatchIds)
            : applyLinear(
                transitionWaitKeyValueProjection,
                parameterStore,
                encodeWaitPointContexts(
                    parameterStore,
                    EpsilonMaskedRows.gather(inputs.pointLedger100(), compactBatchIds),
                    EpsilonMaskedRows.gather(inputs.stateCategories(), compactBatchIds),
                    compactActionTypes,
                    compactTileCodes,
                    compactWaitTileIds,
                    compactWaitWinFacts,
                    encodedState.stateEmbedding().getDataType(),
                    training,
                    runtimeParameters),
                training,
                runtimeParameters);
    NDArray compactCategoryEmbeddings =
        actionFeatureEmbedding
            .forward(parameterStore, new NDList(compactCategoryIds), training, runtimeParameters)
            .singletonOrThrow();
    NDArray compactFeatures =
        compactCategoryEmbeddings
            .reshape(
                presentCount,
                1,
                1,
                DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE * CATEGORY_EMBEDDING)
            .concat(compactNumerics, 3);
    NDArray compactFeatureEmbeddings =
        EpsilonMahjongStateEncoder.silu(
            applyLinear(
                transitionFeatureProjection,
                parameterStore,
                compactFeatures,
                training,
                runtimeParameters));
    NDArray projectedTileKeyValues =
        projectTransitionTiles(
            parameterStore,
            encodedState.tileProjectionEmbeddings(),
            allPlayerTileContexts,
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
            compactWaitKeyValues,
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
    NDArray compactStateContexts =
        EpsilonMaskedRows.gather(encodedState.stateEmbedding(), compactBatchIds)
            .reshape(presentCount, 1, 1, hiddenSize);
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
                    compactStateContexts))
            : policyForward.transition(
                parameterStore,
                runtimeParameters,
                compactFeatureEmbeddings,
                compactTileContexts,
                compactDiscardTileEmbeddings,
                compactDiscardSafetyContexts,
                compactStateContexts);
    compactTransitionEmbeddings =
        transitionLayerNorm
            .forward(
                parameterStore,
                new NDList(compactTransitionEmbeddings),
                training,
                runtimeParameters)
            .singletonOrThrow();

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
    NDArray transitionWeights =
        DecisionPolicyTensorOps.maskedTransitionWeights(discardScores, presentMask);
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

  EpsilonPointProjection pointProjection() {
    return pointProjection;
  }

  EpsilonPointOutcomeEncoder pointOutcomeEncoder() {
    return pointOutcomeEncoder;
  }

  NDArray encodeWaitPointContexts(
      ParameterStore parameterStore,
      NDArray pointLedger100,
      NDArray stateCategories,
      NDArray actionTypes,
      NDArray transitionTileCodes,
      NDArray waitTileIds,
      NDArray waitWinFacts,
      DataType dataType,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDManager outputManager = waitWinFacts.getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          pointLedger100,
          stateCategories,
          actionTypes,
          transitionTileCodes,
          waitTileIds,
          waitWinFacts);
      long rows = waitWinFacts.getShape().get(0);
      NDArray validRows =
          waitWinFacts
              .get("...,{}", DecisionInputSchema.WinFact.VALID.ordinal())
              .reshape(rows, -1)
              .sum(new int[] {1});
      NDArray waitRows = EpsilonMaskedRows.indices(validRows);
      if (waitRows.size() == 0) {
        // 待ちのないバッチでは固定計算を省き、空のEncoderだけでゼロ勾配を維持する。
        NDArray emptyContexts =
            pointOutcomeEncoder.encode(
                parameterStore,
                scope.zeros(new Shape(0, EpsilonPointProjection.FEATURE_WIDTH), dataType),
                training,
                runtimeParameters);
        Shape outputShape = waitTileIds.getShape().add(contextWidth);
        NDArray result =
            EpsilonMaskedRows.scatter(
                    emptyContexts.reshape(0, outputShape.size() / rows), waitRows, rows)
                .reshape(outputShape);
        outputManager.attachAll(result);
        return result;
      }
      pointLedger100 = EpsilonMaskedRows.gather(pointLedger100, waitRows);
      stateCategories = EpsilonMaskedRows.gather(stateCategories, waitRows);
      actionTypes = EpsilonMaskedRows.gather(actionTypes, waitRows);
      transitionTileCodes = EpsilonMaskedRows.gather(transitionTileCodes, waitRows);
      waitTileIds = EpsilonMaskedRows.gather(waitTileIds, waitRows);
      waitWinFacts = EpsilonMaskedRows.gather(waitWinFacts, waitRows);
      NDArray akaAvailable = waitAkaAvailable(transitionTileCodes, waitTileIds);
      EpsilonPointProjection.Projection projected =
          pointProjection.projectWaits(
              pointLedger100, stateCategories, actionTypes, waitWinFacts, akaAvailable);
      NDArray features = projected.features();
      NDArray presentIndices = EpsilonMaskedRows.indices(projected.validMask());
      NDArray presentContexts =
          pointOutcomeEncoder.encode(
              parameterStore,
              EpsilonMaskedRows.gather(
                      features.reshape(-1, EpsilonPointProjection.FEATURE_WIDTH), presentIndices)
                  .toType(dataType, false),
              training,
              runtimeParameters);
      long[] contextShape = features.getShape().getShape();
      contextShape[contextShape.length - 1] = presentContexts.getShape().get(1);
      NDArray contexts =
          EpsilonMaskedRows.scatter(presentContexts, presentIndices, projected.validMask().size())
              .reshape(contextShape);
      int scenarioAxis = contexts.getShape().dimension() - 2;
      NDArray count =
          projected
              .validMask()
              .sum(new int[] {scenarioAxis})
              .maximum(1.0f)
              .expandDims(scenarioAxis);
      NDArray pooled =
          contexts.sum(new int[] {scenarioAxis}).div(count.toType(contexts.getDataType(), false));
      long[] pooledShape = pooled.getShape().getShape();
      long width = 1;
      for (int axis = 1; axis < pooledShape.length; axis++) {
        width *= pooledShape[axis];
      }
      pooledShape[0] = rows;
      NDArray result =
          EpsilonMaskedRows.scatter(pooled.reshape(-1, width), waitRows, rows).reshape(pooledShape);
      outputManager.attachAll(result);
      return result;
    }
  }

  NDArray encodeSelectedWaitKeyValues(
      ParameterStore store,
      NDArray ledger,
      NDArray states,
      NDArray actions,
      NDArray tiles,
      NDArray waitIds,
      NDArray facts,
      DataType dataType,
      PairList<String, Object> parameters,
      NDArray indices,
      NDArray batchIds) {
    NDManager owner = facts.getManager();
    try (NDManager scope = owner.newSubManager()) {
      scope.tempAttachAll(ledger, states, actions, tiles, waitIds, facts);
      NDArray pooled =
          poolSelectedWaitSlots(
              store,
              ledger,
              states,
              actions,
              tiles,
              waitIds,
              facts,
              dataType,
              parameters,
              indices,
              batchIds);
      int width = TRANSITION_TILE_KEY_WIDTH + contextWidth;
      // 無効な待ちも密なLinearのゼロ入力と同じバイアス・演算型を保持する。
      NDArray zeroProjection =
          applyLinear(
              transitionWaitKeyValueProjection,
              store,
              scope.zeros(new Shape(1, contextWidth), pooled.getDataType()),
              false,
              parameters);
      NDArray output = zeroProjection.broadcast(waitIds.size(), width).duplicate();
      if (indices.size() != 0) {
        NDArray selected =
            applyLinear(transitionWaitKeyValueProjection, store, pooled, false, parameters);
        output.set(new NDIndex("{}", indices), selected);
      }
      output = output.reshape(waitIds.getShape().add(width));
      owner.attachAll(output);
      return output;
    }
  }

  private NDArray poolSelectedWaitSlots(
      ParameterStore store,
      NDArray ledger,
      NDArray states,
      NDArray actions,
      NDArray tiles,
      NDArray waitIds,
      NDArray facts,
      DataType dataType,
      PairList<String, Object> parameters,
      NDArray indices,
      NDArray batchIds) {
    NDManager owner = facts.getManager();
    try (NDManager scope = owner.newSubManager()) {
      scope.tempAttachAll(ledger, states, actions, tiles, waitIds, facts);
      long rows = facts.getShape().get(0);
      long slots = waitIds.size() / rows;
      NDArray output;
      if (indices.size() == 0) {
        output = scope.zeros(new Shape(0, contextWidth), dataType);
      } else {
        NDArray rowIndices = indices.floorDivide(slots).toType(DataType.INT64, false);
        NDArray selectedIds =
            EpsilonMaskedRows.gather(waitIds.reshape(-1, 1), indices).reshape(-1, 1, 1, 1);
        NDArray selectedTiles = EpsilonMaskedRows.gather(tiles, rowIndices);
        NDArray stateRows =
            batchIds == null
                ? rowIndices
                : EpsilonMaskedRows.gather(batchIds.reshape(-1, 1), rowIndices).reshape(-1);
        var projected =
            pointProjection.projectWaits(
                EpsilonMaskedRows.gather(ledger, stateRows),
                EpsilonMaskedRows.gather(states, stateRows),
                EpsilonMaskedRows.gather(actions, rowIndices),
                EpsilonMaskedRows.gather(facts.reshape(rows * slots, -1), indices)
                    .reshape(
                        -1,
                        1,
                        1,
                        1,
                        DecisionInputSchema.WaitWinType.values().length,
                        DecisionInputSchema.WinFact.values().length),
                waitAkaAvailable(selectedTiles, selectedIds));
        NDArray contexts =
            pointOutcomeEncoder.encode(
                store, projected.features().toType(dataType, false), false, parameters);
        contexts =
            contexts.mul(projected.validMask().toType(contexts.getDataType(), false).expandDims(5));
        output =
            contexts
                .sum(new int[] {4})
                .div(
                    projected
                        .validMask()
                        .sum(new int[] {4})
                        .maximum(1)
                        .toType(contexts.getDataType(), false)
                        .expandDims(4))
                .reshape(-1, contextWidth);
      }
      owner.attachAll(output);
      return output;
    }
  }

  private static NDArray waitAkaAvailable(NDArray transitionTileCodes, NDArray waitTileIds) {
    int tileAxis = transitionTileCodes.getShape().dimension() - 1;
    NDArray indices = waitTileIds.toType(DataType.INT64, false).maximum(1).sub(1).stopGradient();
    return transitionTileCodes
        .gather(indices, tileAxis)
        .toType(DataType.INT32, false)
        .maximum(1)
        .sub(1)
        .mod(2)
        .mul(waitTileIds.neq(0).toType(DataType.INT32, false))
        .stopGradient();
  }

  private NDArray attendTransitionTiles(
      ParameterStore parameterStore,
      NDArray projectedTileKeyValues,
      NDArray projectedTileGroupIndices,
      NDArray transitionFeatureEmbeddings,
      NDArray transitionTileCodes,
      NDArray waitTileIds,
      NDArray projectedWaitKeyValues,
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
          projectedWaitKeyValues,
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
          projectedWaitKeyValues);
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
              projectedWaitKeyValues,
              rowCount,
              actionCapacity,
              transitionCapacity,
              false,
              runtimeParameters);
      outputManager.attachAll(context);
      return context;
    }
  }

  /**
   * 遷移クエリへ、元バッチの34牌テーブルとクエリ固有の関係・待ち差分をまとめて注意機構する。
   *
   * <p>密な/コンパクト、学習/推論、演算データ型による経路分岐は持たない。クエリごとのINT32 バッチ IDを渡し、共有牌テーブルと関係
   * パラメーターテーブルを複製せず同じ対応付けを使う注意機構演算へ委ねる。
   */
  private NDArray attendTransitionTilesInternal(
      ParameterStore parameterStore,
      NDArray projectedTileKeyValues,
      NDArray projectedTileGroupIndices,
      NDArray transitionFeatureEmbeddings,
      NDArray transitionTileCodes,
      NDArray waitTileIds,
      NDArray projectedWaitKeyValues,
      long rowCount,
      int actionCapacity,
      int transitionCapacity,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    long candidatesPerState = Math.multiplyExact(actionCapacity, transitionCapacity);
    long candidateCount = Math.multiplyExact(rowCount, candidatesPerState);
    NDArray storedRelationCodes = transitionTileCodes.stopGradient().toType(DataType.INT32, false);
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

  /**
   * チェックポイント互換の関係埋め込みパラメーターをテーブルのまま注意機構へ渡す。
   *
   * <p>混合精度時だけクエリデータ型へ微分可能なcastを行う。変換結果はバッチ管理元へ所属させ、長寿命のモデル 管理元へ小バッチごとの一時テンソルを残さない。
   */
  private NDArray transitionTileRelationTable(
      ParameterStore parameterStore, NDArray query, boolean training) {
    NDArray relationTable =
        transitionTileRelationEmbedding.getValue(parameterStore, query.getDevice(), training);
    if (relationTable.getDataType() == query.getDataType()) {
      return relationTable;
    }
    NDArray converted = relationTable.getNDArrayInternal().differentiableCast(query.getDataType());
    converted.attach(query.getManager());
    return converted;
  }

  /** プレイヤー別メモリを牌種クエリで読み、プレイヤー軸保持表現と全プレイヤー融合表現を一度だけ構築する。 */
  private PlayerTileContexts encodePlayerTileContexts(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      DecisionPolicyExecution.Forward policyForward,
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
          encodedState.playerMemoryKeyValues(),
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
      DecisionPolicyExecution.Forward policyForward,
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

  /**
   * 各牌種をクエリとして、相手ごとにプレイヤートークン・河・面子だけを読む。
   *
   * <p>結果はプレイヤー軸と牌種軸を保った {@code [batch,4,34,context]}。全プレイヤー融合や打牌元のプレイヤー抽出より前に
   * プレイヤー軸を潰さないため、「この牌を特定相手の河・副露へ切る／その相手の捨て牌へ応答する」対応を保持する。
   */
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
    return attendProjectedPlayerMemories(
        queries, encodedState.playerMemoryKeyValues(), encodedState.playerMemoryMask());
  }

  /** トークン優先射影を転置せず、プレイヤー別の短系列注意機構へ渡す。 */
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

  /** 34牌のK/Vへ公開履歴文脈を一度だけ足し、全行動遷移で共有する。 */
  NDArray projectTransitionTiles(
      ParameterStore parameterStore,
      NDArray tileEmbeddings,
      NDArray publicContexts,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray tileKeyValues =
        EpsilonLinear.projectBatchSlices(
            transitionTileKeyValueProjection,
            parameterStore,
            tileEmbeddings,
            training,
            runtimeParameters);
    NDArray publicKeyValues =
        applyLinear(
            transitionPublicContextProjection,
            parameterStore,
            publicContexts,
            training,
            runtimeParameters);
    return tileKeyValues.add(publicKeyValues);
  }

  /** 同一物理打牌のDAMA表現とRIICHI表現を直接比較し、RIICHI 位置へ二値ロジットを置く。 */
  private NDArray scoreRiichiVersusDama(
      ParameterStore parameterStore,
      NDArray stateEmbedding,
      DecisionPolicyContextPool.DiscardBranchContexts discardBranchContexts,
      DecisionPolicyCandidates.DiscardChoices discardChoices,
      long rowCount,
      int actionCapacity,
      DecisionInferenceDeviceBatch.PolicyExecutionIndices policyExecutionIndices,
      DecisionPolicyExecution.Forward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray stateContext = stateEmbedding.reshape(rowCount, 1, hiddenSize);
    NDArray active =
        discardChoices.damaPresent().mul(discardChoices.riichiPresent()).stopGradient();
    if (policyExecutionIndices != null) {
      if (policyForward != null) {
        return policyForward.scoreActions(
            parameterStore,
            stateEmbedding,
            discardBranchContexts.dama(),
            discardBranchContexts.riichi(),
            policyExecutionIndices.riichiActions(),
            rowCount,
            actionCapacity,
            runtimeParameters);
      }
      return riichiGate.scoreActions(
          parameterStore,
          stateEmbedding,
          discardBranchContexts.dama(),
          discardBranchContexts.riichi(),
          policyExecutionIndices.riichiActions(),
          rowCount,
          actionCapacity,
          training,
          runtimeParameters);
    }
    return riichiGate
        .scoreBroadcastState(
            parameterStore,
            stateContext,
            discardBranchContexts.dama(),
            discardBranchContexts.riichi(),
            training,
            runtimeParameters)
        .reshape(rowCount, actionCapacity)
        .mul(active)
        .mul(discardChoices.riichiMask());
  }

  /**
   * 方策グラフと同じ条件付き子分布から各選択肢と分岐の文脈を作り、上位節点を採点する。
   *
   * <p>子分布の重みは勾配を遮断する。したがって上位損失は子ロジットを直接操作せず、選ばれた表現経路と上位採点器だけを学習する。
   */
  private NDArray scoreAlternativesFromCandidatePolicy(
      ParameterStore parameterStore,
      NDArray stateEmbedding,
      NDArray candidateEmbeddings,
      DecisionPolicyCandidates.CandidateMasks candidateMasks,
      NDArray actionCandidateScores,
      NDArray riichiGateScores,
      DecisionPolicyCandidates.DiscardChoices discardChoices,
      DecisionPolicyContextPool.DiscardBranchContexts discardBranchContexts,
      NDArray ronPointGateFeatures,
      NDArray tsumoPointGateFeatures,
      long rowCount,
      DecisionInferenceDeviceBatch.PolicyExecutionIndices policyExecutionIndices,
      DecisionPolicyExecution.Forward policyForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    DecisionPolicyContextPool.MappedCandidateContexts mappedCandidateContexts =
        policyForward == null
            ? DecisionPolicyContextPool.poolMappedCandidateTypes(
                candidateEmbeddings, actionCandidateScores, candidateMasks)
            : policyForward.candidateContexts(
                candidateEmbeddings, actionCandidateScores, candidateMasks);
    NDArray[] conditionalCandidateContexts = new NDArray[DecisionAlternative.NETWORK_SIZE];
    NDArray[] alternativePresenceByIndex = new NDArray[DecisionAlternative.NETWORK_SIZE];
    for (DecisionAlternative alternative : DecisionAlternative.values()) {
      conditionalCandidateContexts[alternative.networkIndex()] =
          mappedCandidateContexts.context(alternative);
      alternativePresenceByIndex[alternative.networkIndex()] =
          mappedCandidateContexts.present(alternative);
    }
    NDArray candidateContexts = mappedCandidateContexts.alternativeContexts();
    NDArray alternativePresenceMask = mappedCandidateContexts.alternativePresence();
    NDArray stateContext = stateEmbedding.reshape(rowCount, 1, hiddenSize);
    NDArray alternativeHiddenContexts;
    if (policyForward == null) {
      NDManager batchManager = stateEmbedding.getManager();
      NDArray offsets;
      try (NDManager embeddingScope = batchManager.newSubManager()) {
        embeddingScope.tempAttachAll(alternativeIds);
        offsets =
            alternativeOffset
                .forward(parameterStore, new NDList(alternativeIds), training, runtimeParameters)
                .singletonOrThrow();
        batchManager.attachAll(offsets);
      }
      alternativeHiddenContexts =
          EpsilonMahjongStateEncoder.silu(
              EpsilonPartitionedLinear.apply(
                      alternativeHidden, parameterStore, training, stateContext, candidateContexts)
                  .add(offsets.reshape(1, DecisionAlternative.NETWORK_SIZE, hiddenSize)));
    } else {
      alternativeHiddenContexts =
          policyForward.alternative(
              parameterStore, runtimeParameters, stateContext, candidateContexts);
    }
    NDArray scoredAlternatives =
        applyLinear(
                alternativeScoreHead,
                parameterStore,
                alternativeHiddenContexts,
                training,
                runtimeParameters)
            .reshape(rowCount, DecisionAlternative.NETWORK_SIZE)
            .mul(alternativePresenceMask);

    NDArray passContext = mappedCandidateContexts.passContext();
    NDArray meldContext =
        DecisionPolicyContextPool.aggregateAlternativeContexts(
            scoredAlternatives,
            candidateContexts,
            alternativePresenceMask,
            DecisionAlternative.CHI,
            DecisionAlternative.PON,
            DecisionAlternative.DAIMINKAN);
    NDArray passPresence = mappedCandidateContexts.passPresence();
    NDArray meldPresence =
        DecisionPolicyContextPool.presentAny(
            alternativePresenceByIndex,
            DecisionAlternative.CHI,
            DecisionAlternative.PON,
            DecisionAlternative.DAIMINKAN);
    NDArray callScore =
        scoreGate(
            DecisionPolicyIndexedAffineExecution.Site.CALL,
            callGate,
            policyForward,
            parameterStore,
            stateEmbedding,
            passContext,
            meldContext,
            passPresence,
            meldPresence,
            policyExecutionIndices == null ? null : policyExecutionIndices.callRows(),
            rowCount,
            training,
            runtimeParameters);

    NDArray declinedRonContext =
        mixBinaryBranchContexts(
            policyForward,
            DecisionPolicyContextExecution.Site.CALL,
            passContext,
            meldContext,
            callScore,
            passPresence,
            meldPresence);
    NDArray declinedRonPresence =
        DecisionPolicyContextPool.presentEither(passPresence, meldPresence);
    NDArray ronContext = conditionalCandidateContexts[DecisionAlternative.RON.networkIndex()];
    NDArray ronPresence = alternativePresenceByIndex[DecisionAlternative.RON.networkIndex()];
    NDArray ronScore =
        scorePointGate(
            ronGate,
            parameterStore,
            stateEmbedding,
            declinedRonContext,
            ronContext,
            ronPointGateFeatures,
            declinedRonPresence,
            ronPresence,
            policyExecutionIndices == null ? null : policyExecutionIndices.ronRows(),
            rowCount,
            training,
            runtimeParameters);

    NDArray continueContext =
        DecisionPolicyContextPool.poolDiscardContinuation(
            discardBranchContexts, actionCandidateScores, riichiGateScores, discardChoices);
    NDArray kanContext =
        DecisionPolicyContextPool.aggregateAlternativeContexts(
            scoredAlternatives,
            candidateContexts,
            alternativePresenceMask,
            DecisionAlternative.ANKAN,
            DecisionAlternative.KAKAN);
    NDArray continuePresence =
        DecisionPolicyContextPool.present(discardChoices.representativeMask());
    NDArray kanPresence =
        DecisionPolicyContextPool.presentAny(
            alternativePresenceByIndex, DecisionAlternative.ANKAN, DecisionAlternative.KAKAN);
    NDArray kanScore =
        scoreGate(
            DecisionPolicyIndexedAffineExecution.Site.KAN,
            kanGate,
            policyForward,
            parameterStore,
            stateEmbedding,
            continueContext,
            kanContext,
            continuePresence,
            kanPresence,
            policyExecutionIndices == null ? null : policyExecutionIndices.kanRows(),
            rowCount,
            training,
            runtimeParameters);

    NDArray continuedTurnContext =
        mixBinaryBranchContexts(
            policyForward,
            DecisionPolicyContextExecution.Site.KAN,
            continueContext,
            kanContext,
            kanScore,
            continuePresence,
            kanPresence);
    NDArray continuedTurnPresence =
        DecisionPolicyContextPool.presentEither(continuePresence, kanPresence);
    NDArray kyushuContext = conditionalCandidateContexts[DecisionAlternative.KYUSHU.networkIndex()];
    NDArray kyushuPresence = alternativePresenceByIndex[DecisionAlternative.KYUSHU.networkIndex()];
    NDArray kyushuScore =
        scoreGate(
            DecisionPolicyIndexedAffineExecution.Site.KYUSHU,
            kyushuGate,
            policyForward,
            parameterStore,
            stateEmbedding,
            continuedTurnContext,
            kyushuContext,
            continuedTurnPresence,
            kyushuPresence,
            policyExecutionIndices == null ? null : policyExecutionIndices.kyushuRows(),
            rowCount,
            training,
            runtimeParameters);
    NDArray declinedTsumoContext =
        mixBinaryBranchContexts(
            policyForward,
            DecisionPolicyContextExecution.Site.KYUSHU,
            continuedTurnContext,
            kyushuContext,
            kyushuScore,
            continuedTurnPresence,
            kyushuPresence);
    NDArray declinedTsumoPresence =
        DecisionPolicyContextPool.presentEither(continuedTurnPresence, kyushuPresence);
    NDArray tsumoContext = conditionalCandidateContexts[DecisionAlternative.TSUMO.networkIndex()];
    NDArray tsumoPresence = alternativePresenceByIndex[DecisionAlternative.TSUMO.networkIndex()];
    NDArray tsumoScore =
        scorePointGate(
            tsumoGate,
            parameterStore,
            stateEmbedding,
            declinedTsumoContext,
            tsumoContext,
            tsumoPointGateFeatures,
            declinedTsumoPresence,
            tsumoPresence,
            policyExecutionIndices == null ? null : policyExecutionIndices.tsumoRows(),
            rowCount,
            training,
            runtimeParameters);

    NDArray[] scoresByIndex = new NDArray[DecisionAlternative.NETWORK_SIZE];
    for (DecisionAlternative alternative : DecisionAlternative.values()) {
      NDArray score =
          scoredAlternatives.get(":,{}", alternative.networkIndex()).reshape(rowCount, 1);
      score =
          switch (alternative) {
            case RON -> ronScore;
            case CALL -> callScore;
            case TSUMO -> tsumoScore;
            case KYUSHU -> kyushuScore;
            case KAN -> kanScore;
            default -> score;
          };
      scoresByIndex[alternative.networkIndex()] = score;
    }
    return DecisionPolicyContextPool.concatenateAlternativeAxis(scoresByIndex)
        .reshape(rowCount, DecisionAlternative.NETWORK_SIZE);
  }

  private static NDArray mixBinaryBranchContexts(
      DecisionPolicyExecution.Forward policyForward,
      DecisionPolicyContextExecution.Site site,
      NDArray baselineContext,
      NDArray selectedContext,
      NDArray selectedLogit,
      NDArray baselinePresence,
      NDArray selectedPresence) {
    if (policyForward != null) {
      return policyForward.mixBinaryBranchContexts(
          site,
          baselineContext,
          selectedContext,
          selectedLogit,
          baselinePresence,
          selectedPresence);
    }
    return DecisionPolicyContextPool.mixBinaryBranchContexts(
        baselineContext, selectedContext, selectedLogit, baselinePresence, selectedPresence);
  }

  private static NDArray scoreGate(
      DecisionPolicyIndexedAffineExecution.Site site,
      EpsilonBinaryBranchGate gate,
      DecisionPolicyExecution.Forward policyForward,
      ParameterStore parameterStore,
      NDArray stateEmbedding,
      NDArray baselineContext,
      NDArray selectedContext,
      NDArray baselinePresence,
      NDArray selectedPresence,
      NDArray activeRows,
      long rowCount,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (activeRows != null) {
      if (policyForward != null) {
        return policyForward.scoreRows(
            site,
            parameterStore,
            stateEmbedding,
            baselineContext,
            selectedContext,
            activeRows,
            rowCount,
            runtimeParameters);
      }
      return gate.scoreRows(
          parameterStore,
          stateEmbedding,
          baselineContext,
          selectedContext,
          activeRows,
          rowCount,
          training,
          runtimeParameters);
    }
    // インデックス付きの経路はホストで構築したactiveRowsだけを使うため、密なマスクは代替処理内で遅延生成する。
    NDArray activeMask = baselinePresence.mul(selectedPresence);
    return gate.score(
            parameterStore,
            stateEmbedding,
            baselineContext,
            selectedContext,
            training,
            runtimeParameters)
        .mul(activeMask);
  }

  private static NDArray scorePointGate(
      EpsilonBinaryBranchGate gate,
      ParameterStore parameterStore,
      NDArray stateEmbedding,
      NDArray baselineContext,
      NDArray selectedContext,
      NDArray directFeatures,
      NDArray baselinePresence,
      NDArray selectedPresence,
      NDArray activeRows,
      long rowCount,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (activeRows != null) {
      return gate.scoreRows(
          parameterStore,
          stateEmbedding,
          baselineContext,
          selectedContext,
          directFeatures,
          activeRows,
          rowCount,
          training,
          runtimeParameters);
    }
    NDArray activeMask = baselinePresence.mul(selectedPresence);
    return gate.score(
            parameterStore,
            stateEmbedding,
            baselineContext,
            selectedContext,
            directFeatures,
            training,
            runtimeParameters)
        .mul(activeMask);
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    actionCategoryOffsets =
        manager
            .create(DecisionCategoryLayout.actionOffsets())
            .reshape(1, 1, DecisionInputSchema.ACTION_INT_STRIDE);
    transitionCategoryOffsets =
        manager
            .create(DecisionCategoryLayout.transitionOffsets())
            .reshape(1, 1, 1, DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE);
    alternativeIds = manager.create(ALTERNATIVE_IDS);
    actionFeatureEmbedding.initialize(
        manager, dataType, new Shape(-1, -1, DecisionInputSchema.ACTION_INT_STRIDE));
    transitionTileRelationEmbedding.initialize(
        manager, dataType, new Shape(-1, -1, -1, DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT));
    pointProjection.initialize(
        manager, dataType, new Shape(-1, EpsilonPointProjection.FEATURE_WIDTH));
    pointOutcomeEncoder.initialize(
        manager, dataType, new Shape(-1, EpsilonPointProjection.FEATURE_WIDTH));
    alternativeOffset.initialize(manager, dataType, new Shape(DecisionAlternative.NETWORK_SIZE));
    actionFeatureProjection.initialize(
        manager,
        dataType,
        new Shape(-1, DecisionInputSchema.ACTION_INT_STRIDE * CATEGORY_EMBEDDING + contextWidth));
    transitionFeatureProjection.initialize(
        manager,
        dataType,
        new Shape(
            -1,
            DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE * CATEGORY_EMBEDDING
                + DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE));
    transitionTileKeyValueProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    transitionPublicContextProjection.initialize(manager, dataType, new Shape(-1, contextWidth));
    transitionTileQueryProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    transitionWaitKeyValueProjection.initialize(manager, dataType, new Shape(-1, contextWidth));
    playerTileQueryProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    playerTileContextMixer.initialize(
        manager, dataType, new Shape(-1, GameState.NUM_PLAYERS, Tile.NUM_TILE_TYPES, contextWidth));
    transitionFusion.initialize(manager, dataType, new Shape(-1, hiddenSize));
    transitionLayerNorm.initialize(manager, dataType, new Shape(-1, -1, -1, hiddenSize));
    discardCandidateScorer.initialize(manager, dataType, new Shape(-1, hiddenSize));
    candidateFusion.initialize(manager, dataType, new Shape(-1, hiddenSize));
    candidateLayerNorm.initialize(manager, dataType, new Shape(-1, -1, hiddenSize));
    actionCandidateHidden.initialize(manager, dataType, new Shape(-1, hiddenSize));
    actionCandidateScoreHead.initialize(manager, dataType, new Shape(-1, hiddenSize));
    alternativeHidden.initialize(manager, dataType, new Shape(-1, hiddenSize * 2L));
    alternativeScoreHead.initialize(manager, dataType, new Shape(-1, hiddenSize));
    ronGate.initialize(manager, dataType, new Shape(-1, hiddenSize));
    callGate.initialize(manager, dataType, new Shape(-1, hiddenSize));
    tsumoGate.initialize(manager, dataType, new Shape(-1, hiddenSize));
    kyushuGate.initialize(manager, dataType, new Shape(-1, hiddenSize));
    kanGate.initialize(manager, dataType, new Shape(-1, hiddenSize));
    riichiGate.initialize(manager, dataType, new Shape(-1, hiddenSize));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    throw new UnsupportedOperationException("Use score with DecisionNetworkInputs");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {
      new Shape(-1, DecisionAlternative.NETWORK_SIZE), new Shape(-1, -1), new Shape(-1, -1)
    };
  }

  private Linear hiddenLinear() {
    return Linear.builder().setUnits(hiddenSize).build();
  }

  private static long[] alternativeIds() {
    long[] ids = new long[DecisionAlternative.NETWORK_SIZE];
    for (int index = 0; index < ids.length; index++) {
      ids[index] = index;
    }
    return ids;
  }

  /** 独立した低精度牌ビューだけを現在の一時有効範囲へ移し、Linear中間テンソルの寿命を局所化する。 */
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

  private record PlayerTileContexts(NDArray byPlayer, NDArray allPlayers) {}

  private record CandidateSet(NDArray embeddings, NDArray scores, NDArray gateFeatures) {}

  private record TransitionInputs(
      NDArray categories,
      NDArray numerics,
      NDArray tileCodes,
      NDArray waitTileIds,
      NDArray waitWinFacts,
      boolean hostCompacted) {

    private static TransitionInputs dense(DecisionNetworkInputs inputs) {
      return new TransitionInputs(
          inputs.transitionCategories(),
          inputs.transitionNumerics(),
          inputs.transitionTiles(),
          inputs.waitTileIds(),
          inputs.waitWinFacts(),
          false);
    }

    private static TransitionInputs compact(DecisionInferenceInputs inputs) {
      return new TransitionInputs(
          inputs.transitionCategories(),
          inputs.transitionNumerics(),
          inputs.transitionTiles(),
          inputs.waitTileIds(),
          inputs.waitWinFacts(),
          true);
    }
  }

  private record TransitionSet(NDArray discardScores, NDArray aggregatedEmbeddings) {}
}

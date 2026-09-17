package com.epsilon.nano.ai.decision.policy;

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
import com.epsilon.ai.model.EpsilonLinear;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.config.settings.DecisionInferenceFusionMode;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.input.DecisionCategoryLayout;
import com.epsilon.nano.ai.decision.input.DecisionInferenceDeviceBatch;
import com.epsilon.nano.ai.decision.input.DecisionInferenceInputs;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.nano.ai.decision.input.DecisionPolicyInputs;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionAffineExecution;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionCandidateContextExecution;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionCandidatePrefixExecution;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionContextExecution;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionIndexedAffineExecution;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionPlayerTileContextExecution;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;

/**
 * 状態、行動、行動適用後の遷移を採点し、麻雀の条件付き方策グラフへ渡す。
 *
 * <p>各行動は一件以上の遷移を持つ。チー・ポンでは直後に選べる打牌ごとに一件、通常の打牌・見送りでは一件、槓では嶺上牌を待つ遷移を一件、和了・九種九牌では終局への遷移を一件持つ。遷移集合は共有の打牌スコア計算処理によるマスク付きソフトマックスで重み付けし、一つの表現にまとめる。
 *
 * <p>通常の打牌と鳴き直後の打牌は、同じ {@link EpsilonDiscardCandidateScorer}
 * で採点する。鳴きの損失を計算する際は集約用の重みから勾配を切り離すため、鳴きの表現を学習しつつ、実行しなかった後続打牌のロジットは直接更新しない。打牌のスコア計算処理は、実際に選択された打牌の損失から学習する。行動、MELD・KAN
 * の種類、CALL・KAN の上位分岐も、実際の方策グラフが使う条件付き分布で再帰的に集約する。
 *
 * <p>遷移と34牌種の関係は、牌トークンを参照するマルチヘッド注意機構へ渡す。和了形になる待ち、ロン・ツモで役が成立する待ち、および公開情報から求まる役と得点を、対応する牌種の表現へ加える。役は転送時にビット列へまとめ、ネットワーク内では各役の埋め込みを加算するため、役の組合せを越えて同じパラメーターを使う。
 *
 * <p>34牌種のクエリで各プレイヤーの公開情報・河・副露の表現を一度だけ読み出す。打牌では牌種に対応する全プレイヤーの表現を、応答ではイベントの牌種と発生元のプレイヤーに対応する表現を取り出す。槓では対象牌の全プレイヤーの表現を使い、牌とプレイヤーの関係を保ったまま判断する。
 */
public final class EpsilonDecisionPolicyHead extends AbstractBlock {

  private static final int CATEGORY_EMBEDDING = 8;
  private static final int WAIT_YAKU_EMBEDDING = 16;
  private static final int TRANSITION_TILE_ATTENTION_HEADS = 4;
  private static final int TRANSITION_TILE_KEY_SIZE = 8;
  private static final int TRANSITION_TILE_KEY_WIDTH =
      TRANSITION_TILE_ATTENTION_HEADS * TRANSITION_TILE_KEY_SIZE;
  private static final int WAIT_YAKU_CODE_COUNT = 1 << DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK;
  private static final int MAX_CONTEXT_WIDTH = 64;
  private static final long[] ALTERNATIVE_IDS = alternativeIds();
  private static final long[] POLICY_NODE_IDS = policyNodeIds();

  private final int hiddenSize;
  private final int contextWidth;
  private final IdEmbedding actionFeatureEmbedding;
  private final IdEmbedding transitionTileRelationEmbedding;
  private final Linear transitionWaitYakuProjection;
  private final IdEmbedding policyNodeEmbedding;
  private final IdEmbedding alternativeEmbedding;
  private final Linear actionFeatureProjection;
  private final Linear transitionFeatureProjection;
  private final Linear transitionTileKeyValueProjection;
  private final Linear transitionTileQueryProjection;
  private final Linear transitionWaitKeyValueProjection;
  private final Linear playerTileQueryProjection;
  private final Linear playerMemoryKeyValueProjection;
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
  private NDArray waitYakuBitDivisors;
  private NDArray inferenceWaitYakuLookup;
  private NDArray alternativeIds;
  private NDArray policyNodeIds;

  /**
   * 指定隠れ層幅で遷移を考慮する方策スコア計算処理を構築する。
   *
   * @param hiddenSize 状態、行動、遷移埋め込みの共通幅
   */
  public EpsilonDecisionPolicyHead(int hiddenSize) {
    if (hiddenSize % TRANSITION_TILE_ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be divisible by transition tile attention heads");
    }
    this.hiddenSize = hiddenSize;
    contextWidth = Math.min(hiddenSize, MAX_CONTEXT_WIDTH);
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
    transitionWaitYakuProjection =
        addChildBlock(
            "transitionWaitYakuProjection",
            Linear.builder().setUnits(WAIT_YAKU_EMBEDDING).optBias(false).build());
    policyNodeEmbedding =
        addChildBlock(
            "policyNodeEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(DecisionPolicyNode.NETWORK_SIZE)
                .setEmbeddingSize(hiddenSize)
                .build());
    alternativeEmbedding =
        addChildBlock(
            "alternativeEmbedding",
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
    playerMemoryKeyValueProjection =
        addChildBlock(
            "playerMemoryKeyValueProjection",
            Linear.builder().setUnits(contextWidth * 2L).optBias(false).build());
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
    ronGate = addChildBlock("ronGate", new EpsilonBinaryBranchGate(hiddenSize));
    callGate = addChildBlock("callGate", new EpsilonBinaryBranchGate(hiddenSize));
    tsumoGate = addChildBlock("tsumoGate", new EpsilonBinaryBranchGate(hiddenSize));
    kyushuGate = addChildBlock("kyushuGate", new EpsilonBinaryBranchGate(hiddenSize));
    kanGate = addChildBlock("kanGate", new EpsilonBinaryBranchGate(hiddenSize));
    riichiGate = addChildBlock("riichiGate", new EpsilonBinaryBranchGate(hiddenSize));
  }

  /**
   * 凍結推論パイプラインが同時に保持する枠数を指定して方策実行境界を作る。
   *
   * <p>新しいブロックやパラメーターは登録しない。枠数はEAGER 順伝播の所有権上限と、Fusion セッションが確保する再利用する出力 バッファ数の両方に使う。
   *
   * @param manager パイプラインと同じデバイスを持つ長寿命管理元
   * @param parameterStore 凍結パラメーターの取得先
   * @param runtimeParameters 埋め込み初期化へ渡す実行時パラメーター
   * @param fusionSettings 各推論構成要素のFusion実行方式
   * @param expectedDataType 凍結パラメーターに要求する推論データ型
   * @param maxBatch 通常容量区分の最大行数
   * @param multiTransitionMaxBatch 複数遷移容量区分の最大行数
   * @param executionSlots 同時に保持できる未回収順伝播数
   * @param frozenParameters パラメーターが方策実行境界の寿命中に更新されないなら{@code true}
   * @return パイプライン専用の方策推論実行境界
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
                  policyNodeEmbedding,
                  alternativeEmbedding,
                  policyNodeIds,
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
                  runtimeParameters,
                  candidateFusion,
                  transitionFusion,
                  alternativeHidden,
                  policyNodeEmbedding,
                  alternativeEmbedding,
                  policyNodeIds,
                  alternativeIds,
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
                    riichiGate, callGate, ronGate, kanGate, kyushuGate, tsumoGate, executionSlots);
            case FUSION ->
                new DecisionPolicyFusionIndexedAffineExecution(
                    manager,
                    parameterStore,
                    riichiGate,
                    callGate,
                    ronGate,
                    kanGate,
                    kyushuGate,
                    tsumoGate,
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

  /** 必要な要素数に合わせた遷移連結は凍結済みFLOAT16の計算グラフ外GPU EAGERでだけ利用する。 */
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
   * 一容量区分内の全最初の行動と遷移を一括採点する。
   *
   * @param parameterStore 順伝播に使うパラメーターの取得先
   * @param encodedState 状態埋め込み、牌トークン、プレイヤーごとのメモリ
   * @param inputs 行動・遷移・待ちの型付きテンソル
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
   * @param encodedState 状態埋め込み、牌トークン、プレイヤーごとのメモリ
   * @param inputs 行動・遷移・待ちの型付きテンソル
   * @param transitionPresentIndices PRESENT 遷移の 1次元化したインデックス。省略時は {@code null}
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
          encodedState.playerMemory(),
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

  /** ホストの存在する順に有効な要素だけを集約済みの遷移テンソルを指定位置の要素を集めるせず採点する。 */
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
          encodedState.playerMemory(),
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
    NDArray scores =
        scoreRootCandidates(parameterStore, candidateEmbeddings, training, runtimeParameters)
            .reshape(rowCount, actionCapacity);
    return new CandidateSet(candidateEmbeddings, scores);
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
   * 学習では幅1を密なまま保ち、幅が複数になる鳴き候補だけを有効な要素だけに集約する。
   */
  private TransitionSet encodeCompactedTransitions(
      ParameterStore parameterStore,
      EpsilonMahjongStateEncoder.EncodedState encodedState,
      TransitionInputs transitionInputs,
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
    NDArray compactCategoryIds = compactCategories.add(transitionCategoryOffsets);
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

  /**
   * 遷移クエリへ、元バッチの34牌表とクエリ固有の関係・待ち差分をまとめて注意機構する。
   *
   * <p>密な/有効要素のみの、学習/推論、演算データ型による経路分岐は持たない。クエリごとのINT32 バッチ IDを渡し、共有牌表と関係
   * パラメーター表を複製せず同じ対応付けを使う注意機構演算へ委ねる。
   */
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

  /**
   * チェックポイント互換の関係埋め込みパラメーターを表のまま注意機構へ渡す。
   *
   * <p>混合精度時だけクエリデータ型へ微分可能な型変換を行う。変換結果はバッチ管理元へ所属させ、長寿命のモデル 管理元へマイクロバッチごとの一時テンソルを残さない。
   */
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

  /** 連結した待ち役マスクを一括展開し、RON/TSUMOで共有する役埋め込みへ変換する。 */
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
   * <p>結果はプレイヤー軸と牌種軸を保った {@code [batch,4,34,context]}。全プレイヤー融合や入力元プレイヤー抽出より前に
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

  /** 同一物理打牌のDAMA表現とRIICHI表現を直接比較し、RIICHI 枠へ二値ロジットを置く。 */
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
   * 方策グラフと同じ条件付き子分布から各分岐候補と分岐の文脈を作り、上位節点を採点する。
   *
   * <p>子分布の重みは勾配を切り離す。したがって上位損失は子ロジットを直接操作せず、選ばれた表現経路と上位スコア計算処理だけを学習する。
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
      NDArray allAlternativeEmbeddings;
      NDArray allPolicyNodeEmbeddings;
      try (NDManager embeddingScope = batchManager.newSubManager()) {
        embeddingScope.tempAttachAll(alternativeIds, policyNodeIds);
        allAlternativeEmbeddings =
            alternativeEmbedding
                .forward(parameterStore, new NDList(alternativeIds), training, runtimeParameters)
                .singletonOrThrow();
        allPolicyNodeEmbeddings =
            policyNodeEmbedding
                .forward(parameterStore, new NDList(policyNodeIds), training, runtimeParameters)
                .singletonOrThrow();
        batchManager.attachAll(allAlternativeEmbeddings, allPolicyNodeEmbeddings);
      }
      NDArray fixedAlternativeContexts =
          allPolicyNodeEmbeddings
              .concat(allAlternativeEmbeddings, 1)
              .reshape(1, DecisionAlternative.NETWORK_SIZE, hiddenSize * 2L);
      alternativeHiddenContexts =
          EpsilonMahjongStateEncoder.silu(
              EpsilonPartitionedLinear.apply(
                  alternativeHidden,
                  parameterStore,
                  training,
                  stateContext,
                  candidateContexts,
                  fixedAlternativeContexts));
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
        scoreGate(
            DecisionPolicyIndexedAffineExecution.Site.RON,
            ronGate,
            policyForward,
            parameterStore,
            stateEmbedding,
            declinedRonContext,
            ronContext,
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
        scoreGate(
            DecisionPolicyIndexedAffineExecution.Site.TSUMO,
            tsumoGate,
            policyForward,
            parameterStore,
            stateEmbedding,
            declinedTsumoContext,
            tsumoContext,
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
    // インデックス指定の経路はホストで構築したactiveRowsだけを使うため、密なマスクは代替処理内で遅延生成する。
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
    float[] bitDivisors = new float[DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK];
    for (int bit = 0; bit < bitDivisors.length; bit++) {
      bitDivisors[bit] = 1 << bit;
    }
    waitYakuBitDivisors = manager.create(bitDivisors);
    alternativeIds = manager.create(ALTERNATIVE_IDS);
    policyNodeIds = manager.create(POLICY_NODE_IDS);
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
    policyNodeEmbedding.initialize(manager, dataType, new Shape(DecisionAlternative.NETWORK_SIZE));
    alternativeEmbedding.initialize(manager, dataType, new Shape(DecisionAlternative.NETWORK_SIZE));
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
    actionCandidateHidden.initialize(manager, dataType, new Shape(-1, hiddenSize));
    actionCandidateScoreHead.initialize(manager, dataType, new Shape(-1, hiddenSize));
    alternativeHidden.initialize(manager, dataType, new Shape(-1, hiddenSize * 4L));
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

  private static long[] policyNodeIds() {
    long[] ids = new long[DecisionAlternative.NETWORK_SIZE];
    for (DecisionAlternative alternative : DecisionAlternative.values()) {
      ids[alternative.networkIndex()] = alternative.scoredNode().networkIndex();
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

  private record CandidateSet(NDArray embeddings, NDArray scores) {}

  private record TransitionInputs(
      NDArray categories,
      NDArray numerics,
      NDArray tileCodes,
      NDArray waitTileIds,
      NDArray waitYakus,
      NDArray waitScores,
      boolean hostCompacted) {

    private static TransitionInputs dense(DecisionNetworkInputs inputs) {
      return new TransitionInputs(
          inputs.transitionCategories(),
          inputs.transitionNumerics(),
          inputs.transitionTiles(),
          inputs.waitTileIds(),
          inputs.waitYakus(),
          inputs.waitScores(),
          false);
    }

    private static TransitionInputs compact(DecisionInferenceInputs inputs) {
      return new TransitionInputs(
          inputs.transitionCategories(),
          inputs.transitionNumerics(),
          inputs.transitionTiles(),
          inputs.waitTileIds(),
          inputs.waitYakus(),
          inputs.waitScores(),
          true);
    }
  }

  private record TransitionSet(NDArray discardScores, NDArray aggregatedEmbeddings) {}
}

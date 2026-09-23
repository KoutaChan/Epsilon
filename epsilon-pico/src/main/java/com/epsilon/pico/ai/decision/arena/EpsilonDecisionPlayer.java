package com.epsilon.pico.ai.decision.arena;

import com.epsilon.ai.decision.DecisionBranchBudget;
import com.epsilon.ai.decision.DecisionBranchComparison;
import com.epsilon.ai.decision.DecisionBranchGate;
import com.epsilon.ai.decision.DecisionBranchOutcome;
import com.epsilon.ai.decision.DecisionBranchSelection;
import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.ai.decision.DecisionSelectionMode;
import com.epsilon.ai.decision.EpsilonDecisionSeeds;
import com.epsilon.ai.grp.EpsilonGrpRankPredictor;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.ai.grp.EpsilonGrpSequence;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionFullSupportSettings;
import com.epsilon.config.settings.DecisionRolloutSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;
import com.epsilon.engine.Player;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import com.epsilon.pico.ai.decision.EpsilonDecisionReturns;
import com.epsilon.pico.ai.decision.EpsilonUtilityTargets;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionCompletedGame;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionGameBoundary;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionLegalActions;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionSampleRecord;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore.PayloadRef;
import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.policy.EpsilonDecisionBehaviorPolicy;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.pico.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.pico.config.settings.DecisionSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;

/** ニューラルネットワークの予測に従い、与えられた合法手の中から行動を選ぶプレイヤー。 */
public class EpsilonDecisionPlayer implements Player {

  private final EpsilonDecisionEvaluator evaluator;
  private final boolean collectTrajectory;
  private final long actorSnapshotId;
  private final DecisionSelectionMode selectionMode;
  private final DecisionFullSupportSettings fullSupport;
  private final float causalTraceLambda;
  private final float explorationCreditMix;
  private final DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession;
  private final SplittableRandom random;
  private final SplittableRandom adaptivePercentileRandom;
  private final EpsilonGrpRankPredictor grpInference;
  private final EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore;
  private final List<PendingDecision> trajectory = new ArrayList<>();
  private final List<BoundarySnapshot> boundarySnapshots = new ArrayList<>();
  private final EpsilonGrpSequence grpSequence = new EpsilonGrpSequence();
  private final int[] seatDecisionOrdinals = new int[GameState.NUM_PLAYERS];
  private BoundarySnapshot currentBoundary;
  private final DecisionBranchComparisonSettings branchSettings;
  private final SplittableRandom branchRandom;
  private BranchCandidate branchCandidate;
  private List<PendingDecision> roundComparisons;
  private boolean comparisonOnly;
  private DecisionBoundaryContext branchContext;

  /**
   * 一つの Decision 行動選択プレイヤーが使う対局生成規則。
   *
   * @param selectionMode 最大確率の行動を選ぶ方式、モデル無作為抽出、FULL_SUPPORT の選択方式
   * @param fullSupport 条件付き節点別の探索に割り当てる確率
   * @param causalTraceLambda 価値計算用と方策更新用の判断列のスカラー値のトレース係数
   * @param explorationCreditMix 探索行動へ残す選択行動学習への寄与の線形混合率
   */
  public record RolloutConfig(
      DecisionSelectionMode selectionMode,
      DecisionFullSupportSettings fullSupport,
      float causalTraceLambda,
      float explorationCreditMix) {

    /**
     * 設定のスカラー値のトレース係数を補って対局生成規則を構築する。
     *
     * @param selectionMode 行動の選択方式
     * @param fullSupport 節点別の探索に割り当てる確率
     */
    public RolloutConfig(
        DecisionSelectionMode selectionMode, DecisionFullSupportSettings fullSupport) {
      this(
          selectionMode,
          fullSupport,
          DecisionSelectedPgCampaignSettings.defaults().causalTraceLambda(),
          DecisionSelectedPgCampaignSettings.defaults().explorationCreditMix());
    }

    /** 選択方式、探索設定、スカラー値のトレース係数の組合せを検証する。 */
    public RolloutConfig {
      if (selectionMode == DecisionSelectionMode.POLICY_SAMPLE && fullSupport.enabled()) {
        throw new IllegalArgumentException("POLICY_SAMPLE requires disabled FULL_SUPPORT");
      }
      if (!Float.isFinite(causalTraceLambda)
          || causalTraceLambda < 0.0f
          || causalTraceLambda > 1.0f) {
        throw new IllegalArgumentException("causalTraceLambda must be in [0, 1]");
      }
      requireExplorationCreditMix(explorationCreditMix);
    }

    public static RolloutConfig fromSettings(SettingsLoader config) {
      var campaign = config.bind(DecisionSelectedPgCampaignSettings.class);
      return new RolloutConfig(
          config.bind(DecisionRolloutSettings.class).selectionMode(),
          config.bind(DecisionFullSupportSettings.class),
          campaign.causalTraceLambda(),
          campaign.explorationCreditMix());
    }

    public static RolloutConfig opponentFromSettings(SettingsLoader config) {
      var campaign = config.bind(DecisionSelectedPgCampaignSettings.class);
      return new RolloutConfig(
          DecisionSelectionMode.POLICY_GREEDY,
          DecisionFullSupportSettings.disabled(),
          campaign.causalTraceLambda(),
          campaign.explorationCreditMix());
    }

    public static RolloutConfig fromSettings() {
      return new RolloutConfig(
          trainingSelectionMode(),
          EpsilonSettings.defaults().bind(DecisionFullSupportSettings.class),
          DecisionSelectedPgCampaignSettings.defaults().causalTraceLambda(),
          DecisionSelectedPgCampaignSettings.defaults().explorationCreditMix());
    }

    public static RolloutConfig opponentFromSettings() {
      return new RolloutConfig(
          DecisionSelectionMode.POLICY_GREEDY,
          DecisionFullSupportSettings.disabled(),
          DecisionSelectedPgCampaignSettings.defaults().causalTraceLambda(),
          DecisionSelectedPgCampaignSettings.defaults().explorationCreditMix());
    }

    public RolloutConfig withCausalTraceLambda(float nextCausalTraceLambda) {
      return new RolloutConfig(
          selectionMode, fullSupport, nextCausalTraceLambda, explorationCreditMix);
    }

    public RolloutConfig withExplorationCreditMix(float nextExplorationCreditMix) {
      return new RolloutConfig(
          selectionMode, fullSupport, causalTraceLambda, nextExplorationCreditMix);
    }

    public String summary() {
      return "selectionMode="
          + selectionMode
          + ",fullSupport="
          + String.format(
              Locale.ROOT,
              "terminalGate=%.3f,callGate=%.3f,meldType=%.3f,meldCandidate=%.3f,"
                  + "kanGate=%.3f,kanType=%.3f,kanCandidate=%.3f,discardIdentity=%.3f,"
                  + "riichiGate=%.3f,minimumLeafProbability=%.6f",
              fullSupport.terminalGateExplorationMass(),
              fullSupport.callGateExplorationMass(),
              fullSupport.meldTypeExplorationMass(),
              fullSupport.meldCandidateExplorationMass(),
              fullSupport.kanGateExplorationMass(),
              fullSupport.kanTypeExplorationMass(),
              fullSupport.kanCandidateExplorationMass(),
              fullSupport.discardIdentityExplorationMass(),
              fullSupport.riichiGateExplorationMass(),
              fullSupport.minimumLeafProbability())
          + ",causalTraceLambda="
          + causalTraceLambda
          + ",explorationCreditMix="
          + explorationCreditMix
          + ",adaptiveExploration="
          + fullSupport.adaptiveExploration();
    }
  }

  /**
   * 指定評価器を使うプレイヤービルダーを作る。
   *
   * @param evaluator 型付きバッチを方策・価値へ評価する評価器
   * @return 行動選択プレイヤー設定を追加するビルダー
   */
  public static Builder builder(EpsilonDecisionEvaluator evaluator) {
    return builder(evaluator, EpsilonSettings.defaults());
  }

  public static Builder builder(EpsilonDecisionEvaluator evaluator, SettingsLoader config) {
    return new Builder(evaluator, config);
  }

  /** 対局中の行動履歴収集、無作為抽出、GRP 入力に使う事前予測依存を明示して行動選択プレイヤーを構築するビルダー。 */
  public static final class Builder {
    private final EpsilonDecisionEvaluator evaluator;
    private final DecisionBranchComparisonSettings branchSettings;
    private boolean collectTrajectory;
    private long actorSnapshotId;
    private DecisionSelectionMode selectionMode;
    private DecisionFullSupportSettings fullSupport;
    private float causalTraceLambda;
    private float explorationCreditMix;
    private final com.epsilon.ai.decision.EpsilonUtilityProfile utilityProfile;
    private long randomSeed;
    private EpsilonGrpRankPredictor grpInference;
    private EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore;
    private DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession;

    private Builder(EpsilonDecisionEvaluator evaluator, SettingsLoader config) {
      this.evaluator = evaluator;
      branchSettings = config.bind(DecisionBranchComparisonSettings.class);
      selectionMode = config.bind(DecisionRolloutSettings.class).selectionMode();
      fullSupport = config.bind(DecisionFullSupportSettings.class);
      causalTraceLambda = config.bind(DecisionSelectedPgCampaignSettings.class).causalTraceLambda();
      explorationCreditMix =
          config.bind(DecisionSelectedPgCampaignSettings.class).explorationCreditMix();
      utilityProfile = config.bind(DecisionSettings.class).utilityProfile();
    }

    /**
     * 対局中の行動履歴に記録する学習サイクル-start 学習側モデルのスナップショット ID を設定する。
     *
     * @param actorSnapshotId チェックポイントマニフェストと照合する正のスナップショット ID
     * @return このビルダー
     */
    public Builder actorSnapshotId(long actorSnapshotId) {
      this.actorSnapshotId = actorSnapshotId;
      return this;
    }

    /**
     * 行動の選択方式を設定する。
     *
     * @param selectionMode 最大確率の行動を選ぶ方式、方策無作為抽出、FULL_SUPPORT のいずれか
     * @return このビルダー
     */
    public Builder selectionMode(DecisionSelectionMode selectionMode) {
      this.selectionMode = selectionMode;
      return this;
    }

    /**
     * 方策グラフの節点別探索に割り当てる確率を設定する。
     *
     * @param fullSupport 実際の行動選択に使う方策に混ぜる探索設定
     * @return このビルダー
     */
    public Builder fullSupport(DecisionFullSupportSettings fullSupport) {
      this.fullSupport = fullSupport;
      return this;
    }

    /**
     * 対局中の行動履歴の価値学習と方策学習で分けた判断列スカラー値のトレース係数を設定する。
     *
     * @param causalTraceLambda 0以上1以下のトレース減衰率
     * @return このビルダー
     */
    public Builder causalTraceLambda(float causalTraceLambda) {
      if (!Float.isFinite(causalTraceLambda)
          || causalTraceLambda < 0.0f
          || causalTraceLambda > 1.0f) {
        throw new IllegalArgumentException("causalTraceLambda must be in [0, 1]");
      }
      this.causalTraceLambda = causalTraceLambda;
      return this;
    }

    /**
     * 探索行動へ残す選択行動学習への寄与の線形混合率を設定する。
     *
     * @param explorationCreditMix 0より大きく1以下の混合率
     * @return このビルダー
     */
    public Builder explorationCreditMix(float explorationCreditMix) {
      requireExplorationCreditMix(explorationCreditMix);
      this.explorationCreditMix = explorationCreditMix;
      return this;
    }

    /**
     * 確率的な行動の無作為抽出の乱数シードを設定する。
     *
     * @param randomSeed 行動選択プレイヤー-局所的な乱数乱数シード
     * @return このビルダー
     */
    public Builder randomSeed(long randomSeed) {
      this.randomSeed = randomSeed;
      return this;
    }

    /**
     * 局境界の公開情報入力に使う事前予測に使う重みを固定したGRP 予測器を設定する。
     *
     * @param grpInference 重みを固定したGRP 予測器。利用しない場合は {@code null}
     * @return このビルダー
     */
    public Builder grpInference(EpsilonGrpRankPredictor grpInference) {
      this.grpInference = grpInference;
      return this;
    }

    /**
     * 選択方式、探索に割り当てる確率、スカラー値のトレース係数、探索による選択の学習への寄与混合率をまとめて設定する。
     *
     * @param rolloutConfig 行動選択プレイヤーが従う対局生成規則
     * @return このビルダー
     */
    public Builder rolloutConfig(RolloutConfig rolloutConfig) {
      return selectionMode(rolloutConfig.selectionMode())
          .fullSupport(rolloutConfig.fullSupport())
          .causalTraceLambda(rolloutConfig.causalTraceLambda())
          .explorationCreditMix(rolloutConfig.explorationCreditMix());
    }

    /**
     * 学習サイクル境界で固定された不確実性CDFと収集診断用の集計を設定する。
     *
     * <p>実行時状態なので対局生成設定や学習器互換性識別子には含めない。
     */
    Builder adaptiveExplorationSession(
        DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession) {
      this.adaptiveExplorationSession = adaptiveExplorationSession;
      return this;
    }

    /**
     * 現在の設定から Decision プレイヤーを構築する。
     *
     * @return 合法手一覧の範囲内だけを選ぶプレイヤー
     */
    public EpsilonDecisionPlayer build() {
      return new EpsilonDecisionPlayer(this);
    }
  }

  private final com.epsilon.ai.decision.EpsilonUtilityProfile utilityProfile;

  private EpsilonDecisionPlayer(Builder builder) {
    utilityProfile = builder.utilityProfile;
    this.evaluator = builder.evaluator;
    this.collectTrajectory = builder.collectTrajectory;
    branchSettings =
        collectTrajectory && builder.branchSettings.enabled() ? builder.branchSettings : null;
    branchRandom =
        branchSettings == null
            ? null
            : new SplittableRandom(EpsilonDecisionSeeds.branch(builder.randomSeed, 0));
    this.actorSnapshotId = builder.actorSnapshotId;
    this.selectionMode = builder.selectionMode;
    this.fullSupport = builder.fullSupport;
    this.causalTraceLambda = builder.causalTraceLambda;
    this.explorationCreditMix = builder.explorationCreditMix;
    this.adaptiveExplorationSession = builder.adaptiveExplorationSession;
    this.random = new SplittableRandom(builder.randomSeed);
    this.adaptivePercentileRandom =
        adaptiveExplorationSession == null
            ? null
            : new SplittableRandom(EpsilonDecisionSeeds.adaptivePercentile(builder.randomSeed));
    this.grpInference = builder.grpInference;
    this.trajectoryPayloadStore = builder.trajectoryPayloadStore;
  }

  static EpsilonDecisionPlayer trainingRollout(
      EpsilonDecisionEvaluator evaluator,
      long actorSnapshotId,
      long randomSeed,
      RolloutConfig rolloutConfig,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore) {
    return trainingRollout(
        evaluator, actorSnapshotId, randomSeed, rolloutConfig, trajectoryPayloadStore, null);
  }

  static EpsilonDecisionPlayer trainingRollout(
      EpsilonDecisionEvaluator evaluator,
      long actorSnapshotId,
      long randomSeed,
      RolloutConfig rolloutConfig,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore,
      EpsilonGrpRankPredictor grpInference) {
    return trainingRollout(
        evaluator,
        actorSnapshotId,
        randomSeed,
        rolloutConfig,
        trajectoryPayloadStore,
        grpInference,
        null);
  }

  static EpsilonDecisionPlayer trainingRollout(
      EpsilonDecisionEvaluator evaluator,
      long actorSnapshotId,
      long randomSeed,
      RolloutConfig rolloutConfig,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore,
      EpsilonGrpRankPredictor grpInference,
      DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession) {
    return trainingRollout(
        evaluator,
        actorSnapshotId,
        randomSeed,
        rolloutConfig,
        trajectoryPayloadStore,
        grpInference,
        adaptiveExplorationSession,
        EpsilonSettings.defaults());
  }

  static EpsilonDecisionPlayer trainingRollout(
      EpsilonDecisionEvaluator evaluator,
      long actorSnapshotId,
      long randomSeed,
      RolloutConfig rolloutConfig,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore,
      EpsilonGrpRankPredictor grpInference,
      DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession,
      SettingsLoader config) {
    Builder builder =
        builder(evaluator, config)
            .actorSnapshotId(actorSnapshotId)
            .rolloutConfig(rolloutConfig)
            .randomSeed(randomSeed)
            .grpInference(grpInference)
            .adaptiveExplorationSession(adaptiveExplorationSession);
    builder.collectTrajectory = true;
    builder.trajectoryPayloadStore = trajectoryPayloadStore;
    return builder.build();
  }

  static EpsilonDecisionPlayer rolloutActionOnly(
      EpsilonDecisionEvaluator evaluator,
      long actorSnapshotId,
      long randomSeed,
      RolloutConfig rolloutConfig) {
    return builder(evaluator)
        .actorSnapshotId(actorSnapshotId)
        .rolloutConfig(rolloutConfig)
        .randomSeed(randomSeed)
        .build();
  }

  @Override
  public Action selectAction(GameState state, int playerIndex, List<Action> legalActions) {
    if (legalActions.isEmpty()) {
      throw new IllegalArgumentException("legalActions must not be empty");
    }
    if (legalActions.size() == 1 && !collectTrajectory) {
      return legalActions.getFirst();
    }
    DecisionBucket bucket = DecisionBatchBuilder.selectInferenceBucket(legalActions);
    DecisionBatchBuilder builder = DecisionBatchBuilder.inference(1, bucket);
    DecisionBoundaryContext boundaryContext = prepareBoundaryContextAsync(state).join();
    builder.addDetachedInferenceRow(
        state, playerIndex, legalActions, state.publicState(), boundaryContext);
    DecisionHostBatch batch = builder.build();
    EpsilonDecisionInferenceServer.Prediction prediction =
        evaluator.evaluateBatch(batch).getFirst();
    DecisionHostBatch.RowSlice input = collectTrajectory ? batch.sliceRows(0, 1) : null;
    return selectResolvedAction(0L, legalActions, input, prediction);
  }

  Action selectActionFromPrediction(
      long engineDecisionId,
      List<Action> legalActions,
      DecisionHostBatch.RowSlice input,
      EpsilonDecisionInferenceServer.Prediction prediction) {
    if (engineDecisionId <= 0L) {
      throw new IllegalArgumentException("engineDecisionId must be positive");
    }
    return selectResolvedAction(engineDecisionId, legalActions, input, prediction);
  }

  private Action selectResolvedAction(
      long engineDecisionId,
      List<Action> legalActions,
      DecisionHostBatch.RowSlice input,
      EpsilonDecisionInferenceServer.Prediction prediction) {
    if (selectionMode == DecisionSelectionMode.POLICY_GREEDY && !collectTrajectory) {
      return legalActions.get(prediction.greedyActionSlot());
    }
    float[] modelRolloutPolicy = rolloutPolicy(prediction);
    DecisionAdaptiveExploration.Observation explorationObservation =
        selectionMode == DecisionSelectionMode.FULL_SUPPORT && adaptiveExplorationSession != null
            ? adaptiveExplorationSession.decide(
                legalActions,
                modelRolloutPolicy,
                adaptivePercentileRandom.nextDouble(),
                !comparisonOnly)
            : null;
    float explorationMultiplier =
        explorationObservation == null ? 1.0f : explorationObservation.multiplier();
    EpsilonDecisionBehaviorPolicy.Mixture mixture =
        switch (selectionMode) {
          case POLICY_GREEDY -> null;
          case FULL_SUPPORT ->
              EpsilonDecisionBehaviorPolicy.distributionInPlace(
                  modelRolloutPolicy, legalActions, fullSupport, explorationMultiplier);
          case POLICY_SAMPLE -> EpsilonDecisionBehaviorPolicy.directInPlace(modelRolloutPolicy);
        };
    float[] rolloutPolicy = mixture == null ? modelRolloutPolicy : mixture.rolloutPolicy();
    float[] actionProbabilities =
        switch (selectionMode) {
          case POLICY_GREEDY -> oneHot(prediction.greedyActionSlot(), legalActions.size());
          case FULL_SUPPORT, POLICY_SAMPLE -> mixture.behaviorPolicy();
        };
    int selectedSlot = selectSlot(prediction, actionProbabilities);
    if (explorationObservation != null && !comparisonOnly) {
      adaptiveExplorationSession.recordSelection(
          explorationObservation, rolloutPolicy, actionProbabilities, selectedSlot);
    }
    if (collectTrajectory) {
      if (input == null || input.size() != 1) {
        throw new IllegalArgumentException("one typed input row is required for trajectory");
      }
      DecisionHostBatch source = input.source();
      int inputRow = input.fromInclusive();
      float logProb = (float) Math.log(actionProbabilities[selectedSlot]);
      BoundarySnapshot boundary = requireCurrentBoundary();
      float[] grpFeatureSequence = boundary.grpFeatureSequence();
      float rolloutValue = prediction.valueUtility();
      int legalCount = source.legalActionCount(inputRow);
      int[] legalActionIdBySlot = new int[legalCount];
      for (int slot = 0; slot < legalCount; slot++) {
        legalActionIdBySlot[slot] = source.legalActionId(inputRow, slot);
      }
      PayloadRef payloadRef = writeTrajectoryPayload(input, actionProbabilities, rolloutPolicy);
      trajectory.add(
          new PendingDecision(
              engineDecisionId,
              payloadRef,
              legalActionIdBySlot,
              legalCount,
              source.playerSeat(inputRow),
              source.sourcePlayerRelativeSeat(inputRow),
              source.currentPlayerRelativeSeat(inputRow),
              selectedSlot,
              legalActions.get(selectedSlot).toIndex(),
              logProb,
              actionProbabilities[selectedSlot],
              rolloutPolicy[selectedSlot],
              rolloutValue,
              grpFeatureSequence,
              boundary.index(),
              seatDecisionOrdinals[source.playerSeat(inputRow)]++,
              explorationObservation == null ? 0 : explorationObservation.roleCode()));
      if (engineDecisionId == 0L && explorationObservation != null) {
        adaptiveExplorationSession.recordLearningRole(
            explorationObservation,
            legalCount == 1 ? DecisionLearningRole.FORCED : DecisionLearningRole.CAUSAL);
      }
    }
    if (collectTrajectory && branchSettings != null) {
      prepareBranchCandidate(
          engineDecisionId,
          legalActions,
          rolloutPolicy,
          actionProbabilities,
          selectedSlot,
          explorationMultiplier);
    }
    return legalActions.get(selectedSlot);
  }

  void recordDecisionOutcome(
      long engineDecisionId, Action selectedAction, DecisionLearningRole learningRole) {
    if (!collectTrajectory) {
      return;
    }
    for (int i = trajectory.size() - 1; i >= 0; i--) {
      PendingDecision decision = trajectory.get(i);
      if (decision.engineDecisionId() != engineDecisionId) {
        continue;
      }
      if (decision.chosenActionId() != selectedAction.toIndex()) {
        throw new IllegalStateException(
            "Engine outcome action does not match collected trajectory decision "
                + engineDecisionId);
      }
      decision.resolveLearningRole(learningRole);
      if (adaptiveExplorationSession != null && decision.explorationRoleCode() != 0) {
        adaptiveExplorationSession.recordLearningRole(decision.explorationRoleCode(), learningRole);
      }
      return;
    }
    throw new IllegalStateException(
        "No collected trajectory decision for engine decision " + engineDecisionId);
  }

  record BranchCandidate(
      long decisionId,
      Action alternative,
      DecisionBranchGate gate,
      float oldProbability,
      boolean mainAccepted,
      int trajectoryIndex,
      int playerSeat,
      float[] grpPrefix) {}

  BranchCandidate takeBranchCandidate() {
    BranchCandidate candidate = branchCandidate;
    branchCandidate = null;
    return candidate;
  }

  DecisionBranchComparison admitBranch(BranchCandidate candidate, DecisionBranchBudget budget) {
    PendingDecision decision = trajectory.get(candidate.trajectoryIndex());
    DecisionBranchComparison comparison =
        new DecisionBranchComparison(
            candidate.gate(), candidate.oldProbability(), candidate.mainAccepted(), budget);
    decision.comparison = comparison;
    if (roundComparisons == null) {
      roundComparisons = new ArrayList<>();
    }
    roundComparisons.add(decision);
    return comparison;
  }

  CompletableFuture<Float> branchUtility(RoundSettlement settlement, BranchCandidate candidate) {
    return DecisionBranchOutcome.utility(
        settlement, candidate.grpPrefix(), candidate.playerSeat(), utilityProfile, grpInference);
  }

  EpsilonDecisionPlayer branchPlayer(long seed, SettingsLoader config) {
    Builder builder =
        builder(evaluator, config)
            .actorSnapshotId(actorSnapshotId)
            .randomSeed(seed)
            .rolloutConfig(
                new RolloutConfig(
                    selectionMode, fullSupport, causalTraceLambda, explorationCreditMix));
    builder.adaptiveExplorationSession = adaptiveExplorationSession;
    EpsilonDecisionPlayer player = builder.build();
    player.comparisonOnly = true;
    player.branchContext =
        currentBoundary == null
            ? DecisionBoundaryContext.uniform()
            : currentBoundary.contextFuture().join();
    return player;
  }

  private void prepareBranchCandidate(
      long decisionId,
      List<Action> actions,
      float[] rollout,
      float[] behavior,
      int selected,
      float explorationMultiplier) {
    branchCandidate = null;
    if (branchSettings == null) {
      return;
    }
    PendingDecision decision = trajectory.getLast();
    for (DecisionBranchGate gate : DecisionBranchGate.PRIORITY) {
      boolean kyushu = gate == DecisionBranchGate.KYUSHU;
      if (kyushu ? !branchSettings.kyushu() : !branchSettings.winDecline()) {
        continue;
      }
      int acceptedSlot = -1;
      boolean hasDecline = false;
      for (int slot = 0; slot < actions.size(); slot++) {
        if (gate.accepts(actions.get(slot))) {
          acceptedSlot = slot;
        } else if (gate.contains(actions.get(slot))) {
          hasDecline = true;
        }
      }
      if (acceptedSlot < 0 || !hasDecline) {
        continue;
      }
      if (kyushu) {
        decision.branchTarget = DecisionBranchTarget.KYUSHU_ONLY;
      }
      if (!gate.contains(actions.get(selected))) {
        return;
      }
      float probability = DecisionBranchSelection.acceptanceProbability(gate, actions, rollout);
      // 丸めで0/1になったgateは比較比率を定義できない。KYUSHUの通常勾配停止は維持する。
      if (!(probability > 0 && probability < 1)) {
        return;
      }
      boolean accepted = gate.accepts(actions.get(selected));
      if (!kyushu) {
        if (accepted || selectionMode != DecisionSelectionMode.FULL_SUPPORT) {
          continue;
        }
        float mass = fullSupport.terminalGateExplorationMass() * explorationMultiplier;
        double posterior =
            DecisionBranchSelection.explorationPosterior(
                probability, behavior[selected], mass, fullSupport.minimumLeafProbability());
        if (branchRandom.nextDouble() >= posterior) {
          continue;
        }
      }
      int alternateSlot =
          accepted
              ? DecisionBranchSelection.sampleDecline(
                  gate, actions, behavior, branchRandom.nextDouble())
              : acceptedSlot;
      branchCandidate =
          new BranchCandidate(
              decisionId,
              actions.get(alternateSlot),
              gate,
              probability,
              accepted,
              trajectory.size() - 1,
              decision.playerSeat(),
              decision.grpFeatureSequence());
      return;
    }
  }

  private DecisionBranchTarget branchTargetFor(PendingDecision decision) {
    if (decision.comparison == null) {
      return decision.branchTarget;
    }
    DecisionBranchTarget target =
        decision.learningRole().advancesActorClock()
            ? decision.comparison.freeze()
            : decision.branchTarget;
    decision.comparison.cancel();
    return target;
  }

  EpsilonDecisionEvaluator evaluator() {
    return evaluator;
  }

  boolean collectsTrajectory() {
    return collectTrajectory;
  }

  CompletableFuture<DecisionBoundaryContext> prepareBoundaryContextAsync(GameState state) {
    if (!collectTrajectory) {
      return CompletableFuture.completedFuture(
          branchContext == null ? DecisionBoundaryContext.uniform() : branchContext);
    }
    return ensureBoundarySnapshot(grpSequence.include(state)).contextFuture();
  }

  @Override
  public void onRoundSettled(RoundSettlement settlement) {
    if (!collectTrajectory) {
      return;
    }
    RoundTransition transition = settlement.transition();
    switch (transition) {
      case RoundTransition.NextRound next -> {
        grpSequence.include(
            next.kyokuIndex(), next.honba(), next.kyotakuCount(), settlement.snapshotFinalScores());
        ensureBoundarySnapshot(grpSequence.current());
      }
      case RoundTransition.HanchanFinished ignored -> {
        // 最終局は次のGRP境界を追加せず、出力確定時に終局効用で局内トレースを閉じる。
      }
    }
    if (roundComparisons != null && !roundComparisons.isEmpty()) {
      for (PendingDecision decision : roundComparisons) {
        CompletableFuture<Float> utility =
            transition instanceof RoundTransition.NextRound
                ? currentBoundary
                    .grpRankProbabilities()
                    .thenApply(
                        marginals ->
                            DecisionBranchOutcome.expectedUtility(
                                marginals, decision.playerSeat(), utilityProfile))
                : DecisionBranchOutcome.utility(
                    settlement,
                    decision.grpFeatureSequence(),
                    decision.playerSeat(),
                    utilityProfile,
                    grpInference);
        decision.comparison.completeMain(utility);
      }
      roundComparisons.clear();
    }
  }

  /**
   * 半荘終了時の対局中の行動履歴を、入力と確率分布の本体を読み戻さずに学習サンプルビューへ変換する。
   *
   * <p>型付き入力 / behaviorPolicy / rolloutPolicy は処理中の一時保存領域から読まず、選択時に保存した合法手メタデータ
   * と軽量統計を使って遅延サンプルを作る。対局単位の学習データ書き込み処理は入力と確率分布の本体ファイルテーブルとオフセット/長さ参照だけを保存する。
   */
  EpsilonDecisionCompletedGame flushTrajectoryDeferred(long gameId, int[] finalScores) {
    int[] ranks = ScoreRanking.byScoreThenSeat(finalScores);
    int finalRanksCode = EpsilonGrpRanks.encode(ranks);
    if (trajectory.isEmpty()) {
      resetTrajectory();
      return new EpsilonDecisionCompletedGame(gameId, finalRanksCode, List.of(), List.of());
    }
    requireDecisionOutcomesResolved();
    int lastObservedBoundary = -1;
    for (PendingDecision decision : trajectory) {
      lastObservedBoundary = Math.max(lastObservedBoundary, decision.boundaryIndex());
    }
    ArrayList<EpsilonDecisionGameBoundary> boundaries = new ArrayList<>(lastObservedBoundary + 1);
    for (int boundaryIndex = 0; boundaryIndex <= lastObservedBoundary; boundaryIndex++) {
      BoundarySnapshot boundary = boundarySnapshots.get(boundaryIndex);
      boundaries.add(
          new EpsilonDecisionGameBoundary(
              boundary.index(),
              boundary.grpFeatureSequence(),
              boundary.grpRankProbabilities().join()));
    }
    ArrayList<EpsilonDecisionSampleRecord> samples = new ArrayList<>(trajectory.size());
    int trainingProfileIndex = utilityProfile.ordinal();
    TrainingTarget[] targets = buildTrainingTargets(finalScores);
    for (int i = 0; i < trajectory.size(); i++) {
      PendingDecision decision = trajectory.get(i);
      TrainingTarget target = targets[i];
      samples.add(
          trajectoryPayloadStore.deferredSample(
              decision.payloadRef(),
              decision.legalActionIdBySlot(),
              decision.legalActionCount(),
              decision.playerSeat(),
              decision.sourcePlayerRelativeSeat(),
              decision.currentPlayerRelativeSeat(),
              decision.chosenLegalSlot(),
              decision.chosenActionId(),
              decision.behaviorLogProb(),
              decision.behaviorProb(),
              target.valueTarget(),
              target.advantage(),
              target.finalRank(),
              actorSnapshotId,
              trainingProfileIndex,
              gameId,
              decision.boundaryIndex(),
              decision.seatDecisionOrdinal(),
              decision.grpFeatureSequence(),
              finalRanksCode,
              decision.learningRole(),
              branchTargetFor(decision)));
    }
    EpsilonDecisionCompletedGame completed =
        new EpsilonDecisionCompletedGame(gameId, finalRanksCode, boundaries, samples);
    resetTrajectory();
    return completed;
  }

  private void resetTrajectory() {
    trajectory.clear();
    branchCandidate = null;
    if (roundComparisons != null) {
      roundComparisons.clear();
    }
    boundarySnapshots.clear();
    currentBoundary = null;
    Arrays.fill(seatDecisionOrdinals, 0);
    grpSequence.clear();
  }

  private TrainingTarget[] buildTrainingTargets(int[] finalScores) {
    int[] ranks = ScoreRanking.byScoreThenSeat(finalScores);
    TrainingTarget[] targets = new TrainingTarget[trajectory.size()];
    int[] activeBoundaryBySeat = new int[GameState.NUM_PLAYERS];
    float[] nextValuePredictionBySeat = new float[GameState.NUM_PLAYERS];
    boolean[] hasNextValueBySeat = new boolean[GameState.NUM_PLAYERS];
    float[] nextValueTargetBySeat = new float[GameState.NUM_PLAYERS];
    float[] nextActorPredictionBySeat = new float[GameState.NUM_PLAYERS];
    boolean[] hasNextActorBySeat = new boolean[GameState.NUM_PLAYERS];
    float[] nextActorTargetBySeat = new float[GameState.NUM_PLAYERS];
    Arrays.fill(activeBoundaryBySeat, -1);
    for (int i = trajectory.size() - 1; i >= 0; i--) {
      PendingDecision decision = trajectory.get(i);
      int seat = decision.playerSeat();
      int finalRank = ranks[seat];
      if (activeBoundaryBySeat[seat] != decision.boundaryIndex()) {
        activeBoundaryBySeat[seat] = decision.boundaryIndex();
        float boundaryTarget = nextBoundaryValueTarget(decision.boundaryIndex(), seat, finalRank);
        hasNextValueBySeat[seat] = false;
        nextValueTargetBySeat[seat] = boundaryTarget;
        hasNextActorBySeat[seat] = false;
        nextActorTargetBySeat[seat] = boundaryTarget;
      }
      float currentValue = decision.rolloutValue();
      float coefficient = decision.traceCoefficient(explorationCreditMix);
      float nextValue =
          hasNextValueBySeat[seat]
              ? nextValuePredictionBySeat[seat]
              : nextValueTargetBySeat[seat];
      float valueTarget =
          EpsilonDecisionReturns.scalarVTraceTarget(
              currentValue,
              nextValue,
              nextValueTargetBySeat[seat],
              causalTraceLambda,
              coefficient);
      float advantage = 0.0f;
      if (decision.learningRole().advancesActorClock()) {
        float nextActorValue =
            hasNextActorBySeat[seat]
                ? nextActorPredictionBySeat[seat]
                : nextActorTargetBySeat[seat];
        float actorTarget =
            EpsilonDecisionReturns.actorLookaheadTarget(
                nextActorValue, nextActorTargetBySeat[seat], causalTraceLambda);
        advantage = actorTarget - currentValue;
        nextActorPredictionBySeat[seat] = currentValue;
        hasNextActorBySeat[seat] = true;
        nextActorTargetBySeat[seat] =
            EpsilonDecisionReturns.scalarVTraceTarget(
                currentValue,
                nextActorValue,
                nextActorTargetBySeat[seat],
                causalTraceLambda,
                coefficient);
      }
      targets[i] = new TrainingTarget(valueTarget, advantage, finalRank);
      nextValuePredictionBySeat[seat] = currentValue;
      hasNextValueBySeat[seat] = true;
      nextValueTargetBySeat[seat] = valueTarget;
    }
    return targets;
  }

  /**
   * 指定局の価値トレースを終端する期待効用を返す。
   *
   * <p>重みを固定したGRPが利用可能で即時次境界が存在する場合はその席周辺分布の期待効用を使い、それ以外は半荘終局効用へ戻す。
   */
  private float nextBoundaryValueTarget(int boundaryIndex, int seat, int finalRank) {
    int nextBoundaryIndex = boundaryIndex + 1;
    if (grpInference != null && nextBoundaryIndex < boundarySnapshots.size()) {
      float[] marginals =
          requireBoundaryMarginals(
              boundarySnapshots.get(nextBoundaryIndex).grpRankProbabilities().join());
      return EpsilonUtilityTargets.expectedRankUtilityTrusted(
          utilityProfile, EpsilonGrpRanks.seatMarginal(marginals, seat));
    }
    return utilityProfile.utilityForRank(finalRank);
  }

  private BoundarySnapshot ensureBoundarySnapshot(float[] grpFeatureSequence) {
    int steps = com.epsilon.ai.grp.EpsilonGrpFeature.steps(grpFeatureSequence);
    if (steps <= 0) {
      throw new IllegalStateException("Decision boundary requires a non-empty GRP prefix");
    }
    if (currentBoundary != null) {
      int currentSteps =
          com.epsilon.ai.grp.EpsilonGrpFeature.steps(currentBoundary.grpFeatureSequence());
      if (currentSteps == steps) {
        return currentBoundary;
      }
    }
    float[] sequence = grpFeatureSequence;
    CompletableFuture<float[]> grpMarginals =
        grpInference == null
            ? CompletableFuture.completedFuture(uniformBoundaryMarginals())
            : grpInference
                .predictMarginalProbabilitiesAsync(sequence)
                .thenApply(EpsilonDecisionPlayer::requireBoundaryMarginals);
    BoundarySnapshot boundary =
        new BoundarySnapshot(
            boundarySnapshots.size(),
            sequence,
            grpMarginals,
            grpMarginals.thenApply(
                probabilities -> DecisionBoundaryContext.fromGrpBoundary(sequence, probabilities)));
    boundarySnapshots.add(boundary);
    currentBoundary = boundary;
    return boundary;
  }

  private BoundarySnapshot requireCurrentBoundary() {
    if (currentBoundary == null) {
      throw new IllegalStateException("Decision boundary context was not prepared");
    }
    return currentBoundary;
  }

  private static float[] uniformBoundaryMarginals() {
    float[] marginals = new float[EpsilonGrpRanks.MATRIX_SIZE];
    Arrays.fill(marginals, 1.0f / EpsilonGrpRanks.RANK_COUNT);
    return marginals;
  }

  private static float[] requireBoundaryMarginals(float[] probabilities) {
    if (probabilities == null || probabilities.length != EpsilonGrpRanks.MATRIX_SIZE) {
      throw new IllegalStateException(
          "GRP boundary prior must contain " + EpsilonGrpRanks.MATRIX_SIZE + " probabilities");
    }
    for (int seat = 0; seat < EpsilonGrpRanks.SEAT_COUNT; seat++) {
      double sum = 0.0;
      for (int rank = 0; rank < EpsilonGrpRanks.RANK_COUNT; rank++) {
        float probability = probabilities[EpsilonGrpRanks.index(seat, rank)];
        if (!Float.isFinite(probability) || probability <= 0.0f) {
          throw new IllegalStateException("GRP boundary prior must be finite and positive");
        }
        sum += probability;
      }
      if (Math.abs(sum - 1.0) > 1.0e-4) {
        throw new IllegalStateException(
            "GRP boundary prior row is not normalized: seat=" + seat + " sum=" + sum);
      }
    }
    return probabilities;
  }

  private void requireDecisionOutcomesResolved() {
    long unresolved =
        trajectory.stream().filter(decision -> !decision.decisionOutcomeResolved()).count();
    if (unresolved > 0L) {
      throw new IllegalStateException(
          "Trajectory has unresolved engine decision outcomes: decisions=" + unresolved);
    }
  }

  private int sample(float[] probabilities) {
    double threshold = random.nextDouble();
    double cumulative = 0.0;
    for (int i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (threshold <= cumulative) {
        return i;
      }
    }
    return probabilities.length - 1;
  }

  private int selectSlot(
      EpsilonDecisionInferenceServer.Prediction prediction, float[] probabilities) {
    return switch (selectionMode) {
      case POLICY_GREEDY -> prediction.greedyActionSlot();
      case FULL_SUPPORT, POLICY_SAMPLE -> sample(probabilities);
    };
  }

  private static float[] rolloutPolicy(EpsilonDecisionInferenceServer.Prediction prediction) {
    return prediction.policyProbabilities();
  }

  static DecisionSelectionMode defaultSelectionMode() {
    return DecisionSelectionMode.POLICY_GREEDY;
  }

  private static DecisionSelectionMode trainingSelectionMode() {
    return EpsilonSettings.defaults().bind(DecisionRolloutSettings.class).selectionMode();
  }

  private static void requireExplorationCreditMix(float explorationCreditMix) {
    if (!Float.isFinite(explorationCreditMix)
        || explorationCreditMix <= 0.0f
        || explorationCreditMix > 1.0f) {
      throw new IllegalArgumentException("explorationCreditMix must be in (0, 1]");
    }
  }

  static float traceCoefficient(
      DecisionLearningRole learningRole,
      float rolloutProbability,
      float behaviorProbability,
      float explorationCreditMix) {
    if (learningRole != DecisionLearningRole.CAUSAL) {
      return 1.0f;
    }
    return EpsilonDecisionReturns.selectedTraceCoefficient(
        rolloutProbability, behaviorProbability, explorationCreditMix);
  }

  private static float[] oneHot(int index, int size) {
    float[] out = new float[size];
    out[index] = 1.0f;
    return out;
  }

  private PayloadRef writeTrajectoryPayload(
      DecisionHostBatch.RowSlice input, float[] behaviorPolicy, float[] rolloutPolicy) {
    try {
      return trajectoryPayloadStore.write(input, behaviorPolicy, rolloutPolicy);
    } catch (IOException e) {
      throw new IllegalStateException("Failed to write Decision in-flight trajectory payload", e);
    }
  }

  private static final class PendingDecision {
    private final long engineDecisionId;
    private final PayloadRef payloadRef;
    private final int[] legalActionIdBySlot;
    private final int legalActionCount;
    private final int playerSeat;
    private final int sourcePlayerRelativeSeat;
    private final int currentPlayerRelativeSeat;
    private final int chosenLegalSlot;
    private final int chosenActionId;
    private final float behaviorLogProb;
    private final float behaviorProb;
    private final float rolloutProb;
    private final float rolloutValue;
    private final float[] grpFeatureSequence;
    private final int boundaryIndex;
    private final int seatDecisionOrdinal;
    private final int explorationRoleCode;
    private boolean decisionOutcomeResolved;
    private DecisionLearningRole learningRole;
    private DecisionBranchTarget branchTarget = DecisionBranchTarget.NONE;
    private DecisionBranchComparison comparison;

    private PendingDecision(
        long engineDecisionId,
        PayloadRef payloadRef,
        int[] legalActionIdBySlot,
        int legalActionCount,
        int playerSeat,
        int sourcePlayerRelativeSeat,
        int currentPlayerRelativeSeat,
        int chosenLegalSlot,
        int chosenActionId,
        float behaviorLogProb,
        float behaviorProb,
        float rolloutProb,
        float rolloutValue,
        float[] grpFeatureSequence,
        int boundaryIndex,
        int seatDecisionOrdinal,
        int explorationRoleCode) {
      this(
          engineDecisionId,
          payloadRef,
          legalActionIdBySlot,
          legalActionCount,
          playerSeat,
          sourcePlayerRelativeSeat,
          currentPlayerRelativeSeat,
          chosenLegalSlot,
          chosenActionId,
          behaviorLogProb,
          behaviorProb,
          rolloutProb,
          rolloutValue,
          grpFeatureSequence,
          boundaryIndex,
          seatDecisionOrdinal,
          explorationRoleCode,
          engineDecisionId == 0L,
          legalActionCount == 1 ? DecisionLearningRole.FORCED : DecisionLearningRole.CAUSAL);
    }

    private PendingDecision(
        long engineDecisionId,
        PayloadRef payloadRef,
        int[] legalActionIdBySlot,
        int legalActionCount,
        int playerSeat,
        int sourcePlayerRelativeSeat,
        int currentPlayerRelativeSeat,
        int chosenLegalSlot,
        int chosenActionId,
        float behaviorLogProb,
        float behaviorProb,
        float rolloutProb,
        float rolloutValue,
        float[] grpFeatureSequence,
        int boundaryIndex,
        int seatDecisionOrdinal,
        int explorationRoleCode,
        boolean decisionOutcomeResolved,
        DecisionLearningRole learningRole) {
      if (engineDecisionId < 0L) {
        throw new IllegalArgumentException("engineDecisionId must be non-negative");
      }
      this.engineDecisionId = engineDecisionId;
      this.payloadRef = payloadRef;
      this.legalActionIdBySlot =
          EpsilonDecisionLegalActions.compact(legalActionIdBySlot, legalActionCount);
      this.legalActionCount = EpsilonDecisionLegalActions.compactCount(legalActionCount);
      this.playerSeat = playerSeat;
      this.sourcePlayerRelativeSeat = sourcePlayerRelativeSeat;
      this.currentPlayerRelativeSeat = currentPlayerRelativeSeat;
      this.chosenLegalSlot = chosenLegalSlot;
      this.chosenActionId = chosenActionId;
      this.behaviorLogProb = behaviorLogProb;
      this.behaviorProb = behaviorProb;
      if (!Float.isFinite(behaviorProb) || behaviorProb <= 0.0f) {
        throw new IllegalArgumentException("behaviorProb must be finite and positive");
      }
      if (!Float.isFinite(rolloutProb) || rolloutProb < 0.0f) {
        throw new IllegalArgumentException("rolloutProb must be finite and non-negative");
      }
      this.rolloutProb = rolloutProb;
      this.rolloutValue = rolloutValue;
      this.grpFeatureSequence = grpFeatureSequence;
      this.boundaryIndex = boundaryIndex;
      this.seatDecisionOrdinal = seatDecisionOrdinal;
      this.explorationRoleCode = explorationRoleCode;
      this.decisionOutcomeResolved = decisionOutcomeResolved;
      this.learningRole = learningRole;
    }

    private void resolveLearningRole(DecisionLearningRole nextLearningRole) {
      if (engineDecisionId == 0L || decisionOutcomeResolved) {
        throw new IllegalStateException(
            "Engine decision outcome already resolved: " + engineDecisionId);
      }
      decisionOutcomeResolved = true;
      learningRole = nextLearningRole;
    }

    private long engineDecisionId() {
      return engineDecisionId;
    }

    private PayloadRef payloadRef() {
      return payloadRef;
    }

    private int[] legalActionIdBySlot() {
      return legalActionIdBySlot;
    }

    private int legalActionCount() {
      return legalActionCount;
    }

    private int playerSeat() {
      return playerSeat;
    }

    private int sourcePlayerRelativeSeat() {
      return sourcePlayerRelativeSeat;
    }

    private int currentPlayerRelativeSeat() {
      return currentPlayerRelativeSeat;
    }

    private int chosenLegalSlot() {
      return chosenLegalSlot;
    }

    private int chosenActionId() {
      return chosenActionId;
    }

    private float behaviorLogProb() {
      return behaviorLogProb;
    }

    private float behaviorProb() {
      return behaviorProb;
    }

    private float traceCoefficient(float explorationCreditMix) {
      return EpsilonDecisionPlayer.traceCoefficient(
          learningRole, rolloutProb, behaviorProb, explorationCreditMix);
    }

    private float rolloutValue() {
      return rolloutValue;
    }

    private float[] grpFeatureSequence() {
      return grpFeatureSequence;
    }

    private int boundaryIndex() {
      return boundaryIndex;
    }

    private int seatDecisionOrdinal() {
      return seatDecisionOrdinal;
    }

    private int explorationRoleCode() {
      return explorationRoleCode;
    }

    private boolean decisionOutcomeResolved() {
      return decisionOutcomeResolved;
    }

    private DecisionLearningRole learningRole() {
      return learningRole;
    }
  }

  private static final class BoundarySnapshot {
    private final int index;
    private final float[] grpFeatureSequence;
    private final CompletableFuture<float[]> grpRankProbabilities;
    private final CompletableFuture<DecisionBoundaryContext> contextFuture;

    private BoundarySnapshot(
        int index,
        float[] grpFeatureSequence,
        CompletableFuture<float[]> grpRankProbabilities,
        CompletableFuture<DecisionBoundaryContext> contextFuture) {
      this.index = index;
      this.grpFeatureSequence = grpFeatureSequence;
      this.grpRankProbabilities = grpRankProbabilities;
      this.contextFuture = contextFuture;
    }

    private int index() {
      return index;
    }

    private float[] grpFeatureSequence() {
      return grpFeatureSequence;
    }

    private CompletableFuture<DecisionBoundaryContext> contextFuture() {
      return contextFuture;
    }

    private CompletableFuture<float[]> grpRankProbabilities() {
      return grpRankProbabilities;
    }
  }

  private record TrainingTarget(float valueTarget, float advantage, int finalRank) {}
}

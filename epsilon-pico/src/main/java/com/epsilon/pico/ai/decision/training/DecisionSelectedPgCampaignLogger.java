package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.arena.DecisionAdaptiveExploration;
import com.epsilon.pico.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.pico.ai.decision.audit.EpsilonDecisionSelectedPgDebugAudit;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.config.settings.DecisionSelectedPgCampaignSettings;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 選択行動の方策勾配学習について、進捗ログの出力を状態遷移の処理から分離する。 */
final class DecisionSelectedPgCampaignLogger {

  private static final Logger log =
      LoggerFactory.getLogger(EpsilonDecisionSelectedPgCampaign.class);

  private final Path checkpointRoot;
  private final DecisionSelectedPgCampaignSettings settings;
  private final DecisionSelectedPgRunContext run;
  private final String learnerContractId;

  DecisionSelectedPgCampaignLogger(
      Path checkpointRoot,
      DecisionSelectedPgCampaignSettings settings,
      DecisionSelectedPgRunContext run,
      String learnerContractId) {
    this.checkpointRoot = checkpointRoot;
    this.settings = settings;
    this.run = run;
    this.learnerContractId = learnerContractId;
  }

  void start(
      boolean optimizerStateRestored,
      Path learnerCheckpoint,
      Path initialChampion,
      boolean grpEnabled,
      EpsilonDecisionPlayer.RolloutConfig actorRollout,
      EpsilonDecisionPlayer.RolloutConfig opponentRollout) {
    log.info(
        "Decision dense selected-PG learner initialized: source={} checkpoint={} "
            + "optimizerStateRestored={}",
        optimizerStateRestored ? "RESUMED_WORKING" : "ARENA_CHAMPION",
        learnerCheckpoint,
        optimizerStateRestored);
    log.info(
        "Decision dense selected-PG experiment contract: actorRollout={} opponentRollout={} "
            + "explorationCreditMix={} entropyCoefficient={}",
        actorRollout.summary(),
        opponentRollout.summary(),
        settings.explorationCreditMix(),
        settings.entropyCoefficient());
    log.info(
        "Decision dense selected-PG campaign start: checkpointDir={} macrosPerDuel={} "
            + "maximumMacros={} maximumDuels={} gamesPerMacro={} microBatchSize={} "
            + "maximumDeviceTransitionCells={} ppoEpochs={} optimizerStepsPerEpoch={} "
            + "optimizerStepsPerMacro={} actorInitialLearningRate={} valueLearningRate={} "
            + "actorTargetKlDecay=COSINE actorTargetKlDecayClock=ACCEPTED_ACTOR_OPTIMIZER_STEPS "
            + "actorTargetKlPlannedOptimizerSteps={} actorTargetKlControl=SAME_PATH_BEFORE_AFTER "
            + "actorTargetKlFinalScale={} "
            + "optimizerMode=INDEPENDENT_ACTOR_VALUE_ADAMW policyTrainMode=FULL_POLICY "
            + "legalCandidateWidth={} causalTraceLambda={} opportunityWeights={}/{}/{} "
            + "rolloutPolicyKlMode=DIAGNOSTIC_PRE_UPDATE_REJECT_ONLY policyUpdateClipRange={} "
            + "explorationCreditMix={} entropyCoefficient={} maximumOptimizerShardMeanKl={} "
            + "debugActorValidationEnabled={} debugFinalAuditEnabled={} "
            + "finalAuditSampleLimit={} samplePipeline=STREAMING_FRAGMENT "
            + "runSeedBase={} runSeedMode={} learnerContractId={} grpInferenceOnly={} "
            + "initialChampion={}",
        checkpointRoot,
        settings.macrosPerDuel(),
        settings.maximumMacros(),
        settings.maximumDuels(),
        settings.gamesPerMacro(),
        settings.microBatchSize(),
        settings.maximumDeviceTransitionCells(),
        settings.ppoEpochs(),
        settings.optimizerStepsPerEpoch(),
        settings.optimizerStepsPerMacro(),
        settings.optimizer().actorKlControl().initialLearningRate(),
        settings.optimizer().valueLearningRate(),
        settings.optimizer().actorKlControl().targetDecay().plannedOptimizerSteps(),
        settings.optimizer().actorKlControl().targetDecay().finalScale(),
        DecisionInputSchema.MAX_LEGAL_ACTIONS,
        settings.causalTraceLambda(),
        settings.dahaiWeight(),
        settings.riichiWeight(),
        settings.reactionWeight(),
        settings.policyUpdateClipRange(),
        settings.explorationCreditMix(),
        settings.entropyCoefficient(),
        settings.maximumOptimizerShardMeanKl(),
        settings.debugActorValidationEnabled(),
        settings.debugFinalAuditEnabled(),
        settings.finalAuditSampleLimit(),
        run.seedBase(),
        run.seedMode(),
        learnerContractId,
        grpEnabled,
        initialChampion);
  }

  void lineageStarted(
      int lineage,
      int candidateIteration,
      int campaignMacros,
      Path immutableChampion,
      Path learnerCheckpoint) {
    log.info(
        "Decision dense selected-PG lineage start: lineage={} candidateIteration={} "
            + "cumulativeMacros={} invocationMacroBudget={} champion={} optimizerCheckpoint={}",
        lineage,
        candidateIteration,
        campaignMacros,
        settings.maximumMacros(),
        immutableChampion,
        learnerCheckpoint);
  }

  void intervalStarted(
      int lineage,
      int duelRound,
      int candidateIteration,
      int campaignMacros,
      Path actorReplica,
      long[][] opponentIds) {
    log.info(
        "Decision dense selected-PG duel interval start: lineage={} duelRound={} "
            + "candidateIteration={} macros={} cumulativeMacros={} invocationMacroBudget={} "
            + "actorReplica={} opponentsByActorSeat={}",
        lineage,
        duelRound,
        candidateIteration,
        settings.macrosPerDuel(),
        campaignMacros,
        settings.maximumMacros(),
        actorReplica,
        Arrays.deepToString(opponentIds));
  }

  void macroRejected(
      int lineage,
      int duelRound,
      int candidateIteration,
      int campaignMacro,
      DecisionAdaptiveExploration.Report exploration,
      String reason) {
    log.warn(
        "Decision dense selected-PG macro rejected: lineage={} duelRound={} "
            + "candidateIteration={} campaignMacro={} exploration={} reason={}",
        lineage,
        duelRound,
        candidateIteration,
        campaignMacro,
        exploration.summary(),
        reason);
  }

  void macroCompleted(
      int lineage,
      int duelRound,
      int candidateIteration,
      int macroWithinInterval,
      int campaignMacro,
      int lineageMacro,
      DecisionSelectedPgMacroRunner.Execution execution,
      DecisionActorLearningControl.Adjustment adjustment,
      long elapsedMillis) {
    log.info(
        "Decision dense selected-PG macro complete: lineage={} duelRound={} candidateIteration={}"
            + " macro={}/{} cumulativeMacro={} invocationMacroBudget={} lineageMacro={} games={}"
            + " samples={} actorLr={}->{} lrAction={} actorKlAcceptedMacros={}"
            + " acceptedActorOptimizerSteps={} actorUpdateKl={} actorMedianKl={} actorTargetKl={}"
            + " actor={} value={} exploration={} audit={} elapsedMs={}",
        lineage,
        duelRound,
        candidateIteration,
        macroWithinInterval,
        settings.macrosPerDuel(),
        campaignMacro,
        settings.maximumMacros(),
        lineageMacro,
        execution.games(),
        execution.training().trainingSamples(),
        adjustment.appliedLearningRate(),
        adjustment.nextLearningRate(),
        adjustment.action(),
        adjustment.acceptedMacros(),
        adjustment.acceptedActorOptimizerSteps(),
        adjustment.observedMeanKl(),
        adjustment.windowMedianMeanKl(),
        adjustment.targetMeanKl(),
        trainingMetricsSummary(execution.training().actorMetrics()),
        trainingMetricsSummary(execution.training().valueMetrics()),
        execution.exploration().summary(),
        auditSummary(execution.training().audit()),
        elapsedMillis);
  }

  void intervalCompleted(
      int lineage,
      int duelRound,
      int candidateIteration,
      DecisionSelectedPgDuelResult.Status status,
      int intervalMacros,
      int campaignMacros,
      int lineageMacros,
      int games,
      long samples,
      Path candidate,
      long elapsedMillis) {
    log.info(
        "Decision dense selected-PG duel interval complete: lineage={} duelRound={}"
            + " candidateIteration={} status={} intervalMacros={}/{} cumulativeMacros={}"
            + " invocationMacroBudget={} lineageMacros={} games={} samples={} preserveLearner={}"
            + " candidate={} elapsedMs={}",
        lineage,
        duelRound,
        candidateIteration,
        status,
        intervalMacros,
        settings.macrosPerDuel(),
        campaignMacros,
        settings.maximumMacros(),
        lineageMacros,
        games,
        samples,
        status.preservesLearnerState(),
        candidate,
        elapsedMillis);
  }

  private String auditSummary(Optional<EpsilonDecisionSelectedPgDebugAudit.Result> audit) {
    return audit
        .map(EpsilonDecisionSelectedPgDebugAudit.Result::summary)
        .orElse(settings.debugFinalAuditEnabled() ? "not-run" : "disabled");
  }

  private static String trainingMetricsSummary(DecisionTrainingResult metrics) {
    return String.format(
        Locale.ROOT,
        "training{optimizerSteps=%d,microBatches=%d,loss=%.6f,actorLoss=%.6f,"
            + "entropyBonusLoss=%.6f,behaviorCloningLoss=%.6f,valueLoss=%.6f,entropy=%.6f,"
            + "rolloutEntropy=%.6f,behaviorToRolloutPolicyKl=%.6f,chosenProb=%.6f,"
            + "behaviorProbMean=%.6f,behaviorProbMin=%.6f,meanRolloutPolicyKl=%.6f,"
            + "maximumRolloutPolicyKl=%.6f,actorWeight=%.6f,policyRatios=%s,"
            + "scalarAdvantageSigns=+%d/-%d/0:%d,rejected=%s,reason=%s,%s}",
        metrics.optimizerSteps(),
        metrics.microBatches(),
        metrics.loss(),
        metrics.actorLoss(),
        metrics.entropyBonusLoss(),
        metrics.behaviorCloningLoss(),
        metrics.valueLoss(),
        metrics.entropy(),
        metrics.rolloutEntropy(),
        metrics.behaviorToRolloutPolicyKl(),
        metrics.chosenProb(),
        metrics.behaviorProbMean(),
        metrics.behaviorProbMin(),
        metrics.meanRolloutPolicyKl(),
        metrics.maximumRolloutPolicyKl(),
        metrics.actorWeight(),
        metrics.policyRatios(),
        metrics.positiveScalarAdvantageSamples(),
        metrics.negativeScalarAdvantageSamples(),
        metrics.zeroScalarAdvantageSamples(),
        metrics.rejected(),
        metrics.reason(),
        metrics.performance());
  }
}

package com.epsilon.nano.ai.decision.training;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionFullSupportSettings;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.DecisionTrainSettings;
import com.epsilon.config.settings.DecisionTrainingDeviceTransferSchedule;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.decision.arena.DecisionAdaptiveExploration;
import com.epsilon.nano.ai.decision.audit.EpsilonDecisionFinalPolicyAudit;
import com.epsilon.nano.ai.decision.audit.EpsilonDecisionSelectedPgDebugAudit;
import com.epsilon.nano.ai.decision.data.DecisionJsonFiles;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings.ActorKlControlSettings;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/** 選択行動の方策勾配学習と対戦評価の一連の実行の機械可読レポートをバージョン付きJSONへ統一して保存する。 */
final class DecisionSelectedPgReportWriter {

  static final String MACRO_FILE_PREFIX = "selected-pg-macro-";
  static final String DUEL_FILE = "selected-pg-duel.json";
  static final String LATEST_DUEL_FILE = "decision-selected-pg.json";
  static final String CAMPAIGN_FILE = "decision-train.json";
  static final String CHAMPION_DUEL_FILE = "decision-champion-duel.json";

  private static final int SCHEMA_VERSION = 1;
  private static final int MACRO_SCHEMA_VERSION = 4;
  private static final Gson GSON = new GsonBuilder().serializeNulls().setPrettyPrinting().create();

  private DecisionSelectedPgReportWriter() {}

  static void writeMacro(
      Path duelDir,
      int lineage,
      int duelRound,
      int checkpointIteration,
      int campaignMacro,
      int lineageMacro,
      int games,
      DecisionSelectedPgMacroRunner.TrainingResult training,
      DecisionActorLearningControl.Adjustment learningRate,
      DecisionAdaptiveExploration.Report exploration,
      long collectionElapsedMillis,
      long learnerElapsedMillis,
      long elapsedMillis)
      throws IOException {
    EpsilonDecisionFinalPolicyAudit.Report finalAudit =
        training.audit().map(EpsilonDecisionSelectedPgDebugAudit.Result::report).orElse(null);
    MacroReport report =
        new MacroReport(
            "selected-pg-macro-v4",
            MACRO_SCHEMA_VERSION,
            lineage,
            duelRound,
            checkpointIteration,
            campaignMacro,
            lineageMacro,
            training.passed() ? MacroStatus.ACCEPTED : MacroStatus.REJECTED,
            games,
            training.trainingSamples(),
            training.optimizerSteps(),
            "SAME_PATH_BEFORE_AFTER_ACTOR_WEIGHTED",
            learningRate,
            training.actorMetrics(),
            training.valueMetrics(),
            finalAudit,
            exploration,
            training.reason(),
            collectionElapsedMillis,
            learnerElapsedMillis,
            elapsedMillis);
    write(
        duelDir.resolve(MACRO_FILE_PREFIX + String.format("%03d", campaignMacro) + ".json"),
        report);
  }

  static void writeDuel(
      Path checkpointDir,
      Path duelDir,
      DecisionSelectedPgCampaignSettings settings,
      String learnerContractId,
      long runSeedBase,
      DecisionSelectedPgRunContext.SeedMode runSeedMode,
      DecisionSelectedPgDuelResult result,
      DecisionTrainingResult actorMetrics,
      DecisionTrainingResult valueMetrics,
      Optional<EpsilonDecisionSelectedPgDebugAudit.Result> audit,
      DecisionActorLearningControl.Adjustment learningRate,
      int optimizerReinitializations,
      int optimizerRestorations,
      long elapsedMillis,
      SettingsLoader config)
      throws IOException {
    DuelReport report =
        new DuelReport(
            "selected-pg-duel-v1",
            SCHEMA_VERSION,
            EffectiveSettings.from(settings, config),
            learnerContractId,
            runSeedBase,
            runSeedMode,
            DuelSummary.from(result),
            learningRate,
            actorMetrics,
            valueMetrics,
            audit.map(EpsilonDecisionSelectedPgDebugAudit.Result::report).orElse(null),
            optimizerReinitializations,
            optimizerRestorations,
            elapsedMillis);
    write(duelDir.resolve(DUEL_FILE), report);
    write(checkpointDir.resolve(LATEST_DUEL_FILE), report);
  }

  static void writeCampaign(
      Path checkpointDir,
      DecisionSelectedPgCampaignSettings settings,
      String learnerContractId,
      long runSeedBase,
      DecisionSelectedPgRunContext.SeedMode runSeedMode,
      DecisionSelectedPgCampaignResult result,
      Path arenaChampion,
      Path productionChampion,
      Path workingCheckpoint,
      SettingsLoader config)
      throws IOException {
    CampaignReport report =
        new CampaignReport(
            "selected-pg-campaign-v1",
            SCHEMA_VERSION,
            EffectiveSettings.from(settings, config),
            learnerContractId,
            runSeedBase,
            runSeedMode,
            result.maximumMacros(),
            result.completedMacros(),
            result.completed(),
            result.status(),
            result.duels().size(),
            result.promotions(),
            result.games(),
            result.samples(),
            result.duels().stream().map(DuelSummary::from).toList(),
            arenaChampion.toString(),
            productionChampion.toString(),
            workingCheckpoint.toString());
    write(checkpointDir.resolve(CAMPAIGN_FILE), report);
  }

  static void writeChampionDuel(Path reportDir, DecisionSelectedPgDuelRunner.ChampionDuel duel)
      throws IOException {
    EpsilonDecisionWallDuelEvaluator.Result result = duel.result();
    DuelEvaluation.Result descriptive = result.descriptiveResult();
    ChampionDuelReport report =
        new ChampionDuelReport(
            "decision-champion-duel-v1",
            SCHEMA_VERSION,
            duel.iteration(),
            duel.lineage(),
            duel.duelRound(),
            duel.seedBase(),
            result.games(),
            result.wallSeeds(),
            result.plannedWallSeeds(),
            result.progressUpdates(),
            result.completedLooks(),
            result.mode(),
            result.duelSequence(),
            result.alpha(),
            result.decisionAlpha(),
            result.decision(),
            result.utilityProfile(),
            result.pairedUtilityDeltaMean(),
            result.pairedUtilityDeltaLower(),
            result.pairedUtilityDeltaUpper(),
            result.pairedRankDeltaMean(),
            result.pairedRankDeltaLower(),
            result.pairedRankDeltaUpper(),
            result.promotionMargin(),
            result.harmfulMargin(),
            result.candidateCheckpoint().toString(),
            result.parentCheckpoint().toString(),
            new DescriptiveDuelMetrics(
                descriptive.candidateAverageRank(),
                descriptive.opponentAverageRank(),
                descriptive.candidateTopRate(),
                descriptive.opponentTopRate(),
                descriptive.candidateLastRate(),
                descriptive.opponentLastRate(),
                descriptive.scoreAdvantage()));
    write(reportDir.resolve(CHAMPION_DUEL_FILE), report);
  }

  private static void write(Path target, Object report) throws IOException {
    DecisionJsonFiles.write(target, report, GSON);
  }

  private record EffectiveSettings(
      int macrosPerDuel,
      int maximumMacros,
      int gamesPerMacro,
      DecisionFullSupportSettings fullSupport,
      DecisionTensorTransfer trainingTensorTransfer,
      DecisionTrainingDeviceTransferSchedule trainingDeviceTransferSchedule,
      DecisionComputePrecision trainingComputePrecision,
      int microBatchSize,
      int maximumDeviceTransitionCells,
      int ppoEpochs,
      int optimizerStepsPerEpoch,
      int optimizerStepsPerMacro,
      float valueLearningRate,
      ActorKlControlSettings actorKlControl,
      float causalTraceLambda,
      float dahaiWeight,
      float riichiWeight,
      float reactionWeight,
      float policyUpdateClipRange,
      float explorationCreditMix,
      float entropyCoefficient,
      float maximumOptimizerShardMeanKl,
      boolean debugActorValidationEnabled,
      boolean debugFinalAuditEnabled,
      int finalAuditSampleLimit,
      long configuredRunSeedBase) {

    private static EffectiveSettings from(
        DecisionSelectedPgCampaignSettings settings, SettingsLoader config) {
      DecisionTrainSettings training = config.bind(DecisionTrainSettings.class);
      return new EffectiveSettings(
          settings.macrosPerDuel(),
          settings.maximumMacros(),
          settings.gamesPerMacro(),
          config.bind(DecisionFullSupportSettings.class),
          training.tensorTransfer(),
          training.deviceTransferSchedule(),
          training.computePrecision(),
          settings.microBatchSize(),
          settings.maximumDeviceTransitionCells(),
          settings.ppoEpochs(),
          settings.optimizerStepsPerEpoch(),
          settings.optimizerStepsPerMacro(),
          settings.optimizer().valueLearningRate(),
          settings.optimizer().actorKlControl(),
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
          settings.runSeedBase());
    }
  }

  private enum MacroStatus {
    ACCEPTED,
    REJECTED
  }

  private record MacroReport(
      String schema,
      int schemaVersion,
      int lineage,
      int duelRound,
      int checkpointIteration,
      int campaignMacro,
      int lineageMacro,
      MacroStatus status,
      int games,
      int trainingSamples,
      int optimizerSteps,
      String klAggregation,
      DecisionActorLearningControl.Adjustment actorLearningRate,
      DecisionTrainingResult actorMetrics,
      DecisionTrainingResult valueMetrics,
      EpsilonDecisionFinalPolicyAudit.Report finalAudit,
      DecisionAdaptiveExploration.Report exploration,
      String reason,
      long collectionElapsedMillis,
      long learnerElapsedMillis,
      long elapsedMillis) {}

  private record DuelReport(
      String schema,
      int schemaVersion,
      EffectiveSettings settings,
      String learnerContractId,
      long runSeedBase,
      DecisionSelectedPgRunContext.SeedMode runSeedMode,
      DuelSummary result,
      DecisionActorLearningControl.Adjustment lastActorLearningRate,
      DecisionTrainingResult lastActorMetrics,
      DecisionTrainingResult lastValueMetrics,
      EpsilonDecisionFinalPolicyAudit.Report finalAudit,
      int optimizerReinitializations,
      int optimizerRestorations,
      long elapsedMillis) {}

  private record CampaignReport(
      String schema,
      int schemaVersion,
      EffectiveSettings settings,
      String learnerContractId,
      long runSeedBase,
      DecisionSelectedPgRunContext.SeedMode runSeedMode,
      int maximumMacros,
      int completedMacros,
      boolean completed,
      DecisionSelectedPgDuelResult.Status status,
      int completedDuels,
      long promotions,
      int games,
      long samples,
      List<DuelSummary> duels,
      String arenaChampion,
      String productionChampion,
      String workingCheckpoint) {}

  private record DuelSummary(
      int lineage,
      int duelRound,
      int checkpointIteration,
      int campaignMacros,
      int lineageMacros,
      int intervalMacros,
      int intervalGames,
      long intervalSamples,
      DecisionSelectedPgDuelResult.Status status,
      boolean promoted,
      String champion,
      String candidate,
      String reason) {

    private static DuelSummary from(DecisionSelectedPgDuelResult result) {
      return new DuelSummary(
          result.lineage(),
          result.duelRound(),
          result.checkpointIteration(),
          result.campaignMacros(),
          result.lineageMacros(),
          result.intervalMacros(),
          result.intervalGames(),
          result.intervalSamples(),
          result.status(),
          result.promoted(),
          result.champion().toString(),
          result.candidate().toString(),
          result.reason());
    }
  }

  private record ChampionDuelReport(
      String schema,
      int schemaVersion,
      int iteration,
      int lineage,
      int duelRound,
      long seedBase,
      int games,
      int wallSeeds,
      int plannedWallSeeds,
      int progressUpdates,
      int completedLooks,
      EpsilonDecisionWallDuelEvaluator.Mode mode,
      long duelSequence,
      double alpha,
      double decisionAlpha,
      EpsilonDecisionWallDuelEvaluator.Decision decision,
      EpsilonUtilityProfile utilityProfile,
      double pairedUtilityDeltaMean,
      double pairedUtilityDeltaLower,
      double pairedUtilityDeltaUpper,
      double pairedRankDeltaMean,
      double pairedRankDeltaLower,
      double pairedRankDeltaUpper,
      double promotionMargin,
      double harmfulMargin,
      String candidateCheckpoint,
      String parentCheckpoint,
      DescriptiveDuelMetrics descriptive) {}

  private record DescriptiveDuelMetrics(
      double candidateAverageRank,
      double opponentAverageRank,
      double candidateTopRate,
      double opponentTopRate,
      double candidateLastRate,
      double opponentLastRate,
      double scoreAdvantage) {}
}

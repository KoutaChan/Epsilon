package com.epsilon.nano.ai.decision.training;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator;
import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.config.settings.DecisionTrainArenaSettings;
import com.epsilon.config.settings.DecisionTrainSettings;
import com.epsilon.config.settings.DecisionTrainSpoolSettings;
import com.epsilon.config.settings.GrpSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionPopulationCollector;
import com.epsilon.nano.ai.decision.audit.EpsilonDecisionSelectedPgDebugAudit;
import com.epsilon.nano.ai.decision.runtime.DecisionGpuMemoryDiagnostics;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.util.FormatUtils;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Optional;

/** 採用済みモデルから自己対局と方策勾配学習を開始し、定期的な対戦比較の結果に応じて学習を継続または停止する。 */
public final class EpsilonDecisionSelectedPgCampaign {

  private EpsilonDecisionSelectedPgCampaign() {}

  /** モデル、学習器、スナップショット、対戦評価を一つの連続学習として完走させる。 */
  public static DecisionSelectedPgCampaignResult train(
      Path checkpointRoot, DecisionSelectedPgCampaignSettings settings) throws Exception {
    return train(checkpointRoot, settings, EpsilonSettings.defaults());
  }

  public static DecisionSelectedPgCampaignResult train(
      Path checkpointRoot, DecisionSelectedPgCampaignSettings settings, SettingsLoader config)
      throws Exception {
    try (Session session = Session.open(checkpointRoot, settings, config)) {
      return session.run();
    }
  }

  /** 乱数による初期化直後のチェックポイントを本番自己対局の親に使わせない。 */
  static void requirePretrainedCheckpoint(
      EpsilonDecisionCheckpointBundle manifest, Path checkpoint) {
    EpsilonDecisionCheckpointBundle actual = Objects.requireNonNull(manifest, "manifest");
    if (actual.globalStep <= 0 || actual.iteration <= 0) {
      throw new IllegalStateException(
          "Decision dense selected-PG requires supervised pretraining before self-play: "
              + "run pretrain-decision-logs first; globalStep="
              + actual.globalStep
              + " iteration="
              + actual.iteration
              + " checkpoint="
              + checkpoint);
    }
  }

  /** 一学習と対戦評価の一連の実行の資源と変更可能な進捗を所有する。 */
  private static final class Session implements AutoCloseable {

    private final SettingsLoader config;
    private final Path checkpointRoot;
    private final Path workingCheckpoint;
    private final DecisionSelectedPgCampaignSettings settings;
    private final String learnerContractId;
    private final DecisionSelectedPgRunContext run;
    private final DecisionExecutionContext executionContext;
    private final DecisionLearner learner;
    private final EpsilonGrpTrainingSession grpSession;
    private final EpsilonDecisionSnapshotPool snapshotPool;
    private final DecisionSelectedPgMacroRunner macroRunner;
    private final DecisionSelectedPgCampaignLogger campaignLog;
    private final Progress progress;
    private final int initialMacros;
    private final int initialGames;
    private final long initialSamples;
    private final int budgetEnd;
    private Lineage activeLineage;
    private Interval activeInterval;

    private Session(
        Path checkpointRoot,
        DecisionSelectedPgCampaignSettings settings,
        String learnerContractId,
        DecisionSelectedPgRunContext run,
        DecisionLearner learner,
        EpsilonGrpTrainingSession grpSession,
        EpsilonDecisionSnapshotPool snapshotPool,
        Progress progress,
        EpsilonDecisionPlayer.RolloutConfig actorRollout,
        EpsilonDecisionPlayer.RolloutConfig opponentRollout,
        DecisionExecutionContext executionContext,
        SettingsLoader config) {
      this.config = config;
      this.checkpointRoot = checkpointRoot;
      this.workingCheckpoint = EpsilonDecisionCheckpointManager.working(checkpointRoot);
      this.settings = settings;
      this.learnerContractId = learnerContractId;
      this.run = run;
      this.executionContext = executionContext;
      this.learner = learner;
      this.grpSession = grpSession;
      this.snapshotPool = snapshotPool;
      this.progress = progress;
      initialMacros = progress.completedMacros;
      initialGames = progress.games;
      initialSamples = progress.samples;
      budgetEnd = Math.addExact(initialMacros, settings.maximumMacros());
      campaignLog =
          new DecisionSelectedPgCampaignLogger(checkpointRoot, settings, run, learnerContractId);
      macroRunner =
          new DecisionSelectedPgMacroRunner(
              learner,
              grpSession,
              snapshotPool,
              settings,
              actorRollout,
              opponentRollout,
              config.bind(DecisionTrainSettings.class).sampleLimit(),
              config.bind(DecisionTrainSpoolSettings.class).keepFragments(),
              executionContext);
    }

    private static Session open(
        Path checkpointRoot, DecisionSelectedPgCampaignSettings settings, SettingsLoader config)
        throws Exception {
      Path root =
          Objects.requireNonNull(checkpointRoot, "checkpointRoot").toAbsolutePath().normalize();
      DecisionSelectedPgCampaignSettings actual = Objects.requireNonNull(settings, "settings");
      Path arenaChampion = EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(root);
      if (arenaChampion == null) {
        throw new IOException(
            "Decision dense selected-PG requires a pretrained arena champion: " + root);
      }
      EpsilonDecisionCheckpointBundle initialManifest =
          EpsilonDecisionCheckpointManager.loadManifest(arenaChampion);
      requirePretrainedCheckpoint(initialManifest, arenaChampion);

      var resolved = DecisionSelectedPgLearnerContract.resolve(actual, config);
      EpsilonDecisionPlayer.RolloutConfig actorRollout = resolved.actorRollout();
      EpsilonDecisionPlayer.RolloutConfig opponentRollout = resolved.opponentRollout();
      EpsilonDecisionPopulationCollector.requireSelectedPgActor(actorRollout);
      String learnerContractId = resolved.learnerContractId();
      EpsilonDecisionCheckpointManager.recoverWorkingCheckpoint(root);
      Path working = EpsilonDecisionCheckpointManager.working(root);
      DecisionSelectedPgCampaignState saved = DecisionSelectedPgCampaignState.read(working);
      if (saved == null
          && java.nio.file.Files.isRegularFile(
              working.resolve(DecisionLearnerCheckpoint.STATE_FILE))) {
        throw new IOException(
            "Working learner has no campaign resume state; use an explicit campaign bootstrap: "
                + working);
      }
      DecisionSelectedPgRunContext run =
          saved == null
              ? DecisionSelectedPgRunContext.create(
                  root, initialManifest.iteration, actual.runSeedBase())
              : saved.run();
      if (!root.toAbsolutePath()
          .normalize()
          .equals(run.checkpointRoot().toAbsolutePath().normalize())) {
        throw new IOException("Campaign resume belongs to another checkpoint root");
      }

      Device learnerDevice =
          NetworkFactory.getLearnerDevice(
              config.bind(com.epsilon.config.settings.DeviceSettings.class));
      Device grpDevice =
          NetworkFactory.getGrpDevices(
                  config.bind(com.epsilon.config.settings.DeviceSettings.class))
              .primary();
      Model model = null;
      DecisionLearner learner = null;
      EpsilonGrpTrainingSession grpSession = null;
      DecisionExecutionContext executionContext = new DecisionExecutionContext();
      try {
        boolean restored = saved != null;
        if (restored) {
          learner =
              DecisionLearner.openWorking(
                  working,
                  learnerDevice,
                  actual.optimizer(),
                  learnerContractId,
                  executionContext,
                  config);
        } else {
          model = EpsilonDecisionCheckpointManager.load(arenaChampion, learnerDevice);
          learner =
              new DecisionLearner(
                  model,
                  learnerDevice,
                  actual.optimizer(),
                  learnerContractId,
                  executionContext,
                  config);
        }
        grpSession =
            config.bind(GrpSettings.class).enabled()
                ? EpsilonGrpTrainingSession.openInferenceOnly(root, grpDevice, config)
                : EpsilonGrpTrainingSession.disabled(root.resolve("grp"));
        Path learnerCheckpoint = restored ? working : arenaChampion;
        EpsilonDecisionCheckpointBundle generation =
            restored
                ? EpsilonDecisionCheckpointManager.requireValidCheckpoint(working)
                : initialManifest;
        EpsilonDecisionSnapshotPool snapshotPool =
            EpsilonDecisionSnapshotPool.load(
                root,
                config.bind(com.epsilon.config.settings.DecisionSnapshotPoolSettings.class).max());
        snapshotPool.registerArenaChampion(arenaChampion);
        snapshotPool.save(root);
        Progress progress =
            new Progress(
                arenaChampion,
                learnerCheckpoint,
                generation.globalStep,
                generation.iteration,
                generation.selfPlayGames);
        if (saved != null) {
          progress.completedMacros = saved.completedMacros();
          progress.games = saved.games();
          progress.samples = saved.samples();
          progress.lineage = saved.lineageSequence();
          progress.arenaChampion = saved.arenaChampion();
        }
        Session session =
            new Session(
                root,
                actual,
                learnerContractId,
                run,
                learner,
                grpSession,
                snapshotPool,
                progress,
                actorRollout,
                opponentRollout,
                executionContext,
                config);
        if (saved != null) {
          session.restoreProgress(saved);
        } else {
          session.saveWorking();
        }
        session.campaignLog.start(
            restored,
            progress.learnerCheckpoint,
            progress.arenaChampion,
            grpSession.enabled(),
            actorRollout,
            opponentRollout);
        return session;
      } catch (Exception | Error failure) {
        if (grpSession != null) {
          try {
            grpSession.close();
          } catch (RuntimeException closeFailure) {
            failure.addSuppressed(closeFailure);
          }
        }
        if (learner != null) {
          try {
            learner.close();
          } catch (RuntimeException closeFailure) {
            failure.addSuppressed(closeFailure);
          }
        } else if (model != null) {
          try {
            model.close();
          } catch (RuntimeException | Error closeFailure) {
            failure.addSuppressed(closeFailure);
          }
        }
        try {
          executionContext.close();
        } catch (RuntimeException | Error closeFailure) {
          failure.addSuppressed(closeFailure);
        }
        throw failure;
      }
    }

    private DecisionSelectedPgCampaignResult run() throws Exception {
      while (progress.terminalStatus == null) {
        if (activeLineage == null) {
          if (!hasMacroBudget()) break;
          activeLineage = beginLineage();
          saveProgress();
        }
        if (activeInterval == null) {
          if (!hasMacroBudget()) break;
          activeInterval = beginInterval(activeLineage);
          saveProgress();
        }
        Lineage lineage = activeLineage;
        Interval interval = activeInterval;
        if (interval.guardFailure.isEmpty() && interval.completedMacros < interval.plannedMacros) {
          if (!hasMacroBudget()) break;
          runMacro(lineage, interval, interval.completedMacros + 1);
          continue;
        }
        DecisionSelectedPgDuelRunner.Resolution resolution = resolveInterval(lineage, interval);
        DecisionSelectedPgDuelResult duelResult = toDuelResult(lineage, interval, resolution);
        progress.duels.add(duelResult);
        applyResolution(lineage, resolution, duelResult);
        writeDuelReport(interval, duelResult);
        campaignLog.intervalCompleted(
            lineage.number,
            interval.duelRound,
            lineage.candidateIteration,
            resolution.status(),
            interval.completedMacros,
            progress.completedMacros,
            lineage.completedMacros,
            interval.games,
            interval.samples,
            duelResult.candidate(),
            elapsedMillis(interval.startedNanos));
        activeInterval = null;
        if (resolution.status() == DecisionSelectedPgDuelResult.Status.PROMOTED) {
          activeLineage = null;
        } else if (!resolution.status().preservesLearnerState()) {
          activeLineage = null;
          progress.terminalStatus = resolution.status();
        }
        if (resolution.status().preservesLearnerState()) saveProgress();
        else saveWorking();
      }
      DecisionSelectedPgDuelResult.Status status =
          progress.terminalStatus == null
              ? DecisionSelectedPgDuelResult.Status.PAUSED
              : progress.terminalStatus;
      progress.terminalStatus = status;
      saveProgress();
      DecisionSelectedPgCampaignResult result =
          new DecisionSelectedPgCampaignResult(
              settings.maximumMacros(),
              progress.completedMacros - initialMacros,
              progress.games - initialGames,
              progress.samples - initialSamples,
              progress.duels,
              status);
      DecisionSelectedPgReportWriter.writeCampaign(
          checkpointRoot,
          settings,
          learnerContractId,
          run.seedBase(),
          run.seedMode(),
          result,
          EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(checkpointRoot),
          EpsilonDecisionCheckpointManager.resolveProductionChampionStrict(checkpointRoot),
          workingCheckpoint,
          config);
      return result;
    }

    private boolean hasMacroBudget() {
      return progress.completedMacros < budgetEnd;
    }

    private void saveWorking() throws IOException {
      DecisionSelectedPgCampaignState state = snapshotProgress();
      EpsilonDecisionCheckpointManager.saveWorking(
          learner,
          checkpointRoot,
          progress.globalStep,
          progress.iteration,
          progress.selfPlayGames,
          state::write);
      progress.learnerCheckpoint = workingCheckpoint;
    }

    private void saveProgress() throws IOException {
      snapshotProgress().write(workingCheckpoint);
    }

    private DecisionSelectedPgCampaignState snapshotProgress() {
      DecisionSelectedPgCampaignState.Lineage lineage =
          activeLineage == null
              ? null
              : new DecisionSelectedPgCampaignState.Lineage(
                  activeLineage.number,
                  activeLineage.candidateIteration,
                  activeLineage.directory,
                  activeLineage.immutableParent,
                  activeLineage.parentLearner,
                  activeLineage.seedBase,
                  activeLineage.completedMacros,
                  activeLineage.duelRound);
      DecisionSelectedPgCampaignState.Interval interval =
          activeInterval == null
              ? null
              : new DecisionSelectedPgCampaignState.Interval(
                  activeInterval.duelRound,
                  activeInterval.directory,
                  activeInterval.seedBase,
                  activeInterval.opponentIds,
                  activeInterval.plannedMacros,
                  activeInterval.duelSettings,
                  activeInterval.games,
                  activeInterval.samples,
                  activeInterval.completedMacros,
                  activeInterval.guardFailure.orElse(null),
                  activeInterval.lastActorMetrics,
                  activeInterval.lastValueMetrics,
                  activeInterval.lastAudit.orElse(null),
                  activeInterval.lastLearningRateAdjustment,
                  activeInterval.candidate);
      return new DecisionSelectedPgCampaignState(
          1,
          run,
          progress.globalStep,
          progress.iteration,
          progress.selfPlayGames,
          progress.completedMacros,
          progress.games,
          progress.samples,
          progress.lineage,
          progress.arenaChampion,
          lineage,
          interval,
          macroRunner.explorationState(),
          progress.terminalStatus);
    }

    private void restoreProgress(DecisionSelectedPgCampaignState saved) throws IOException {
      macroRunner.restoreExplorationState(saved.exploration());
      if (saved.lineage() != null) {
        var state = saved.lineage();
        activeLineage =
            new Lineage(
                state.number(),
                state.candidateIteration(),
                EpsilonDecisionCheckpointManager.loadManifest(state.immutableParent()),
                state.directory(),
                state.immutableParent(),
                state.parentLearner(),
                state.seedBase(),
                workingCheckpoint);
        activeLineage.completedMacros = state.completedMacros();
        activeLineage.duelRound = state.duelRound();
      }
      if (saved.interval() != null) {
        var state = saved.interval();
        activeInterval =
            new Interval(
                state.duelRound(),
                state.directory(),
                state.seedBase(),
                state.opponentIds(),
                state.plannedMacros(),
                state.duelSettings(),
                System.nanoTime());
        activeInterval.games = state.games();
        activeInterval.samples = state.samples();
        activeInterval.completedMacros = state.completedMacros();
        activeInterval.guardFailure = Optional.ofNullable(state.guardFailure());
        activeInterval.lastActorMetrics = state.lastActorMetrics();
        activeInterval.lastValueMetrics = state.lastValueMetrics();
        activeInterval.lastAudit = Optional.ofNullable(state.lastAudit());
        activeInterval.lastLearningRateAdjustment = state.lastLearningRateAdjustment();
        activeInterval.candidate = state.candidate();
      }
    }

    private Lineage beginLineage() throws IOException {
      int lineageNumber = Math.addExact(progress.lineage, 1);
      progress.lineage = lineageNumber;
      Path parent = requireArenaChampion(progress.arenaChampion);
      EpsilonDecisionCheckpointBundle parentManifest =
          EpsilonDecisionCheckpointManager.loadManifest(parent);
      int candidateIteration = Math.addExact(parentManifest.iteration, 1);
      Path directory = run.lineageDirectory(lineageNumber, candidateIteration);
      Path immutableParent = directory.resolve("parent");
      EpsilonDecisionCheckpointManager.copyAtIteration(
          parent, immutableParent, parentManifest.iteration);
      Lineage lineage =
          new Lineage(
              lineageNumber,
              candidateIteration,
              parentManifest,
              directory,
              immutableParent,
              directory.resolve("parent-learner"),
              run.lineageSeed(lineageNumber),
              progress.learnerCheckpoint);
      EpsilonDecisionCheckpointManager.saveLearner(
          learner,
          lineage.parentLearner,
          progress.globalStep,
          progress.iteration,
          progress.selfPlayGames,
          snapshotProgress()::write);
      campaignLog.lineageStarted(
          lineage.number,
          lineage.candidateIteration,
          progress.completedMacros,
          lineage.immutableParent,
          progress.learnerCheckpoint);
      return lineage;
    }

    private Interval beginInterval(Lineage lineage) {
      int duelRound = Math.addExact(lineage.duelRound, 1);
      lineage.duelRound = duelRound;
      Path directory = run.duelDirectory(lineage.directory, duelRound);
      long seedBase = run.duelSeed(lineage.seedBase, duelRound);
      long[][] opponentIds =
          snapshotPool.sampleOpponentIdsForSeats(
              lineage.candidateIteration,
              seedBase,
              config.bind(DecisionTrainArenaSettings.class).maximumOpponentSnapshotsPerInterval());
      Interval interval =
          new Interval(
              duelRound,
              directory,
              seedBase,
              opponentIds,
              settings.macrosPerDuel(),
              config.bind(DecisionChampionDuelSettings.class),
              System.nanoTime());
      campaignLog.intervalStarted(
          lineage.number,
          duelRound,
          lineage.candidateIteration,
          progress.completedMacros,
          lineage.actorReplicaCheckpoint,
          opponentIds);
      return interval;
    }

    private boolean runMacro(Lineage lineage, Interval interval, int macroWithinInterval)
        throws Exception {
      long macroStartedNanos = System.nanoTime();
      int campaignMacro = Math.addExact(progress.completedMacros, 1);
      int lineageMacro = Math.addExact(lineage.completedMacros, 1);
      long actorSnapshotId = run.actorSnapshotId(lineage.seedBase, lineageMacro);
      long macroSeed = interval.seedBase + (long) macroWithinInterval * 1_000_000L;
      DecisionSelectedPgMacroRunner.Execution execution =
          macroRunner.run(
              new DecisionSelectedPgMacroRunner.Request(
                  run.spoolDirectory(
                      lineage.number,
                      lineage.candidateIteration,
                      interval.duelRound,
                      campaignMacro,
                      config.bind(DecisionTrainSpoolSettings.class).dir()),
                  lineage.actorReplicaCheckpoint,
                  actorSnapshotId,
                  interval.opponentIds,
                  settings.gamesPerMacro(),
                  macroSeed));
      DecisionSelectedPgMacroRunner.TrainingResult training = execution.training();
      progress.games = Math.addExact(progress.games, execution.games());
      progress.samples = Math.addExact(progress.samples, execution.collectedSamples());
      progress.selfPlayGames = Math.addExact(progress.selfPlayGames, execution.games());
      interval.record(execution);
      progress.globalStep = Math.addExact(progress.globalStep, training.optimizerSteps());

      if (!training.passed()) {
        DecisionActorLearningControl.Adjustment adjustment =
            learner.rejectMacroKl(training.meanActorUpdateKl());
        interval.lastLearningRateAdjustment = adjustment;
        interval.guardFailure =
            Optional.of(
                "duelRound="
                    + interval.duelRound
                    + ",macro="
                    + macroWithinInterval
                    + ",campaignMacro="
                    + campaignMacro
                    + ","
                    + training.reason());
        DecisionSelectedPgReportWriter.writeMacro(
            interval.directory,
            lineage.number,
            interval.duelRound,
            lineage.candidateIteration,
            campaignMacro,
            lineageMacro,
            execution.games(),
            training,
            adjustment,
            execution.exploration(),
            execution.collectionElapsedMillis(),
            execution.learnerElapsedMillis(),
            elapsedMillis(macroStartedNanos));
        rollbackRejectedMacro();
        campaignLog.macroRejected(
            lineage.number,
            interval.duelRound,
            lineage.candidateIteration,
            campaignMacro,
            execution.exploration(),
            interval.guardFailure.orElseThrow());
        return false;
      }

      DecisionActorLearningControl.Adjustment adjustment =
          learner.acceptMacroKl(training.meanActorUpdateKl(), training.optimizerSteps());
      interval.lastLearningRateAdjustment = adjustment;
      progress.completedMacros = campaignMacro;
      lineage.completedMacros = lineageMacro;
      interval.completedMacros++;
      progress.iteration = lineage.candidateIteration;
      Path macroCheckpoint =
          interval.directory.resolve("macro_" + FormatUtils.zeroPad(campaignMacro, 3));
      EpsilonDecisionCheckpointManager.save(
          learner.model(),
          macroCheckpoint,
          progress.globalStep,
          lineage.candidateIteration,
          progress.selfPlayGames);
      saveWorking();
      DecisionSelectedPgReportWriter.writeMacro(
          interval.directory,
          lineage.number,
          interval.duelRound,
          lineage.candidateIteration,
          campaignMacro,
          lineageMacro,
          execution.games(),
          training,
          adjustment,
          execution.exploration(),
          execution.collectionElapsedMillis(),
          execution.learnerElapsedMillis(),
          elapsedMillis(macroStartedNanos));
      lineage.actorReplicaCheckpoint = workingCheckpoint;
      progress.learnerCheckpoint = workingCheckpoint;
      campaignLog.macroCompleted(
          lineage.number,
          interval.duelRound,
          lineage.candidateIteration,
          macroWithinInterval,
          progress.completedMacros,
          lineage.completedMacros,
          execution,
          adjustment,
          elapsedMillis(macroStartedNanos));
      return true;
    }

    private void rollbackRejectedMacro() throws IOException {
      learner.reloadWorking(workingCheckpoint);
      EpsilonDecisionCheckpointBundle rollback =
          EpsilonDecisionCheckpointManager.loadManifest(workingCheckpoint);
      macroRunner.restoreExplorationState(
          DecisionSelectedPgCampaignState.read(workingCheckpoint).exploration());
      progress.globalStep = rollback.globalStep;
      progress.iteration = rollback.iteration;
      progress.selfPlayGames = rollback.selfPlayGames;
    }

    private DecisionSelectedPgDuelRunner.Resolution resolveInterval(
        Lineage lineage, Interval interval) throws IOException {
      Path candidateSource = interval.directory.resolve("candidate");
      EpsilonDecisionCheckpointManager.save(
          learner.model(),
          candidateSource,
          progress.globalStep,
          lineage.candidateIteration,
          progress.selfPlayGames);
      Path candidate =
          EpsilonDecisionCheckpointManager.saveCandidate(
              checkpointRoot,
              run.candidateRunId(lineage.number, lineage.candidateIteration, interval.duelRound),
              candidateSource);
      interval.candidate = candidate;
      saveProgress();
      DecisionSelectedPgDuelRunner.Resolution resolution =
          DecisionSelectedPgDuelRunner.resolve(
              new DecisionSelectedPgDuelRunner.Request(
                  checkpointRoot,
                  interval.directory,
                  candidate,
                  lineage.immutableParent,
                  lineage.number,
                  lineage.candidateIteration,
                  interval.duelRound,
                  interval.seedBase,
                  progress.completedMacros,
                  budgetEnd,
                  interval.guardFailure,
                  interval.duelSettings),
              executionContext,
              config);
      executionContext.awaitIdle();
      DecisionGpuMemoryDiagnostics.logSnapshot(
          "duel-drained",
          executionContext,
          config.bind(com.epsilon.config.settings.DecisionPerfSettings.class));
      return resolution;
    }

    private DecisionSelectedPgDuelResult toDuelResult(
        Lineage lineage, Interval interval, DecisionSelectedPgDuelRunner.Resolution resolution) {
      return new DecisionSelectedPgDuelResult(
          lineage.number,
          interval.duelRound,
          lineage.candidateIteration,
          progress.completedMacros,
          lineage.completedMacros,
          interval.completedMacros,
          interval.games,
          interval.samples,
          resolution.status(),
          lineage.immutableParent,
          Objects.requireNonNull(interval.candidate, "interval candidate"),
          resolution.summary());
    }

    private void applyResolution(
        Lineage lineage,
        DecisionSelectedPgDuelRunner.Resolution resolution,
        DecisionSelectedPgDuelResult duelResult)
        throws IOException {
      if (resolution.status() == DecisionSelectedPgDuelResult.Status.PROMOTED) {
        Path candidate = duelResult.candidate();
        String candidateId = EpsilonDecisionCheckpointManager.candidateId(candidate);
        EpsilonDecisionWallDuelEvaluator.Result duel =
            resolution.championDuel().orElseThrow().result();
        snapshotPool.addArenaSnapshot(
            new EpsilonDecisionSnapshotPool.SnapshotEntry(
                candidate.toAbsolutePath().normalize().toString(),
                lineage.candidateIteration,
                progress.globalStep,
                false,
                EpsilonDecisionSnapshotPool.EvalStats.fromDuelResult(duel.descriptiveResult()),
                candidateId));
        progress.arenaChampion = candidate;
        snapshotPool.registerArenaChampion(candidate);
        snapshotPool.save(checkpointRoot);
      } else if (!resolution.status().preservesLearnerState()) {
        learner.reloadWorking(lineage.parentLearner);
        DecisionSelectedPgCampaignState parentState =
            DecisionSelectedPgCampaignState.read(lineage.parentLearner);
        macroRunner.restoreExplorationState(parentState.exploration());
        progress.globalStep = lineage.parentManifest.globalStep;
        progress.iteration = lineage.parentManifest.iteration;
        progress.selfPlayGames = lineage.parentManifest.selfPlayGames;
        progress.learnerCheckpoint = workingCheckpoint;
      }
    }

    private void writeDuelReport(Interval interval, DecisionSelectedPgDuelResult duelResult)
        throws IOException {
      DecisionSelectedPgReportWriter.writeDuel(
          checkpointRoot,
          interval.directory,
          settings,
          learnerContractId,
          run.seedBase(),
          run.seedMode(),
          duelResult,
          interval.requireLastActorMetrics(),
          interval.requireLastValueMetrics(),
          interval.lastAudit,
          interval.requireLastLearningRateAdjustment(),
          learner.optimizerReinitializations(),
          learner.optimizerRestorations(),
          elapsedMillis(interval.startedNanos),
          config);
    }

    @Override
    public void close() {
      try (executionContext;
          learner;
          grpSession) {
        executionContext.awaitIdle();
        DecisionGpuMemoryDiagnostics.logSnapshot(
            "session-close",
            executionContext,
            config.bind(com.epsilon.config.settings.DecisionPerfSettings.class));
      }
    }
  }

  private static Path requireArenaChampion(Path champion) throws IOException {
    Path resolved = EpsilonDecisionCheckpointManager.resolveExistingStrict(champion);
    if (resolved == null) {
      throw new IOException("Decision arena campaign has no champion: " + champion);
    }
    return resolved;
  }

  private static long elapsedMillis(long startedNanos) {
    return (System.nanoTime() - startedNanos) / 1_000_000L;
  }

  private static final class Progress {
    private Path arenaChampion;
    private Path learnerCheckpoint;
    private int globalStep;
    private int iteration;
    private int selfPlayGames;
    private int completedMacros;
    private int games;
    private long samples;
    private int lineage;
    private DecisionSelectedPgDuelResult.Status terminalStatus;
    private final ArrayList<DecisionSelectedPgDuelResult> duels = new ArrayList<>();

    private Progress(
        Path arenaChampion,
        Path learnerCheckpoint,
        int globalStep,
        int iteration,
        int selfPlayGames) {
      this.arenaChampion = arenaChampion;
      this.learnerCheckpoint = learnerCheckpoint;
      this.globalStep = globalStep;
      this.iteration = iteration;
      this.selfPlayGames = selfPlayGames;
    }
  }

  private static final class Lineage {
    private final int number;
    private final int candidateIteration;
    private final EpsilonDecisionCheckpointBundle parentManifest;
    private final Path directory;
    private final Path immutableParent;
    private final Path parentLearner;
    private final long seedBase;
    private Path actorReplicaCheckpoint;
    private int completedMacros;
    private int duelRound;

    private Lineage(
        int number,
        int candidateIteration,
        EpsilonDecisionCheckpointBundle parentManifest,
        Path directory,
        Path immutableParent,
        Path parentLearner,
        long seedBase,
        Path actorReplicaCheckpoint) {
      this.number = number;
      this.candidateIteration = candidateIteration;
      this.parentManifest = parentManifest;
      this.directory = directory;
      this.immutableParent = immutableParent;
      this.parentLearner = parentLearner;
      this.seedBase = seedBase;
      this.actorReplicaCheckpoint = actorReplicaCheckpoint;
    }
  }

  private static final class Interval {
    private final int duelRound;
    private final Path directory;
    private final long seedBase;
    private final long[][] opponentIds;
    private final int plannedMacros;
    private final DecisionChampionDuelSettings duelSettings;
    private final long startedNanos;
    private int games;
    private long samples;
    private int completedMacros;
    private Optional<String> guardFailure = Optional.empty();
    private DecisionTrainingResult lastActorMetrics;
    private DecisionTrainingResult lastValueMetrics;
    private Optional<EpsilonDecisionSelectedPgDebugAudit.Result> lastAudit = Optional.empty();
    private DecisionActorLearningControl.Adjustment lastLearningRateAdjustment;
    private Path candidate;

    private Interval(
        int duelRound,
        Path directory,
        long seedBase,
        long[][] opponentIds,
        int plannedMacros,
        DecisionChampionDuelSettings duelSettings,
        long startedNanos) {
      this.duelRound = duelRound;
      this.directory = directory;
      this.seedBase = seedBase;
      this.opponentIds = opponentIds;
      this.plannedMacros = plannedMacros;
      this.duelSettings = duelSettings;
      this.startedNanos = startedNanos;
    }

    private void record(DecisionSelectedPgMacroRunner.Execution execution) {
      games = Math.addExact(games, execution.games());
      samples = Math.addExact(samples, execution.collectedSamples());
      lastActorMetrics = execution.training().actorMetrics();
      lastValueMetrics = execution.training().valueMetrics();
      lastAudit = execution.training().audit();
    }

    private DecisionTrainingResult requireLastActorMetrics() {
      return Objects.requireNonNull(lastActorMetrics, "last Actor metrics");
    }

    private DecisionTrainingResult requireLastValueMetrics() {
      return Objects.requireNonNull(lastValueMetrics, "last Value metrics");
    }

    private DecisionActorLearningControl.Adjustment requireLastLearningRateAdjustment() {
      return Objects.requireNonNull(lastLearningRateAdjustment, "last learning-rate adjustment");
    }
  }
}

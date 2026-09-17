package com.epsilon.nano.ai.decision.training;

import com.epsilon.config.settings.DecisionTrainStreamingSettings;
import com.epsilon.nano.ai.decision.arena.DecisionAdaptiveExploration;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionArena;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionPopulationCollector;
import com.epsilon.nano.ai.decision.audit.EpsilonDecisionSelectedPgDebugAudit;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionFragmentStore;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrainingTargetIdentity;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.nano.ai.decision.runtime.DecisionGpuMemoryDiagnostics;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionTrainFragmentPipeline.Spool;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionTrainFragmentPipeline.StreamingTrainingPlan;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionTrainFragmentPipeline.TrainCollection;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.runtime.DecisionCollectionDiagnostics;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 1回分の自己対局データ収集、学習、更新後の方策検証を実行する。 */
final class DecisionSelectedPgMacroRunner {

  private static final Logger log = LoggerFactory.getLogger(DecisionSelectedPgMacroRunner.class);

  private final DecisionLearner learner;
  private final EpsilonGrpTrainingSession grpSession;
  private final EpsilonDecisionSnapshotPool snapshotPool;
  private final DecisionSelectedPgCampaignSettings settings;
  private final EpsilonDecisionPlayer.RolloutConfig actorRollout;
  private final EpsilonDecisionPlayer.RolloutConfig opponentRollout;
  private final DecisionAdaptiveExploration adaptiveExploration;
  private final int trainSampleLimit;
  private final boolean keepFragments;
  private final DecisionExecutionContext executionContext;

  DecisionSelectedPgMacroRunner(
      DecisionLearner learner,
      EpsilonGrpTrainingSession grpSession,
      EpsilonDecisionSnapshotPool snapshotPool,
      DecisionSelectedPgCampaignSettings settings,
      EpsilonDecisionPlayer.RolloutConfig actorRollout,
      EpsilonDecisionPlayer.RolloutConfig opponentRollout,
      int trainSampleLimit,
      boolean keepFragments,
      DecisionExecutionContext executionContext) {
    this.learner = Objects.requireNonNull(learner, "learner");
    this.grpSession = Objects.requireNonNull(grpSession, "grpSession");
    this.snapshotPool = Objects.requireNonNull(snapshotPool, "snapshotPool");
    this.settings = Objects.requireNonNull(settings, "settings");
    this.actorRollout = Objects.requireNonNull(actorRollout, "actorRollout");
    this.opponentRollout = Objects.requireNonNull(opponentRollout, "opponentRollout");
    adaptiveExploration = new DecisionAdaptiveExploration(actorRollout.fullSupport());
    this.trainSampleLimit = trainSampleLimit;
    this.keepFragments = keepFragments;
    this.executionContext = Objects.requireNonNull(executionContext, "executionContext");
  }

  /** 収集学習データファイルの所有権を最後まで管理し、学習の反復の学習結果と段階別時間を返す。 */
  DecisionAdaptiveExploration.State explorationState() {
    return adaptiveExploration.exportState();
  }

  void restoreExplorationState(DecisionAdaptiveExploration.State state) {
    adaptiveExploration.restoreState(state);
  }

  Execution run(Request request) throws Exception {
    Request actual = Objects.requireNonNull(request, "request");
    TrainCollection collection;
    DecisionAdaptiveExploration.MacroSession explorationSession =
        adaptiveExploration.beginMacro(actual.actorSnapshotId());
    DecisionAdaptiveExploration.Report explorationReport;
    long collectionStartedNanos = System.nanoTime();
    try (AutoCloseable phase = drainPhaseOnClose("collection-drained")) {
      try {
        collection = collect(actual, explorationSession);
        explorationReport = adaptiveExploration.completeMacro(explorationSession);
      } catch (Exception | Error failure) {
        adaptiveExploration.abortMacro(explorationSession);
        if (!keepFragments) {
          discardFailedSpool(actual.spoolDirectory(), failure);
        }
        throw failure;
      }
    }
    long collectionElapsedMillis = elapsedMillis(collectionStartedNanos);

    TrainingResult training;
    long learnerStartedNanos = System.nanoTime();
    try (AutoCloseable fragments =
            () -> {
              if (!keepFragments) collection.discardFragments();
            };
        AutoCloseable phase = drainPhaseOnClose("training-drained")) {
      training = train(collection, actual.actorSnapshotId(), actual.seed());
    }
    return new Execution(
        actual.games(),
        collection.samples(),
        training,
        explorationReport,
        collectionElapsedMillis,
        elapsedMillis(learnerStartedNanos));
  }

  private AutoCloseable drainPhaseOnClose(String phase) {
    return () -> {
      executionContext.awaitIdle();
      DecisionGpuMemoryDiagnostics.logSnapshot(
          phase,
          executionContext,
          learner.config().bind(com.epsilon.config.settings.DecisionPerfSettings.class));
    };
  }

  private TrainCollection collect(
      Request request, DecisionAdaptiveExploration.MacroSession explorationSession)
      throws Exception {
    try (EpsilonDecisionEvaluatorFactory.Handle actorHandle =
            EpsilonDecisionEvaluatorFactory.openTrainingActor(
                learner.model(),
                request.actorReplicaCheckpoint(),
                executionContext,
                learner.config());
        DecisionSnapshotEvaluatorSet population =
            new DecisionSnapshotEvaluatorSet(
                snapshotPool,
                actorHandle.evaluator(),
                request.actorSnapshotId(),
                executionContext,
                learner.config())) {
      log.info(
          "Decision dense selected-PG actor inference: actorSnapshot={} source={} "
              + "checkpoint={} shards={} devices={}",
          request.actorSnapshotId(),
          EpsilonDecisionEvaluatorFactory.replicasEnabled(learner.config())
              ? "CHECKPOINT_MODEL_REPLICAS"
              : "IN_MEMORY_MODEL",
          request.actorReplicaCheckpoint(),
          actorHandle.shards(),
          actorHandle.devices());
      int grpTeacherIteration = grpSession.identity().iteration();
      EpsilonDecisionTrainingTargetIdentity targetIdentity =
          EpsilonDecisionTrainingTargetIdentity.selectedPg(
              actorRollout.causalTraceLambda(), actorRollout.explorationCreditMix());
      EpsilonDecisionFragmentStore.ExpectedIdentity expectedFragmentIdentity =
          new EpsilonDecisionFragmentStore.ExpectedIdentity(
              request.actorSnapshotId(), grpTeacherIteration, targetIdentity);
      try (EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore =
          EpsilonDecisionTrajectoryPayloadStore.fromSettings(
              request.spoolDirectory(),
              true,
              learner
                  .config()
                  .bind(com.epsilon.config.settings.DecisionTrainInFlightSpoolSettings.class))) {
        Spool spoolSink =
            EpsilonDecisionTrainFragmentPipeline.openSpool(
                request.spoolDirectory(),
                expectedFragmentIdentity,
                learner
                    .config()
                    .bind(com.epsilon.config.settings.DecisionTrainSpoolSettings.class));
        try {
          executionContext.resetInferenceMetrics();
          EpsilonDecisionArena.Metrics metrics =
              EpsilonDecisionPopulationCollector.collect(
                  EpsilonDecisionPopulationCollector.Request.standard(
                          request.games(),
                          request.seed(),
                          actorHandle.evaluator(),
                          request.actorSnapshotId(),
                          request.opponentIds(),
                          population,
                          actorRollout,
                          opponentRollout,
                          spoolSink,
                          trajectoryPayloadStore)
                      .withGrpInference(grpSession.inference())
                      .withAdaptiveExploration(explorationSession),
                  learner.config());
          executionContext.awaitInferenceIdle();
          DecisionCollectionDiagnostics.log(executionContext, metrics.batching());
        } catch (Exception | Error collectionFailure) {
          try {
            spoolSink.finish();
          } catch (Exception | Error drainFailure) {
            if (drainFailure != collectionFailure) {
              collectionFailure.addSuppressed(drainFailure);
            }
          }
          try {
            trajectoryPayloadStore.discardPayloadFiles();
          } catch (IOException cleanupFailure) {
            collectionFailure.addSuppressed(cleanupFailure);
          }
          throw collectionFailure;
        }
        return spoolSink.finish().withTrajectoryPayloadFiles(trajectoryPayloadStore.payloadFiles());
      }
    }
  }

  private TrainingResult train(TrainCollection collection, long actorSnapshotId, long seed)
      throws IOException {
    DecisionOnlineTrainingPlan plan =
        new DecisionOnlineTrainingPlan(
            settings.microBatchSize(),
            settings.maximumDeviceTransitionCells(),
            settings.policyUpdateClipRange(),
            settings.explorationCreditMix(),
            settings.entropyCoefficient(),
            new DecisionPolicyTrustRegion(settings.maximumOptimizerShardMeanKl()),
            new DecisionPolicySignalMultipliers(
                settings.dahaiWeight(), settings.riichiWeight(), settings.reactionWeight()),
            settings.ppoEpochs(),
            settings.optimizerStepsPerEpoch(),
            settings.debugActorValidationEnabled());
    StreamingTrainingPlan streamingPlan =
        EpsilonDecisionTrainFragmentPipeline.buildStreamingTrainingPlan(
            collection, actorSnapshotId, trainSampleLimit, seed);
    int trainingSamples = streamingPlan.samples();
    DecisionOnlineTrainingResult fused =
        learner
            .trainer()
            .trainProductionSelectedPolicyAndValueStreaming(
                collection.fragmentPaths(),
                streamingPlan.reader(),
                learner.config().bind(DecisionTrainStreamingSettings.class).prefetch(),
                learner.config().bind(DecisionTrainStreamingSettings.class).prefetchWorkers(),
                trainingSamples,
                plan);
    Optional<EpsilonDecisionSelectedPgDebugAudit.Result> audit = Optional.empty();
    if (fused.rejected()) {
      return TrainingResult.failed(
          trainingSamples,
          fused,
          audit,
          "online update rejected: " + fused.actorMetrics().rejection().reason());
    }
    audit =
        EpsilonDecisionSelectedPgDebugAudit.runIfEnabled(
            settings,
            learner.model(),
            trainingSamples,
            collection.fragmentPaths(),
            streamingPlan.reader(),
            learner.config());
    if (audit.isPresent() && !audit.orElseThrow().passed(settings)) {
      return TrainingResult.failed(
          trainingSamples,
          fused,
          audit,
          "policy final audit rejected: " + audit.orElseThrow().summary());
    }
    return TrainingResult.passed(trainingSamples, fused, audit);
  }

  private static void discardFailedSpool(Path spoolDirectory, Throwable failure) {
    try {
      EpsilonDecisionFragmentStore.discardSpool(spoolDirectory);
    } catch (IOException cleanupFailure) {
      failure.addSuppressed(cleanupFailure);
    }
  }

  private static long elapsedMillis(long startedNanos) {
    return (System.nanoTime() - startedNanos) / 1_000_000L;
  }

  /** 収集に必要な学習の反復固有値。 */
  record Request(
      Path spoolDirectory,
      Path actorReplicaCheckpoint,
      long actorSnapshotId,
      long[][] opponentIds,
      int games,
      long seed) {

    Request {
      spoolDirectory = Objects.requireNonNull(spoolDirectory, "spoolDirectory");
      actorReplicaCheckpoint =
          Objects.requireNonNull(actorReplicaCheckpoint, "actorReplicaCheckpoint");
      opponentIds = Objects.requireNonNull(opponentIds, "opponentIds");
      if (games <= 0) {
        throw new IllegalArgumentException("selected PG macro games must be positive");
      }
    }
  }

  /** 学習の反復収集と学習器の結果。 */
  record Execution(
      int games,
      int collectedSamples,
      TrainingResult training,
      DecisionAdaptiveExploration.Report exploration,
      long collectionElapsedMillis,
      long learnerElapsedMillis) {

    Execution {
      if (games <= 0 || collectedSamples < 0) {
        throw new IllegalArgumentException("selected PG macro collection counters are invalid");
      }
      training = Objects.requireNonNull(training, "training");
      exploration = Objects.requireNonNull(exploration, "exploration");
    }
  }

  /** 学習器更新と必要に応じた最終監査を束ねる。 */
  record TrainingResult(
      boolean passed,
      int trainingSamples,
      DecisionOnlineTrainingResult trainingResult,
      Optional<EpsilonDecisionSelectedPgDebugAudit.Result> audit,
      String reason) {

    TrainingResult {
      if (trainingSamples < 0) {
        throw new IllegalArgumentException("selected PG trainingSamples must be non-negative");
      }
      trainingResult = Objects.requireNonNull(trainingResult, "trainingResult");
      audit = Objects.requireNonNull(audit, "audit");
      reason = Objects.requireNonNull(reason, "reason");
    }

    int optimizerSteps() {
      return trainingResult.optimizerSteps();
    }

    float meanActorUpdateKl() {
      return trainingResult.meanActorUpdateKl();
    }

    DecisionTrainingResult actorMetrics() {
      return trainingResult.actorMetrics();
    }

    DecisionTrainingResult valueMetrics() {
      return trainingResult.valueMetrics();
    }

    private static TrainingResult passed(
        int trainingSamples,
        DecisionOnlineTrainingResult trainingResult,
        Optional<EpsilonDecisionSelectedPgDebugAudit.Result> audit) {
      return new TrainingResult(true, trainingSamples, trainingResult, audit, "passed");
    }

    private static TrainingResult failed(
        int trainingSamples,
        DecisionOnlineTrainingResult trainingResult,
        Optional<EpsilonDecisionSelectedPgDebugAudit.Result> audit,
        String reason) {
      return new TrainingResult(false, trainingSamples, trainingResult, audit, reason);
    }
  }
}

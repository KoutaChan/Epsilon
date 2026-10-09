package com.epsilon.nano.ai.decision.training;

import com.epsilon.config.settings.DecisionTrainInFlightSpoolSettings;
import com.epsilon.config.settings.DecisionTrainSettings;
import com.epsilon.config.settings.DecisionTrainSpoolSettings;
import com.epsilon.config.settings.DecisionTrainStreamingSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.GrpSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.GameState;
import com.epsilon.nano.ai.decision.arena.DecisionAdaptiveExploration;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionPopulationCollector;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionFragmentStore;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrainingTargetIdentity;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionTrainFragmentPipeline.StreamingTrainingPlan;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionTrainFragmentPipeline.TrainCollection;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 本番の非公開 spool・learner 構成を変更せず、測定用 main へ接続するテスト側 adapter。 */
public final class NanoProductionBenchmarkSupport {
  private static final long ACTOR = 2L;
  private static final long OPPONENT = 1L;

  private NanoProductionBenchmarkSupport() {}

  public static Map<String, Object> run(
      Path checkpointRoot,
      Path dataset,
      SettingsLoader config,
      int games,
      int warmupPasses,
      int measuredPasses,
      long seed)
      throws Exception {
    Path checkpoint = EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpointRoot);
    if (checkpoint == null)
      throw new IOException("Decision checkpoint not found: " + checkpointRoot);
    var settings = config.bind(DecisionSelectedPgCampaignSettings.class);
    var resolved = DecisionSelectedPgLearnerContract.resolve(settings, config);
    var devices = config.bind(DeviceSettings.class);
    var learnerDevice = NetworkFactory.getLearnerDevice(devices);
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("checkpointRoot", checkpointRoot.toString());
    report.put("decisionCheckpoint", checkpoint.toString());
    report.put("dataset", dataset.toString());
    report.put("seed", seed);
    report.put("learnerDevices", NetworkFactory.getLearnerDevices(devices).toString());
    report.put("selectedPgSettings", settings);
    report.put("trainingSettings", config.bind(DecisionTrainSettings.class));
    report.put(
        "optimizerState", "Fresh production AdamW; retained across warmup and measured passes.");
    report.put(
        "teacherReuse", "One completed-game dataset and fixed GRP teacher; no recollection.");
    report.put(
        "timingScope",
        "Learner includes descriptor reads, CPU materialization, H2D, forward/backward, optimizer"
            + " updates, production KL probes, training-input drain and device sync. Model loading,"
            + " collection, spool-plan construction, warmup and checkpoint export are excluded. No"
            + " checkpoint export or inference-replica synchronization runs. A logical row is"
            + " counted once per committed fused Actor/Value update, not twice. This is a"
            + " production Nano PPO workload, not a matched CFR objective or episode budget.");
    long setup = System.nanoTime();
    try (var context = new DecisionExecutionContext();
        var learner =
            new DecisionLearner(
                EpsilonDecisionCheckpointManager.load(checkpoint, learnerDevice),
                learnerDevice,
                settings.optimizer(),
                resolved.learnerContractId(),
                context,
                config)) {
      report.put("learnerSetupSeconds", seconds(setup));
      TrainCollection collection =
          collect(
              checkpointRoot, checkpoint, dataset, games, seed, learner, resolved, context, report);
      long planning = System.nanoTime();
      StreamingTrainingPlan streaming =
          EpsilonDecisionTrainFragmentPipeline.buildStreamingTrainingPlan(
              collection, ACTOR, config.bind(DecisionTrainSettings.class).sampleLimit(), seed);
      report.put("planSeconds", seconds(planning));
      report.put("selectedTrainingRows", streaming.samples());
      report.put("fragmentCount", collection.fragmentCount());
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
      List<Pass> warmup = new ArrayList<>();
      List<Pass> measured = new ArrayList<>();
      report.put("warmup", warmup);
      report.put("measured", measured);
      boolean rejected = false;
      for (int index = 0; index < warmupPasses; index++) {
        Pass pass = train(learner, context, collection, streaming, plan);
        warmup.add(pass);
        if (pass.result().rejected()) {
          rejected = true;
          break;
        }
      }
      if (!rejected) {
        for (int index = 0; index < measuredPasses; index++) {
          Pass pass = train(learner, context, collection, streaming, plan);
          measured.add(pass);
          if (pass.result().rejected()) {
            rejected = true;
            break;
          }
        }
      }
      report.put("warmupPassesRequested", warmupPasses);
      report.put("measuredPassesRequested", measuredPasses);
      report.put("complete", !rejected && measured.size() == measuredPasses);
      report.put("rejected", rejected);
      long updated = measured.stream().mapToLong(Pass::logicalUpdatedRows).sum();
      double elapsed = measured.stream().mapToDouble(Pass::elapsedSeconds).sum();
      report.put("logicalUpdatedRows", updated);
      report.put("learnerSeconds", elapsed);
      report.put("logicalUpdatedRowsPerSecond", elapsed > 0 ? updated / elapsed : 0);
    }
    return report;
  }

  private static TrainCollection collect(
      Path checkpointRoot,
      Path checkpoint,
      Path dataset,
      int games,
      long seed,
      DecisionLearner learner,
      DecisionSelectedPgLearnerContract.Resolved resolved,
      DecisionExecutionContext context,
      Map<String, Object> report)
      throws Exception {
    SettingsLoader config = learner.config();
    EpsilonDecisionPopulationCollector.requireSelectedPgActor(resolved.actorRollout());
    long[][] opponentIds = new long[GameState.NUM_PLAYERS][GameState.NUM_PLAYERS - 1];
    for (long[] ids : opponentIds) Arrays.fill(ids, OPPONENT);
    long setup = System.nanoTime();
    try (var actor =
            EpsilonDecisionEvaluatorFactory.openTrainingActor(
                learner.model(), checkpoint, context, config);
        var opponent =
            EpsilonDecisionEvaluatorFactory.openPolicyCheckpointEvaluator(
                checkpoint, context, config);
        var grp =
            config.bind(GrpSettings.class).enabled()
                ? EpsilonGrpTrainingSession.openInferenceOnly(
                    checkpointRoot,
                    NetworkFactory.getGrpDevices(config.bind(DeviceSettings.class)).primary(),
                    config)
                : EpsilonGrpTrainingSession.disabled(checkpointRoot.resolve("grp"));
        var payloads =
            EpsilonDecisionTrajectoryPayloadStore.fromSettings(
                dataset, true, config.bind(DecisionTrainInFlightSpoolSettings.class))) {
      report.put("collectionSetupSeconds", seconds(setup));
      report.put("actorDevices", actor.devices());
      report.put("opponentDevices", opponent.devices());
      report.put("grpTeacherIteration", grp.identity().iteration());
      var identity =
          new EpsilonDecisionFragmentStore.ExpectedIdentity(
              ACTOR,
              grp.identity().iteration(),
              EpsilonDecisionTrainingTargetIdentity.selectedPg(
                  resolved.actorRollout().causalTraceLambda(),
                  resolved.actorRollout().explorationCreditMix()));
      var spool =
          EpsilonDecisionTrainFragmentPipeline.openSpool(
              dataset, identity, config.bind(DecisionTrainSpoolSettings.class));
      var exploration = new DecisionAdaptiveExploration(resolved.actorRollout().fullSupport());
      var explorationSession = exploration.beginMacro(ACTOR);
      long started = System.nanoTime();
      int[] completed = {0};
      boolean collected = false;
      TrainCollection collection;
      try {
        var metrics =
            EpsilonDecisionPopulationCollector.collect(
                EpsilonDecisionPopulationCollector.Request.standard(
                        games,
                        seed,
                        actor.evaluator(),
                        ACTOR,
                        opponentIds,
                        ignored -> opponent.evaluator(),
                        resolved.actorRollout(),
                        resolved.opponentRollout(),
                        (index, gameSeed, game) -> {
                          spool.accept(index, gameSeed, game);
                          completed[0]++;
                        },
                        payloads)
                    .withGrpInference(grp.inference())
                    .withAdaptiveExploration(explorationSession),
                config);
        context.awaitInferenceIdle();
        report.put("arenaMetrics", metrics);
        report.put("requestedNetworkDecisions", metrics.inferenceRequestCount());
        report.put("adaptiveExploration", exploration.completeMacro(explorationSession));
        collected = true;
      } finally {
        if (!collected) exploration.abortMacro(explorationSession);
        collection = spool.finish().withTrajectoryPayloadFiles(payloads.payloadFiles());
      }
      payloads.close();
      report.put("requestedGames", games);
      report.put("completedGames", completed[0]);
      report.put("completedRetainedActorDecisions", collection.samples());
      report.put("collectionSeconds", seconds(started));
      report.put(
          "collectionCounterScope",
          "Retained actor decisions are completed-game training records from the one actor seat."
              + " Requested network decisions include all evaluated seats and branch requests."
              + " All-seat engine decisions, including skipped forced opponent actions, are not"
              + " exposed by this production API and are not inferred from either counter."
              + " Collection is not separately warmed; use the selfplay benchmark for collector"
              + " speed.");
      if (completed[0] != games)
        throw new IllegalStateException("Not all requested games completed");
      return collection;
    }
  }

  private static Pass train(
      DecisionLearner learner,
      DecisionExecutionContext context,
      TrainCollection collection,
      StreamingTrainingPlan streaming,
      DecisionOnlineTrainingPlan plan)
      throws IOException {
    var config = learner.config().bind(DecisionTrainStreamingSettings.class);
    context.awaitIdle();
    long started = System.nanoTime();
    DecisionOnlineTrainingResult result =
        learner
            .trainer()
            .trainProductionSelectedPolicyAndValueStreaming(
                collection.fragmentPaths(),
                streaming.reader(),
                config.prefetch(),
                config.prefetchWorkers(),
                streaming.samples(),
                plan);
    context.awaitIdle();
    double elapsed = seconds(started);
    int steps = result.optimizerSteps();
    int epochSteps = plan.optimizerStepsPerEpoch();
    long rows =
        (long) (steps / epochSteps) * streaming.samples()
            + (long) (steps % epochSteps) * (streaming.samples() / epochSteps)
            + Math.min(steps % epochSteps, streaming.samples() % epochSteps);
    return new Pass(
        elapsed,
        rows,
        rows / elapsed,
        learner.actorLearningRate(),
        learner.valueLearningRate(),
        result);
  }

  private static double seconds(long started) {
    return (System.nanoTime() - started) / 1_000_000_000.0;
  }

  public record Pass(
      double elapsedSeconds,
      long logicalUpdatedRows,
      double logicalUpdatedRowsPerSecond,
      float actorLearningRate,
      float valueLearningRate,
      DecisionOnlineTrainingResult result) {}
}

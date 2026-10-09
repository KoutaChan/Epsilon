package com.epsilon.cfr.training;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrReplayStore;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.io.CfrFiles;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.model.CfrModelConfig;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 蒸留方策を使った初回収集の再開と、実際の後悔学習後の方策切り替えを検証する。 */
@Test(groups = "native")
public class CfrWarmStartLifecycleTest {
  public void nanoTeacherPreservesEveryObservationAcrossSplitBatches() throws Exception {
    Path directory = Files.createTempDirectory("epsilon-cfr-teacher-input-");
    List<DecisionContext> contexts = List.of(context(982), context(100), context(9347));
    var expected = contexts.stream().map(CfrObservationCodec::encodePayload).toList();
    try {
      try (var nano =
          NetworkFactory.createDecisionModel(
              Device.cpu(), false, 16, EpsilonUtilityProfile.TENHOU)) {
        EpsilonDecisionCheckpointManager.saveInitial(nano, directory.resolve("nano"));
      }
      try (var teacher = new NanoTeacherPolicy(directory.resolve("nano"), Device.cpu(), 2)) {
        var predictions = teacher.evaluate(contexts);
        teacher.evaluate(List.of(context(417)));
        for (int row = 0; row < contexts.size(); row++) {
          Assert.assertEquals(predictions.get(row).observationPayload(), expected.get(row));
          Assert.assertEquals(
              predictions.get(row).targetPolicy().length, contexts.get(row).legalActions().size());
          Assert.assertEquals(Arrays.stream(predictions.get(row).targetPolicy()).sum(), 1d, 1e-12);
        }
      }
    } finally {
      delete(directory);
    }
  }

  public void firstCollectionResumesAverageAndNextIterationFreezesLearnedRegret() throws Exception {
    Path directory = Files.createTempDirectory("epsilon-cfr-warm-lifecycle-");
    CfrConfig config = config();
    DecisionContext context = context();
    try {
      CfrRunState initial;
      double[] initialPolicy;
      byte[] completedEpisode;
      Path episodePath;
      try (CfrRunStore store = new CfrRunStore(directory)) {
        config.write(directory.resolve("config.json"));
        var modelConfig =
            new CfrModelConfig(
                16,
                1,
                .01f,
                0,
                0,
                9123,
                1,
                DecisionComputePrecision.FLOAT32,
                DecisionTensorTransfer.DIRECT_BUFFER,
                EpsilonUtilityProfile.TENHOU);
        try (CfrModel average = new CfrModel(modelConfig, CfrModel.Role.AVERAGE, Device.cpu());
            CfrModel regret = new CfrModel(modelConfig, CfrModel.Role.REGRET, Device.cpu())) {
          List<CfrTrainingRow> teacher = List.of(distillationRow(context));
          for (int update = 0; update < 3; update++) average.train(teacher);
          Assert.assertEquals(average.updates(), 3L);
          initialPolicy = average.evaluate(List.of(context)).getFirst().targetPolicy();
          Assert.assertTrue(
              Arrays.stream(initialPolicy).max().orElseThrow()
                      - Arrays.stream(initialPolicy).min().orElseThrow()
                  > 1e-6,
              "The fixture must distinguish average softmax from zero-regret" + " matching");
          for (float score : regret.rawScores(teacher.getFirst().observation()))
            Assert.assertEquals(score, 0f);

          String regretPath = store.saveTraining(regret, "regret", 0, "warm-solve");
          String averagePath = store.saveTraining(average, "average", 0, "warm-solve");
          initial =
              new CfrRunState(
                  2,
                  "warm-solve",
                  0,
                  1,
                  CfrRunState.Phase.COLLECTING,
                  0,
                  0,
                  regretPath,
                  averagePath,
                  averagePath,
                  0,
                  null,
                  "disabled",
                  CfrFiles.sha256(directory.resolve("config.json")));
          store.publish(initial);
          episodePath = store.resolve("solves/warm-solve/matches/1/00000000.bin");
          CfrTrainingTestFiles.writeMatch(episodePath, episode(context, initialPolicy, config));
          completedEpisode = Files.readAllBytes(episodePath);
        }
      }

      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrRunState resumed = store.verify();
        Assert.assertEquals(resumed, initial);
        Assert.assertEquals(resumed.averageUpdate(), 0);
        Assert.assertEquals(resumed.regretUpdate(), 0);
        Assert.assertEquals(resumed.replayGeneration(), 0L);
        Assert.assertEquals(resumed.frozenCheckpoint(), initial.averageCheckpoint());
        try (CfrModel frozen =
            CfrModel.loadModel(store.resolve(resumed.frozenCheckpoint()), Device.cpu())) {
          Assert.assertEquals(frozen.role(), CfrModel.Role.AVERAGE);
          Assert.assertEquals(frozen.updates(), 3L);
          Assert.assertEquals(
              frozen.evaluate(List.of(context)).getFirst().targetPolicy(), initialPolicy);
        }

        CfrRunState sealed = CfrIterationRunner.advance(store, config, resumed);
        Assert.assertEquals(sealed.phase(), CfrRunState.Phase.SEALED);
        Assert.assertEquals(sealed.frozenCheckpoint(), initial.averageCheckpoint());
        Assert.assertEquals(Files.readAllBytes(episodePath), completedEpisode);
        var archived =
            CfrConfig.JSON.fromJson(
                Files.readString(
                    store.resolve("solves/" + initial.solveId() + "/snapshots/1.json")),
                CfrRunStore.CollectedPolicy.class);
        Assert.assertEquals(archived.iteration(), 1L);
        Assert.assertEquals(archived.checkpoint(), initial.averageCheckpoint());
        try (CfrModel reference =
            CfrModel.loadModel(store.resolve(archived.checkpoint()), Device.cpu())) {
          Assert.assertEquals(
              reference.evaluate(List.of(context)).getFirst().targetPolicy(), initialPolicy);
        }

        CfrRunState committed = CfrIterationRunner.advance(store, config, sealed);
        Assert.assertEquals(committed.phase(), CfrRunState.Phase.REPLAY_COMMITTED);
        try (CfrReplayStore replay =
            CfrReplayStore.loadGeneration(
                store.resolve("solves/warm-solve/replay"),
                initial.solveId(),
                committed.replayGeneration())) {
          Assert.assertEquals(replay.stats().episodes(), 1L);
          Assert.assertEquals(replay.stats().seenRows(), 1L);
          var row = replay.regretSampler().sample(1, new SplittableRandom(17)).getFirst();
          Assert.assertEquals(row.id().iteration(), 1L);
          Assert.assertEquals(row.targetPolicy(), initialPolicy);
          Assert.assertEquals(row.utility(), 2d);
        }

        CfrRunState averageTraining = CfrIterationRunner.advance(store, config, committed);
        Assert.assertEquals(averageTraining.phase(), CfrRunState.Phase.TRAINING_AVERAGE);
        CfrRunState trained = CfrIterationRunner.advance(store, config, averageTraining);
        Assert.assertEquals(trained.phase(), CfrRunState.Phase.TRAINED);
        Assert.assertEquals(trained.regretUpdate(), 1);
        Assert.assertEquals(trained.averageUpdate(), 1);
        // 評価には毎回、現在のリプレイ世代から学習した平均方策を渡す。
        // 共通の対戦評価は別のテストで検証し、ここでは評価成功後の段階へ進める。
        Assert.assertTrue(config.evaluatesAt(trained.iteration()));
        CfrRunState ready = trained.nextIteration();
        store.publish(ready);
        Assert.assertEquals(ready.phase(), CfrRunState.Phase.READY);
        Assert.assertEquals(ready.iteration(), 2L);
        Assert.assertNull(ready.frozenCheckpoint());

        CfrRunState collecting = CfrIterationRunner.advance(store, config, ready);
        Assert.assertEquals(collecting.phase(), CfrRunState.Phase.COLLECTING);
        try (CfrModel frozen =
                CfrModel.loadModel(store.resolve(collecting.frozenCheckpoint()), Device.cpu());
            CfrModel regret =
                CfrModel.loadModel(store.resolve(collecting.regretCheckpoint()), Device.cpu());
            CfrModel original =
                CfrModel.loadModel(store.resolve(initial.averageCheckpoint()), Device.cpu())) {
          Assert.assertEquals(frozen.role(), CfrModel.Role.REGRET);
          Assert.assertEquals(frozen.updates(), 1L);
          var observation = CfrObservationCodec.encodePayload(context);
          Assert.assertEquals(frozen.rawScores(observation), regret.rawScores(observation));
          Assert.assertTrue(
              Arrays.stream(toDouble(frozen.rawScores(observation))).anyMatch(score -> score != 0));
          Assert.assertEquals(original.updates(), 3L);
          Assert.assertEquals(
              original.evaluate(List.of(context)).getFirst().targetPolicy(), initialPolicy);
        }
      }
    } finally {
      delete(directory);
    }
  }

  private static CfrTrainingRow distillationRow(DecisionContext context) {
    int[] actions = context.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
    double[] teacher = new double[actions.length];
    Arrays.fill(teacher, .2 / actions.length);
    teacher[0] += .8;
    return new CfrTrainingRow(
        new CfrSampleId(1, context.episodeId(), context.decisionId(), context.seat()),
        actions,
        0,
        teacher,
        0,
        Double.NEGATIVE_INFINITY,
        0,
        CfrObservationCodec.encodePayload(context));
  }

  private static CfrEpisode episode(DecisionContext context, double[] policy, CfrConfig config) {
    double exploration = config.collection().exploration();
    double behavior = (1 - exploration) * policy[0] + exploration / policy.length;
    double[] utility = new double[4];
    utility[context.seat()] = 2;
    return new CfrEpisode(
        context.episodeId(),
        List.of(
            new CfrDecision(
                context.decisionId(),
                context.seat(),
                context.legalActions().stream().mapToInt(action -> action.toIndex()).toArray(),
                0,
                policy,
                behavior,
                CfrObservationCodec.encodePayload(context))),
        utility);
  }

  private static CfrConfig config() {
    return new CfrConfig(
        2345,
        1,
        2,
        CfrConfig.defaults().utilityProfile(),
        new CfrConfig.Collection(
            "MATCH_MC",
            1,
            1,
            1,
            .2,
            false,
            1,
            2,
            4,
            new DecisionDuelArenaSettings(
                Math.min(32, Runtime.getRuntime().availableProcessors()), 2, 64),
            new InferenceBatchingSettings(0),
            new DecisionBranchComparisonSettings(false, false, true, 1, 8, .01)),
        new CfrConfig.Learning(
            1,
            1,
            1,
            1,
            1,
            .01f,
            0,
            0,
            DecisionComputePrecision.FLOAT32,
            DecisionTensorTransfer.DIRECT_BUFFER),
        new CfrConfig.Replay(8, 8, 8L << 20, 1L << 20, 1),
        new CfrConfig.Network(16, new DeviceSettings(true, "cpu", "cpu", "cpu")),
        new CfrConfig.Boundary("", 2, 2),
        new CfrConfig.Evaluation(
            1,
            new DecisionChampionDuelSettings(1, 2, .01, .25),
            new DecisionEvalVsSettings(4, 16, 1, 40000),
            new DecisionDuelArenaSettings(1, 1, 4),
            new InferenceBatchingSettings(0)),
        CfrConfig.defaults().initialization());
  }

  private static DecisionContext context() {
    return context(982);
  }

  private static DecisionContext context(long seed) {
    GameEngine engine = new GameEngine(seed);
    var pending = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var point = pending.decisions().getFirst();
    return new DecisionContext(
        0, point.id(), point.player(), engine.observation(point.player()), point.legalActions());
  }

  private static double[] toDouble(float[] values) {
    double[] result = new double[values.length];
    for (int index = 0; index < values.length; index++) result[index] = values[index];
    return result;
  }

  private static void delete(Path directory) throws IOException {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}

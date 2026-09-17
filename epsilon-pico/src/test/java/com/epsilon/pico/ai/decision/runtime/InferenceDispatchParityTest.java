package com.epsilon.pico.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.DecisionInferenceSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.runtime.InferenceProfile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** モデル混在・複数デバイス・上限を超える入力の分割を、実際の共通dispatch経路で確認する。 */
public class InferenceDispatchParityTest {
  @DataProvider
  public Object[][] gpuModes() {
    return new Object[][] {
      {"gpu:0", "FLOAT32", 0.0002f},
      {"gpu:0,gpu:1", "FLOAT32", 0.0002f},
      {"gpu:0", "BFLOAT16", 0.02f},
      {"gpu:0,gpu:1", "BFLOAT16", 0.02f}
    };
  }

  @Test(groups = "rocm", dataProvider = "gpuModes", timeOut = 120000)
  public void mixedModelsKeepRowOrderThroughOversizedAndTailBatches(
      String devices, String precision, float tolerance) throws Exception {
    Path temporary = Files.createTempDirectory("pico-dispatch-");
    SettingsLoader config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.devices.inference",
                devices,
                "epsilon.decision.inference.maxBatch",
                "8",
                "epsilon.decision.inference.multiTransitionMaxBatch",
                "8",
                "epsilon.decision.inference.computePrecision",
                precision));
    var settings = config.bind(DecisionInferenceSettings.class);
    var fusion = config.bind(DecisionInferenceFusionSettings.class);
    Path firstCheckpoint = temporary.resolve("first");
    Path secondCheckpoint = temporary.resolve("second");
    try {
      saveModel(firstCheckpoint, 1701);
      saveModel(secondCheckpoint, 1702);
      DecisionHostBatch large = input(19, 8181);
      DecisionHostBatch tail = input(3, 9121);
      try (var context = new DecisionExecutionContext();
          var first =
              EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                  firstCheckpoint, context, config);
          var second =
              EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                  secondCheckpoint, context, config);
          var greedy =
              EpsilonDecisionEvaluatorFactory.openGreedyPolicyCheckpointEvaluator(
                  firstCheckpoint, 8, context, config);
          Model cpuFirst =
              EpsilonDecisionCheckpointManager.loadForInference(
                  firstCheckpoint, Device.cpu(), settings);
          Model cpuSecond =
              EpsilonDecisionCheckpointManager.loadForInference(
                  secondCheckpoint, Device.cpu(), settings);
          var referenceFirst =
              EpsilonDecisionInferenceServer.forFrozenModel(cpuFirst, 32, settings, fusion);
          var referenceSecond =
              EpsilonDecisionInferenceServer.forFrozenModel(cpuSecond, 32, settings, fusion)) {
        var a = first.evaluator().submitBatch(large);
        var b = second.evaluator().submitBatch(tail);
        var c = greedy.evaluator().submitGreedyActionSlots(large);
        var d = first.evaluator().submitBatch(tail);
        compare(referenceFirst.evaluateBatch(large), a.join(), tolerance);
        compare(referenceSecond.evaluateBatch(tail), b.join(), tolerance);
        compare(referenceFirst.evaluateBatch(tail), d.join(), tolerance);
        int[] selected = c.join();
        Assert.assertEquals(selected.length, large.size());
        for (int row = 0; row < selected.length; row++) {
          Assert.assertEquals(selected[row], a.join().get(row).greedyActionSlot());
        }
        context.awaitInferenceIdle();
        long physicalRows =
            context.inferenceMetrics().stream()
                .flatMap(device -> device.batches().stream())
                .mapToLong(batch -> batch.rows())
                .sum();
        Assert.assertEquals(physicalRows, 44L);
        Assert.assertTrue(
            context.inferenceMetrics().stream()
                .flatMap(device -> device.batches().stream())
                .allMatch(batch -> batch.rows() <= batch.capacity()));
      }
    } finally {
      try (var paths = Files.walk(temporary)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  @Test(groups = "rocm", timeOut = 120000)
  public void profilingPreservesActorAndOpponentRowsAndReleasesBothGpuPipelines() throws Exception {
    Path temporary = Files.createTempDirectory("pico-profile-parity-");
    SettingsLoader config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.devices.inference", "gpu:0,gpu:1",
                "epsilon.decision.inference.maxBatch", "8",
                "epsilon.decision.inference.multiTransitionMaxBatch", "8",
                "epsilon.decision.inference.computePrecision", "BFLOAT16"));
    Path checkpoint = temporary.resolve("model");
    try {
      saveModel(checkpoint, 1703);
      DecisionHostBatch large = input(7, 8383);
      DecisionHostBatch tail = input(3, 9494);
      var rows = DecisionHostBatch.RowBatch.of(large.sliceRows(0, large.size()));
      var tailRows = DecisionHostBatch.RowBatch.of(tail.sliceRows(0, tail.size()));
      try (var context = new DecisionExecutionContext();
          var actor =
              EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(checkpoint, context, config);
          var opponent =
              EpsilonDecisionEvaluatorFactory.openPolicyCheckpointEvaluator(
                  checkpoint, context, config)) {
        for (var handle : List.of(actor, opponent)) {
          String role = handle == actor ? "actor" : "opponent";
          for (var server : handle.serversForReplayBenchmark()) {
            var expected =
                ((DecisionInferenceResult.Predictions) server.submitProfile(rows, null).join())
                    .values();
            var expectedTail =
                ((DecisionInferenceResult.Predictions) server.submitProfile(tailRows, null).join())
                    .values();
            for (String section : List.of("network.forward", "transfer.h2d", "host.stage")) {
              for (boolean operators : new boolean[] {false, true}) {
                Path trace =
                    temporary.resolve(
                        role
                            + "-"
                            + server.device().getDeviceId()
                            + "-"
                            + section
                            + "-"
                            + operators
                            + ".json");
                var profile = new InferenceProfile(section, trace, server.device(), operators);
                var actual =
                    ((DecisionInferenceResult.Predictions)
                            server.submitProfile(rows, profile).join())
                        .values();
                compare(expected, actual, 0.02f);
                Assert.assertNotNull(profile.result(), section);
                Assert.assertTrue(
                    profile.result().completedWallNanos() >= profile.result().enqueueNanos());
                if (operators && !section.startsWith("host.")) {
                  Assert.assertEquals(profile.result().traceFile(), trace.toString());
                  Assert.assertTrue(Files.isRegularFile(trace));
                } else {
                  Assert.assertNull(profile.result().traceFile());
                  Assert.assertFalse(Files.exists(trace));
                }
              }
            }
            // 計測後も同じ位置を再利用し、短いバッチから元の行数へ戻せる。
            compare(
                expectedTail,
                ((DecisionInferenceResult.Predictions) server.submitProfile(tailRows, null).join())
                    .values(),
                0.02f);
            compare(
                expected,
                ((DecisionInferenceResult.Predictions) server.submitProfile(rows, null).join())
                    .values(),
                0.02f);
          }
        }
        context.awaitInferenceIdle();
        Assert.assertEquals(
            context.inferenceMetrics().stream()
                .flatMap(device -> device.batches().stream())
                .mapToLong(batch -> batch.rows())
                .sum(),
            248L);
      }
    } finally {
      try (var paths = Files.walk(temporary)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  @Test(groups = "rocm", timeOut = 120000)
  public void compactChiTransitionsMatchDenseInputThroughEagerAndRing() throws Exception {
    Path temporary = Files.createTempDirectory("pico-compact-dispatch-");
    SettingsLoader config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.devices.inference", "gpu:0",
                "epsilon.decision.inference.maxBatch", "4",
                "epsilon.decision.inference.multiTransitionMaxBatch", "4",
                "epsilon.decision.inference.computePrecision", "BFLOAT16"));
    Path checkpoint = temporary.resolve("model");
    try {
      saveModel(checkpoint, 1704);
      DecisionHostBatch compact = chiInput(7);
      Assert.assertTrue(compact.actionTransitionCount(0, 0) > 1);
      DecisionHostBatch dense = compact.copyRows(0, compact.size());
      var settings = config.bind(DecisionInferenceSettings.class);
      try (var context = new DecisionExecutionContext();
          var actor =
              EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(checkpoint, context, config);
          Model model =
              EpsilonDecisionCheckpointManager.loadForInference(
                  checkpoint, Device.gpu(0), settings);
          var eager =
              EpsilonDecisionInferenceServer.forFrozenModel(
                  model, 16, settings, DecisionInferenceFusionSettings.eager())) {
        var expected = eager.evaluateBatch(dense);
        compare(expected, eager.evaluateBatch(compact), 0.02f);
        var denseResult = actor.evaluator().submitBatch(dense);
        var compactResult = actor.evaluator().submitBatch(compact);
        compare(expected, denseResult.join(), 0.02f);
        compare(expected, compactResult.join(), 0.02f);
        context.awaitInferenceIdle();
      }
    } finally {
      try (var paths = Files.walk(temporary)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  private static DecisionHostBatch chiInput(int rows) {
    GameState state = new GameState(17);
    state.startRound(0, 0, 0, 0);
    state.setCurrentPlayer(1);
    state.initializeWallForReconstruction(new int[] {Tile.M4}, 52);
    state.commitDiscard(1, Tile.TON, false);
    state.hand(1).clear();
    for (int tile :
        new int[] {
          Tile.M3, Tile.M5, Tile.M5, Tile.M5, Tile.M6, Tile.M6, Tile.P1, Tile.P2, Tile.P3, Tile.S1,
          Tile.S2, Tile.S3, Tile.HAKU
        }) state.hand(1).add(tile);
    state.commitDiscard(0, Tile.M4, false);
    var actions = List.of(Action.chiSequence(Tile.M3, Tile.M4, false), Action.pass());
    var builder = DecisionBatchBuilder.inference(rows, new DecisionBucket(16, 16));
    for (int row = 0; row < rows; row++) {
      builder.addDetachedInferenceRow(
          state, 1, actions, state.publicState(), DecisionBoundaryContext.uniform());
    }
    return builder.build();
  }

  private static void saveModel(Path checkpoint, int seed) throws Exception {
    Engine.getEngine("PyTorch").setRandomSeed(seed);
    try (Model model =
        NetworkFactory.createDecisionModel(Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
      EpsilonDecisionCheckpointManager.saveInitial(model, checkpoint);
    }
  }

  private static DecisionHostBatch input(int rows, long seed) {
    var builder = DecisionBatchBuilder.inference(rows, DecisionBucket.forInferenceCounts(16, 1));
    for (int row = 0; row < rows; row++) {
      var engine = new GameEngine(seed + row);
      var boundary = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
      builder.addInferenceRow(
          engine, boundary.decisions().getFirst(), DecisionBoundaryContext.uniform());
    }
    return builder.build();
  }

  private static void compare(
      List<EpsilonDecisionInferenceServer.Prediction> expected,
      List<EpsilonDecisionInferenceServer.Prediction> actual,
      float tolerance) {
    Assert.assertEquals(actual.size(), expected.size());
    for (int row = 0; row < actual.size(); row++) {
      float[] want = expected.get(row).policyProbabilities();
      float[] got = actual.get(row).policyProbabilities();
      Assert.assertEquals(got.length, want.length);
      for (int action = 0; action < want.length; action++)
        Assert.assertEquals(
            got[action], want[action], tolerance, "row=" + row + " action=" + action);
      Assert.assertEquals(
          actual.get(row).valueUtility(), expected.get(row).valueUtility(), tolerance);
    }
  }
}

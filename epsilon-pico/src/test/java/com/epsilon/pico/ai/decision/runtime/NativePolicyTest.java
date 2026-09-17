package com.epsilon.pico.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.pico.PicoModelProvider;
import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import com.epsilon.spi.PolicyExecutionContext;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 復元した方策の入力変換とセッションごとの設定、および複数方策による GPU 実行環境の共有を検証する。 */
public class NativePolicyTest {
  @Test(groups = "native")
  public void restoredPolicyMatchesEngineEncodingAndKeepsSettingsPerSession() throws Exception {
    Engine.getEngine("PyTorch").setRandomSeed(117);
    Path temporary = Files.createTempDirectory("epsilon-pico-policy-");
    Path checkpoint = temporary.resolve("model");
    try (Model source =
        NetworkFactory.createDecisionModel(Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
      EpsilonDecisionCheckpointManager.saveInitial(source, checkpoint);
      GameEngine engine = new GameEngine(8181L);
      var boundary = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
      EngineDecisionPoint point = boundary.decisions().get(0);
      DecisionRequest request =
          new DecisionRequest(
              point.id(),
              point.player(),
              point.kind(),
              engine.observation(point.player()),
              point.legalActions());
      var bucket = DecisionBatchBuilder.selectInferenceBucket(point.legalActions());
      var builder = DecisionBatchBuilder.inference(1, bucket);
      builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
      int[] expected;
      try (var direct =
          EpsilonDecisionInferenceServer.forPolicySession(
              source,
              com.epsilon.pico.config.settings.DecisionInferenceSettings.defaults(),
              EpsilonSettings.defaults().bind(DecisionInferenceFusionSettings.class))) {
        expected = direct.evaluateGreedyActionSlots(builder.build());
      }
      var provider = new PicoModelProvider();
      Path policySettings = temporary.resolve("policy.toml");
      Files.writeString(policySettings, "[epsilon.devices]\ninference = \"cpu\"\n");
      var configured = new java.util.LinkedHashMap<>(options(8));
      configured.remove("device");
      configured.put("settings", policySettings.toString());
      try (var execution = new PolicyExecutionContext(2, 1, 4);
          BatchedPolicy narrow = provider.open(checkpoint, options(4), execution);
          BatchedPolicy wide = provider.open(checkpoint, configured, execution)) {
        Object narrowKey = narrow.batchKey(request);
        Assert.assertEquals(narrow.maxBatchSize(narrowKey), 4);
        Assert.assertEquals(wide.maxBatchSize(wide.batchKey(request)), 8);
        CountDownLatch available = new CountDownLatch(1);
        ArrayList<BatchedPolicy.Ingress> held = new ArrayList<>();
        try {
          for (int index = 0; index < 4; index++) {
            BatchedPolicy policy = index % 2 == 0 ? narrow : wide;
            var reservation = policy.tryAcquire(policy.batchKey(request), () -> {});
            Assert.assertNotNull(reservation);
            held.add(reservation);
          }
          Assert.assertNull(narrow.tryAcquire(narrowKey, () -> {}));
          Assert.assertNull(wide.tryAcquire(wide.batchKey(request), available::countDown));
          held.removeLast().close();
          Assert.assertTrue(available.await(10, TimeUnit.SECONDS));
          try (var resumed = narrow.tryAcquire(narrowKey, () -> {})) {
            Assert.assertNotNull(resumed);
          }
        } finally {
          for (BatchedPolicy.Ingress reservation : held) reservation.close();
        }
        try (var reservation = narrow.tryAcquire(narrowKey, () -> {})) {
          Assert.assertEquals(reservation.submit(List.of(request)).join(), expected);
        }
        try (var reservation = wide.tryAcquire(wide.batchKey(request), () -> {})) {
          Assert.assertEquals(
              reservation.submit(List.of(request, request)).join(),
              new int[] {expected[0], expected[0]});
        }
        Assert.assertEquals(narrow.maxBatchSize(narrowKey), 4);
        Assert.assertEquals(narrow.timings().batches(), 1L);
        Assert.assertEquals(wide.timings().batches(), 1L);
        Assert.assertTrue(wide.timings().encodingNanos() > 0);
        Assert.assertTrue(wide.timings().inferenceRoundTripNanos() > 0);
      }
      cancelledBatchStillDrains(provider, checkpoint, request);
    } finally {
      try (var paths = Files.walk(temporary)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  @Test(groups = "rocm")
  public void independentPoliciesShareOneGpuExecutionContext() throws Exception {
    Path temporary = Files.createTempDirectory("epsilon-pico-shared-gpu-");
    Path checkpoint = temporary.resolve("model");
    try {
      try (Model source =
          NetworkFactory.createDecisionModel(Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
        EpsilonDecisionCheckpointManager.saveInitial(source, checkpoint);
      }
      var provider = new PicoModelProvider();
      Map<String, String> gpu =
          Map.of(
              "device",
              "gpu:0",
              "epsilon.decision.inference.maxBatch",
              "4",
              "epsilon.decision.inference.multiTransitionMaxBatch",
              "4",
              "epsilon.decision.inference.computePrecision",
              "FLOAT32");
      var otherModelSettings = new java.util.LinkedHashMap<>(gpu);
      otherModelSettings.put("epsilon.decision.inference.slotsPerDevice", "4");
      otherModelSettings.put("epsilon.decision.inference.readyBatchesPerDevice", "8");
      try (var execution = new PolicyExecutionContext();
          BatchedPolicy first = provider.open(checkpoint, gpu, execution);
          BatchedPolicy second = provider.open(checkpoint, otherModelSettings, execution)) {
        GameEngine engine = new GameEngine(8109L);
        var boundary = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
        EngineDecisionPoint point = boundary.decisions().get(0);
        DecisionRequest request =
            new DecisionRequest(
                point.id(),
                point.player(),
                point.kind(),
                engine.observation(point.player()),
                point.legalActions());
        try (var firstBatch = first.tryAcquire(first.batchKey(request), () -> {});
            var secondBatch = second.tryAcquire(second.batchKey(request), () -> {})) {
          Assert.assertNotNull(firstBatch);
          Assert.assertNotNull(secondBatch);
          var a = firstBatch.submit(List.of(request));
          var b = secondBatch.submit(List.of(request));
          Assert.assertEquals(a.join(), b.join());
          Assert.assertTrue(a.join()[0] >= 0 && a.join()[0] < point.legalActions().size());
          try (var isolatedExecution = new PolicyExecutionContext(2, 1, 2);
              var isolated = provider.open(checkpoint, gpu, isolatedExecution);
              var isolatedBatch = isolated.tryAcquire(isolated.batchKey(request), () -> {})) {
            Assert.assertEquals(isolatedBatch.submit(List.of(request)).join(), a.join());
          }
        }
      }
    } finally {
      try (var paths = Files.walk(temporary)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  private static void cancelledBatchStillDrains(
      com.epsilon.spi.ModelProvider provider, Path checkpoint, DecisionRequest request)
      throws Exception {
    var started = new java.util.concurrent.CountDownLatch(2);
    var release = new java.util.concurrent.CountDownLatch(1);
    try (var execution = new PolicyExecutionContext(2);
        var policy = provider.open(checkpoint, options(128), execution)) {
      var occupied =
          execution.encodeRows(
              128,
              (from, to) -> {
                started.countDown();
                try {
                  release.await();
                } catch (InterruptedException e) {
                  throw new RuntimeException(e);
                }
              });
      try {
        Assert.assertTrue(started.await(10, java.util.concurrent.TimeUnit.SECONDS));
        try (var ingress = policy.tryAcquire(policy.batchKey(request), () -> {})) {
          var result = ingress.submit(java.util.Collections.nCopies(128, request));
          Assert.assertTrue(result.cancel(false));
        }
      } finally {
        release.countDown();
      }
      occupied.join();
      java.util.concurrent.CompletableFuture.runAsync(policy::close)
          .get(30, java.util.concurrent.TimeUnit.SECONDS);
    }
  }

  private static Map<String, String> options(int maximumBatch) {
    return Map.of(
        "device",
        "cpu",
        "epsilon.decision.inference.maxBatch",
        Integer.toString(maximumBatch),
        "epsilon.decision.inference.multiTransitionMaxBatch",
        Integer.toString(maximumBatch));
  }
}

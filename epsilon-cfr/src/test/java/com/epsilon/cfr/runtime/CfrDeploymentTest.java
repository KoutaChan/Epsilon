package com.epsilon.cfr.runtime;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.model.CfrModelConfig;
import com.epsilon.cfr.training.CfrRunStore;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.network.NetworkDevices;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import com.epsilon.spi.PolicyExecutionContext;
import com.google.gson.JsonObject;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class CfrDeploymentTest {
  @DataProvider
  public Object[][] trainingModes() {
    return new Object[][] {{"HAND_GRP"}, {"MATCH_MC"}};
  }

  @Test(groups = "native", dataProvider = "trainingModes")
  public void exportedAverageRunsThroughReviewAndArenaWithBoardInput(String mode) throws Exception {
    var directory = Files.createTempDirectory("cfr-deployment");
    GameEngine engine = new GameEngine(811L);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    var context =
        new DecisionContext(
            11,
            point.id(),
            point.player(),
            engine.observation(point.player()),
            point.legalActions());
    try (CfrModel original =
        new CfrModel(
            new CfrModelConfig(
                8,
                2,
                1e-3f,
                0,
                0,
                91,
                2,
                DecisionComputePrecision.FLOAT32,
                DecisionTensorTransfer.DIRECT_BUFFER,
                EpsilonUtilityProfile.PLACEMENT),
            CfrModel.Role.AVERAGE,
            Device.cpu())) {
      original.saveModel(directory.resolve("model"));
      writeManifest(directory, EpsilonUtilityProfile.PLACEMENT, mode);
      CfrRunStore.sealArtifact(directory);
      try (var deployment = CfrDeployment.open(directory, Map.of())) {
        Assert.assertEquals(
            deployment.manifest().utilityProfile(), EpsilonUtilityProfile.PLACEMENT);
        Assert.assertEquals(
            deployment.model.config().utilityProfile(), EpsilonUtilityProfile.PLACEMENT);
      }
      double[] expected = original.evaluate(List.of(context)).getFirst().targetPolicy();
      try (var review =
          new CfrReviewPolicyProvider().openReview(directory, Map.of("device", "cpu"))) {
        float[] actual = review.probabilities(context.observation(), context.legalActions());
        for (int slot = 0; slot < actual.length; slot++)
          Assert.assertEquals(actual[slot], expected[slot], 1e-6);
      }
      try (PolicyExecutionContext execution = new PolicyExecutionContext(1);
          var arena = new CfrModelProvider().open(directory, Map.of("device", "cpu"), execution);
          var second = new CfrModelProvider().open(directory, Map.of("device", "cpu"), execution)) {
        var request =
            new DecisionRequest(
                11,
                point.id(),
                point.player(),
                point.kind(),
                context.observation(),
                context.legalActions());
        var ingress = arena.tryAcquire(arena.batchKey(request), () -> {});
        Assert.assertNotNull(ingress);
        int[] selected = ingress.submit(List.of(request)).join();
        Assert.assertEquals(selected.length, 1);
        Assert.assertTrue(selected[0] >= 0 && selected[0] < point.legalActions().size());
        ingress.close();
        try (var repeated = second.tryAcquire(second.batchKey(request), () -> {})) {
          Assert.assertEquals(repeated.submit(List.of(request)).join(), selected);
        }
        Assert.assertEquals(
            execution.devices().inferenceDispatchMetrics().hostCapacity(),
            4,
            "Multiple CFR models share the common device capacity");
        var reservations = new ArrayList<BatchedPolicy.Ingress>();
        var available = new CompletableFuture<Void>();
        try {
          for (int i = 0; i < 4; i++) {
            var owner = i % 2 == 0 ? arena : second;
            var reserved = owner.tryAcquire(owner.batchKey(request), () -> {});
            Assert.assertNotNull(reserved);
            reservations.add(reserved);
          }
          Assert.assertNull(
              second.tryAcquire(second.batchKey(request), () -> available.complete(null)));
          reservations.removeFirst().close();
          available.get(10, TimeUnit.SECONDS);
          var replacement = second.tryAcquire(second.batchKey(request), () -> {});
          Assert.assertNotNull(replacement);
          reservations.add(replacement);
        } finally {
          for (var reserved : reservations) reserved.close();
        }
      }
      try (PolicyExecutionContext execution = new PolicyExecutionContext(1)) {
        for (int repetition = 0; repetition < 2; repetition++) {
          try (var fleet =
              new CfrPolicyFleet(original, NetworkDevices.of(Device.cpu()), execution)) {
            double[] actual = fleet.evaluate(List.of(context)).getFirst().targetPolicy();
            Assert.assertEquals(actual, expected);
          }
          Assert.assertEquals(
              execution.devices().inferenceDispatchMetrics().hostCapacity(),
              0,
              "Closing a fleet releases its registration and leaves the borrowed context open");
        }
      }
      writeManifest(directory, EpsilonUtilityProfile.TENHOU);
      CfrRunStore.sealArtifact(directory);
      var mismatch =
          Assert.expectThrows(
              UncheckedIOException.class, () -> CfrDeployment.open(directory, Map.of()));
      Assert.assertTrue(mismatch.getCause().getMessage().contains("utility profile mismatch"));
      writeManifest(directory, EpsilonUtilityProfile.PLACEMENT);
      CfrRunStore.sealArtifact(directory);
      Files.writeString(directory.resolve("model/model.json"), "tampered");
      Assert.expectThrows(
          UncheckedIOException.class,
          () -> new CfrReviewPolicyProvider().openReview(directory, Map.of()));
    } finally {
      try (var paths = Files.walk(directory)) {
        for (var path : paths.sorted(Comparator.reverseOrder()).toList())
          Files.deleteIfExists(path);
      }
    }
  }

  @DataProvider
  public Object[][] invalidUtilityProfiles() {
    return new Object[][] {{null}, {"UNKNOWN"}, {"SCORE"}};
  }

  @Test(dataProvider = "invalidUtilityProfiles")
  public void deploymentRequiresSupportedUtilityProfile(String profile) throws Exception {
    Path directory = Files.createTempDirectory("cfr-deployment-utility-");
    try {
      writeManifest(directory, EpsilonUtilityProfile.TENHOU);
      var manifest =
          CfrConfig.JSON.fromJson(
              Files.readString(directory.resolve("deployment.json")), JsonObject.class);
      if (profile == null) manifest.remove("utilityProfile");
      else manifest.addProperty("utilityProfile", profile);
      Files.writeString(directory.resolve("deployment.json"), CfrConfig.JSON.toJson(manifest));
      CfrRunStore.sealArtifact(directory);
      var invalid =
          Assert.expectThrows(
              UncheckedIOException.class, () -> CfrDeployment.open(directory, Map.of()));
      Assert.assertEquals(invalid.getCause().getMessage(), "Invalid CFR deployment manifest");
    } finally {
      try (var paths = Files.walk(directory)) {
        for (var path : paths.sorted(Comparator.reverseOrder()).toList())
          Files.deleteIfExists(path);
      }
    }
  }

  private static void writeManifest(Path directory, EpsilonUtilityProfile profile)
      throws Exception {
    writeManifest(directory, profile, "HAND_GRP");
  }

  private static void writeManifest(Path directory, EpsilonUtilityProfile profile, String mode)
      throws Exception {
    Files.writeString(
        directory.resolve("deployment.json"),
        CfrConfig.JSON.toJson(
            new CfrDeployment.Manifest(
                2,
                "epsilon-cfr",
                "AVERAGE_DEPLOYMENT",
                mode,
                "model",
                "test",
                1,
                "policy",
                profile)));
  }
}

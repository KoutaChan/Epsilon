package com.epsilon.cfr.config;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.config.settings.EpsilonSettings;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.SettingsLoader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrConfigTest {
  @Test
  public void configurationAndDeviceAssignmentsRoundTrip() throws Exception {
    Path file = Files.createTempFile("cfr-config-", ".json");
    try {
      CfrConfig config =
          CfrConfig.load(
              EpsilonSettings.of(
                  Map.of(
                      "epsilon.cfr.network.devices.learner", "gpu:0, gpu:1",
                      "epsilon.cfr.network.devices.inference", "gpu:1",
                      "epsilon.cfr.network.devices.grp", "cpu",
                      "epsilon.cfr.utilityProfile", "PLACEMENT")));
      config.write(file);
      CfrConfig restored = CfrConfig.read(file);
      Assert.assertEquals(CfrConfig.JSON.toJson(restored), CfrConfig.JSON.toJson(config));
      Assert.assertEquals(restored.network().devices().learner(), "gpu:0, gpu:1");
      Assert.assertEquals(restored.network().devices().inference(), "gpu:1");
      Assert.assertEquals(restored.network().devices().grp(), "cpu");
      Assert.assertEquals(restored.utilityProfile(), EpsilonUtilityProfile.PLACEMENT);
      Assert.assertEquals(
          restored
              .withInitialization(new CfrConfig.Initialization("nano", 2, 1, 8))
              .utilityProfile(),
          EpsilonUtilityProfile.PLACEMENT);
    } finally {
      Files.delete(file);
    }
  }

  @Test
  public void unknownOrMissingSettingsAreRejected() throws Exception {
    Path file = Files.createTempFile("cfr-config-invalid-", ".json");
    try {
      var json = CfrConfig.JSON.toJsonTree(CfrConfig.defaults()).getAsJsonObject();
      json.addProperty("legacyPpoClip", .2);
      Files.writeString(file, json.toString());
      Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
      json = CfrConfig.JSON.toJsonTree(CfrConfig.defaults()).getAsJsonObject();
      json.getAsJsonObject("network").getAsJsonObject("devices").remove("validateIndices");
      Files.writeString(file, json.toString());
      Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
      json = CfrConfig.JSON.toJsonTree(CfrConfig.defaults()).getAsJsonObject();
      json.getAsJsonObject("learning").remove("averageUpdates");
      Files.writeString(file, json.toString());
      Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
      json = CfrConfig.JSON.toJsonTree(CfrConfig.defaults()).getAsJsonObject();
      json.getAsJsonObject("collection").remove("branchComparison");
      Files.writeString(file, json.toString());
      Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
      json = CfrConfig.JSON.toJsonTree(CfrConfig.defaults()).getAsJsonObject();
      json.getAsJsonObject("collection").add("branchComparison", null);
      Files.writeString(file, json.toString());
      Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
    } finally {
      Files.delete(file);
    }
  }

  @Test
  public void savedUtilityMustBePresentAndRankBased() throws Exception {
    Path file = Files.createTempFile("cfr-utility-", ".json");
    try {
      for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
        if (!profile.rankBased()) continue;
        CfrConfig config =
            CfrConfig.load(
                EpsilonSettings.of(Map.of("epsilon.cfr.utilityProfile", profile.name())));
        config.write(file);
        Assert.assertEquals(CfrConfig.read(file).utilityProfile(), profile);
      }
      var json = CfrConfig.JSON.toJsonTree(CfrConfig.defaults()).getAsJsonObject();
      json.remove("utilityProfile");
      Files.writeString(file, json.toString());
      Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
      for (String invalid : new String[] {null, "UNKNOWN", "SCORE"}) {
        json.addProperty("utilityProfile", invalid);
        Files.writeString(file, json.toString());
        Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
      }
    } finally {
      Files.delete(file);
    }
  }

  @Test
  public void annotatedRangesRejectInvalidInputsBeforeTraining() {
    for (var setting :
        List.of(
            Map.entry("epsilon.cfr.solves", "0"),
            Map.entry("epsilon.cfr.iterations", "-1"),
            Map.entry("epsilon.cfr.utilityProfile", "UNKNOWN"),
            Map.entry("epsilon.cfr.utilityProfile", "SCORE"),
            Map.entry("epsilon.cfr.collection.games", "0"),
            Map.entry("epsilon.cfr.collection.exploration", "0"),
            Map.entry("epsilon.cfr.collection.exploration", "1.01"),
            Map.entry("epsilon.cfr.collection.exploration", "NaN"),
            Map.entry("epsilon.cfr.collection.branchComparison.maxBranchesPerGame", "0"),
            Map.entry("epsilon.cfr.collection.branchComparison.maxInFlightBranches", "0"),
            Map.entry("epsilon.cfr.collection.branchComparison.maxExtraInferenceRatio", "0"),
            Map.entry("epsilon.cfr.collection.branchComparison.maxExtraInferenceRatio", "1.01"),
            Map.entry("epsilon.cfr.collection.branchComparison.maxExtraInferenceRatio", "NaN"),
            Map.entry("epsilon.cfr.learning.learningRate", "Infinity"),
            Map.entry("epsilon.cfr.learning.weightDecay", "-0.1"),
            Map.entry("epsilon.cfr.replay.maxRowBytes", "1023"),
            Map.entry("epsilon.cfr.network.hidden", "18"),
            Map.entry("epsilon.cfr.network.hidden", "12"),
            Map.entry("epsilon.cfr.evaluation.championDuel.alpha", "0"),
            Map.entry("epsilon.cfr.evaluation.championDuel.interimWallSeeds", "0"),
            Map.entry("epsilon.cfr.learning.gradientClip", "-1"),
            Map.entry("epsilon.cfr.learning.computePrecision", "INVALID"),
            Map.entry("epsilon.cfr.boundary.bootstrapGames", "1"),
            Map.entry("epsilon.cfr.boundary.calibrationGames", "1"),
            Map.entry("epsilon.cfr.initialization.games", "1"))) {
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> CfrConfig.load(EpsilonSettings.of(Map.of(setting.getKey(), setting.getValue()))));
    }
  }

  @Test
  public void optionalNanoInitializationRoundTripsAndRejectsInvalidSettings() throws Exception {
    Path file = Files.createTempFile("cfr-config-nano-", ".json");
    try {
      CfrConfig previous = CfrConfig.defaults();
      previous.write(file);
      Assert.assertTrue(Files.readString(file).contains("initialization"));
      Assert.assertFalse(CfrConfig.read(file).initialization().enabled());
      CfrConfig configured =
          previous.withInitialization(new CfrConfig.Initialization("nano", 4, 8, 512));
      configured.write(file);
      Assert.assertEquals(CfrConfig.read(file).initialization(), configured.initialization());
      var json = CfrConfig.JSON.toJsonTree(configured).getAsJsonObject();
      json.getAsJsonObject("initialization").remove("updates");
      Files.writeString(file, json.toString());
      Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
      Assert.expectThrows(
          IllegalArgumentException.class, () -> new CfrConfig.Initialization("nano", 1, 8, 512));
      Assert.expectThrows(
          IllegalArgumentException.class, () -> new CfrConfig.Initialization(null, 4, 8, 512));
    } finally {
      Files.delete(file);
    }
  }

  @Test
  public void bundledDefaultsUseNanoRuntimeAndSupervisedLearningValues() {
    CfrConfig bundled = CfrConfig.defaults();
    CfrConfig annotated =
        CfrConfig.load(
            SettingsLoader.of(
                Map.of(
                    "epsilon.cfr.network.devices.grp", "cpu",
                    "epsilon.cfr.collection.branchComparison.enabled", "true",
                    "epsilon.cfr.collection.branchComparison.winDecline", "true")));
    Assert.assertEquals(CfrConfig.JSON.toJson(bundled), CfrConfig.JSON.toJson(annotated));
    Assert.assertEquals(bundled.seed(), 20261007L);
    Assert.assertEquals(bundled.utilityProfile(), EpsilonUtilityProfile.TENHOU);
    Assert.assertEquals(bundled.collection().games(), 32768);
    Assert.assertEquals(bundled.learning().batchSize(), 1024);
    Assert.assertEquals(bundled.learning().microBatchSize(), 1024);
    Assert.assertEquals(bundled.learning().learningRate(), 2e-4f);
    Assert.assertEquals(bundled.learning().weightDecay(), 1e-4f);
    Assert.assertEquals(bundled.learning().gradientClip(), .5f);
    Assert.assertEquals(bundled.learning().computePrecision(), DecisionComputePrecision.BFLOAT16);
    Assert.assertEquals(bundled.learning().tensorTransfer(), DecisionTensorTransfer.PINNED_BUFFER);
    Assert.assertEquals(bundled.collection().concurrentGames(), 4096);
    Assert.assertEquals(bundled.collection().inferenceBatch(), 4096);
    Assert.assertEquals(bundled.collection().multiTransitionMaxBatch(), 2048);
    Assert.assertEquals(bundled.collection().arena().advanceWorkers(), 32);
    var branches = bundled.collection().branchComparison();
    Assert.assertTrue(branches.enabled());
    Assert.assertTrue(branches.winDecline());
    Assert.assertTrue(branches.kyushu());
    Assert.assertEquals(branches.maxBranchesPerGame(), 1);
    Assert.assertEquals(branches.maxInFlightBranches(), 8);
    Assert.assertEquals(branches.maxExtraInferenceRatio(), .01);
    Assert.assertEquals(bundled.replay().maxStoredBytes(), 128L << 30);
    Assert.assertEquals(bundled.network().devices().learner(), "");
    Assert.assertEquals(bundled.network().devices().inference(), "");
    Assert.assertEquals(bundled.network().devices().grp(), "cpu");
    Assert.assertEquals(bundled.network().hidden(), 256);
    Assert.assertEquals(bundled.evaluation().every(), 10);
    Assert.assertEquals(bundled.evaluation().evalVs().gamesInFlight(), 32768);
    Assert.assertEquals(bundled.evaluation().championDuel().maximumWallSeeds(), 100000);
    Assert.assertEquals(bundled.boundary().checkpoint(), "");
    Assert.assertFalse(bundled.initialization().enabled());
  }

  @Test
  public void partialTomlAndOverridesUseCommonPrecedenceAndExactLongValues() throws Exception {
    Path file = Files.createTempFile("cfr-settings-", ".toml");
    try {
      Files.writeString(
          file,
          """
          [epsilon.cfr]
          seed = 9_007_199_254_740_993
          utilityProfile = "LAST_AVOIDANCE"
          [epsilon.cfr.learning]
          batchSize = 32
          microBatchSize = 8
          [epsilon.cfr.collection.branchComparison]
          enabled = false
          maxExtraInferenceRatio = 0.02
          [epsilon.cfr.network]
          hidden = 256
          [epsilon.cfr.network.devices]
          learner = "gpu:0, gpu:1"
          inference = "gpu:1"
          [epsilon.cfr.replay]
          maxStoredBytes = 137_438_953_473
          [epsilon.cfr.boundary]
          checkpoint = "model/grp"
          [epsilon.cfr.initialization]
          nanoCheckpoint = "model/nano"
          updates = 8
          """);
      CfrConfig config =
          CfrConfig.load(
              EpsilonSettings.load(
                  file,
                  Map.of(
                      "epsilon.cfr.learning.batchSize",
                      "64",
                      "epsilon.cfr.initialization.games",
                      "4")));
      Assert.assertEquals(config.seed(), 9_007_199_254_740_993L);
      Assert.assertEquals(config.utilityProfile(), EpsilonUtilityProfile.LAST_AVOIDANCE);
      Assert.assertEquals(config.learning().batchSize(), 64);
      Assert.assertEquals(config.learning().microBatchSize(), 8);
      Assert.assertEquals(config.learning().averageUpdates(), 256);
      Assert.assertFalse(config.collection().branchComparison().enabled());
      Assert.assertTrue(config.collection().branchComparison().winDecline());
      Assert.assertEquals(config.collection().branchComparison().maxExtraInferenceRatio(), .02);
      Assert.assertEquals(config.network().hidden(), 256);
      Assert.assertEquals(config.network().devices().learner(), "gpu:0, gpu:1");
      Assert.assertEquals(config.network().devices().inference(), "gpu:1");
      Assert.assertEquals(config.replay().maxStoredBytes(), 137_438_953_473L);
      Assert.assertEquals(config.boundary().checkpoint(), "model/grp");
      Assert.assertEquals(
          config.initialization(), new CfrConfig.Initialization("model/nano", 4, 8, 65536));
    } finally {
      Files.delete(file);
    }
  }

  @Test
  public void crossFieldChecksRejectUnsupportedExecutionCombinations() {
    for (Map<String, String> overrides :
        List.of(
            Map.of(
                "epsilon.cfr.learning.batchSize", "8", "epsilon.cfr.learning.microBatchSize", "16"),
            Map.of(
                "epsilon.cfr.replay.maxStoredBytes",
                "1024",
                "epsilon.cfr.replay.maxRowBytes",
                "2048"),
            Map.of("epsilon.cfr.collection.inferenceBatch", "16"),
            Map.of(
                "epsilon.cfr.collection.slotsPerDevice",
                "4",
                "epsilon.cfr.collection.readyBatchesPerDevice",
                "2"),
            Map.of("epsilon.cfr.evaluation.championDuel.maximumWallSeeds", "50000"),
            Map.of("epsilon.cfr.collection.mode", "UNKNOWN"))) {
      Assert.expectThrows(
          IllegalArgumentException.class, () -> CfrConfig.load(EpsilonSettings.of(overrides)));
    }
  }

  @Test
  public void removedAverageCadenceIsRejectedInSavedConfiguration() throws Exception {
    Path file = Files.createTempFile("cfr-old-average-cadence-", ".json");
    try {
      var previous = CfrConfig.JSON.toJsonTree(CfrConfig.defaults()).getAsJsonObject();
      previous.getAsJsonObject("learning").addProperty("averageEvery", 4);
      Files.writeString(file, previous.toString());
      Assert.expectThrows(IOException.class, () -> CfrConfig.read(file));
    } finally {
      Files.delete(file);
    }
  }

  @Test
  public void evaluationScheduleIncludesTheLastIterationOfEachSolve() {
    CfrConfig config = CfrConfig.load(EpsilonSettings.of(Map.of("epsilon.cfr.iterations", "25")));
    for (long iteration = 1; iteration <= 25; iteration++) {
      Assert.assertEquals(
          config.evaluatesAt(iteration),
          iteration == 10 || iteration == 20 || iteration == 25,
          "Unexpected default evaluation at iteration " + iteration);
    }

    config =
        CfrConfig.load(
            EpsilonSettings.of(
                Map.of("epsilon.cfr.iterations", "25", "epsilon.cfr.evaluation.every", "7")));
    for (long iteration = 1; iteration <= 25; iteration++) {
      Assert.assertEquals(
          config.evaluatesAt(iteration),
          iteration == 7 || iteration == 14 || iteration == 21 || iteration == 25,
          "Unexpected configured evaluation at iteration " + iteration);
    }
  }
}

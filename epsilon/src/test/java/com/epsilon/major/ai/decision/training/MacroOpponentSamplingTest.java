package com.epsilon.major.ai.decision.training;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.major.config.settings.DecisionOpponentSettings;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** macro内の相手固定、履歴の識別、抽選と再開の再現性を検証する。 */
public class MacroOpponentSamplingTest {
  @Test
  public void settingsControlRegistrationIntervalAndValidateBounds() {
    var defaults = EpsilonSettings.defaults().bind(DecisionOpponentSettings.class);
    Assert.assertEquals(defaults.championProbability(), 0.7);
    Assert.assertEquals(defaults.snapshotIntervalMacros(), 10);
    var custom =
        EpsilonSettings.of(
                Map.of(
                    "epsilon.decision.train.opponents.championProbability", "0.25",
                    "epsilon.decision.train.opponents.snapshotIntervalMacros", "7"))
            .bind(DecisionOpponentSettings.class);
    Assert.assertEquals(custom.championProbability(), 0.25);
    Assert.assertFalse(custom.shouldRegister(0));
    Assert.assertFalse(custom.shouldRegister(6));
    Assert.assertTrue(custom.shouldRegister(7));
    Assert.assertTrue(custom.shouldRegister(14));
    for (String invalid : new String[] {"-0.1", "1.1", "NaN"}) {
      Assert.expectThrows(
          IllegalArgumentException.class,
          () ->
              EpsilonSettings.of(
                      Map.of("epsilon.decision.train.opponents.championProbability", invalid))
                  .bind(DecisionOpponentSettings.class));
    }
    Assert.expectThrows(
        IllegalArgumentException.class,
        () ->
            EpsilonSettings.of(
                    Map.of("epsilon.decision.train.opponents.snapshotIntervalMacros", "0"))
                .bind(DecisionOpponentSettings.class));
  }

  @Test
  public void probabilityEndpointsSelectOnlyChampionOrOnlyHistory() throws Exception {
    Path root = Files.createTempDirectory("macro-opponent-endpoints-");
    try {
      var pool = new EpsilonDecisionSnapshotPool(2);
      pool.registerArenaChampion(checkpoint(root.resolve("champion"), 2, 10));
      pool.registerMacroCheckpoint(checkpoint(root.resolve("macro_010"), 3, 20));
      for (int seed = 0; seed < 100; seed++) {
        Assert.assertEquals(pool.sampleOpponentIdsForMacro(seed, 1.0)[0][0], 1L);
        Assert.assertEquals(pool.sampleOpponentIdsForMacro(seed, 0.0)[0][0], 2L);
      }
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void fixesEverySeatToOneOpponentAndSamplesSeventyThirty() throws Exception {
    Path root = Files.createTempDirectory("macro-opponent-ratio-");
    try {
      var pool = new EpsilonDecisionSnapshotPool(4);
      pool.registerArenaChampion(checkpoint(root.resolve("champion"), 2, 10));
      pool.registerMacroCheckpoint(checkpoint(root.resolve("macro_010"), 3, 20));
      pool.registerMacroCheckpoint(checkpoint(root.resolve("macro_020"), 3, 30));
      int[] counts = new int[4];
      for (long seed = 0; seed < 10000; seed++) {
        long[][] opponents = pool.sampleOpponentIdsForMacro(seed, 0.7);
        long id = opponents[0][0];
        Assert.assertEquals(opponents.length, 4);
        for (long[] seats : opponents) {
          Assert.assertEquals(seats, new long[] {id, id, id});
        }
        Assert.assertTrue(Arrays.deepEquals(opponents, pool.sampleOpponentIdsForMacro(seed, 0.7)));
        counts[(int) id]++;
      }
      Assert.assertTrue(counts[1] > 6800 && counts[1] < 7200, Arrays.toString(counts));
      Assert.assertTrue(counts[2] > 1300 && counts[2] < 1700, Arrays.toString(counts));
      Assert.assertTrue(counts[3] > 1300 && counts[3] < 1700, Arrays.toString(counts));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void fallsBackToChampionWhenHistoryIsEmpty() throws Exception {
    Path root = Files.createTempDirectory("macro-opponent-only-champion-");
    try {
      var pool = new EpsilonDecisionSnapshotPool(1);
      pool.registerArenaChampion(checkpoint(root.resolve("champion"), 5, 10));
      pool.registerMacroCheckpoint(checkpoint(root.resolve("macro_010"), 6, 20));
      Assert.assertEquals(pool.size(), 1);
      for (int seed = 0; seed < 100; seed++) {
        Assert.assertEquals(pool.sampleOpponentIdsForMacro(seed, 0.7)[0][0], 1L);
      }
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void sameIterationMacrosSurviveReloadWithDeterministicSampling() throws Exception {
    Path root = Files.createTempDirectory("macro-opponent-resume-");
    try {
      var pool = new EpsilonDecisionSnapshotPool(3);
      pool.registerArenaChampion(checkpoint(root.resolve("champion"), 2, 10));
      Path first = checkpoint(root.resolve("macro_010"), 3, 20);
      Path second = checkpoint(root.resolve("macro_020"), 3, 30);
      pool.registerMacroCheckpoint(first);
      pool.registerMacroCheckpoint(second);
      pool.save(root);
      var restored = EpsilonDecisionSnapshotPool.load(root, 3);
      Assert.assertEquals(restored.size(), 3);
      Assert.assertEquals(restored.snapshot(2).iteration(), restored.snapshot(3).iteration());
      Assert.assertEquals(
          DecisionSnapshotEvaluatorSet.resolveSnapshotCheckpoint(2, restored.snapshot(2)), first);
      Assert.assertEquals(
          DecisionSnapshotEvaluatorSet.resolveSnapshotCheckpoint(3, restored.snapshot(3)), second);
      for (int macro = 21; macro < 50; macro++) {
        long seed = 9917L + macro * 1_000_000L;
        Assert.assertTrue(
            Arrays.deepEquals(
                pool.sampleOpponentIdsForMacro(seed, 0.7),
                restored.sampleOpponentIdsForMacro(seed, 0.7)));
      }
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void evictionKeepsChampionAndNewestHistoryWithoutDeletingWeights() throws Exception {
    Path root = Files.createTempDirectory("macro-opponent-eviction-");
    try {
      var pool = new EpsilonDecisionSnapshotPool(2);
      pool.registerArenaChampion(checkpoint(root.resolve("champion"), 2, 10));
      Path first = checkpoint(root.resolve("macro_010"), 3, 20);
      pool.registerMacroCheckpoint(first);
      pool.registerMacroCheckpoint(checkpoint(root.resolve("macro_020"), 3, 30));
      Assert.assertEquals(pool.size(), 2);
      Assert.assertTrue(pool.snapshot(1).currentArenaChampion());
      Assert.assertNull(pool.snapshot(2));
      Assert.assertEquals(pool.snapshot(3).createdStep(), 30);
      Assert.assertTrue(Files.exists(first.resolve("manifest.json")));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void promotionReusesMacroIdentityInsteadOfDuplicatingCurrentWeights() throws Exception {
    Path root = Files.createTempDirectory("macro-opponent-promotion-");
    try {
      var pool = new EpsilonDecisionSnapshotPool(4);
      pool.registerArenaChampion(checkpoint(root.resolve("champion"), 2, 10));
      pool.registerMacroCheckpoint(checkpoint(root.resolve("macro_010"), 3, 20));
      Path promoted = checkpoint(root.resolve("promoted"), 3, 20);
      var stats = new EpsilonDecisionSnapshotPool.EvalStats(2.4f, 0.3f, 0.2f);
      pool.registerArenaChampion(promoted, stats);
      pool.registerArenaChampion(promoted);
      Assert.assertEquals(pool.size(), 2);
      Assert.assertFalse(pool.snapshot(1).currentArenaChampion());
      Assert.assertTrue(pool.snapshot(2).currentArenaChampion());
      Assert.assertEquals(pool.snapshot(2).path(), promoted.toString());
      Assert.assertEquals(pool.snapshot(2).evalStats(), stats);
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void readsVersionTwoRegistryWithoutChangingExistingIds() throws Exception {
    Path root = Files.createTempDirectory("macro-opponent-v2-");
    try {
      var pool = new EpsilonDecisionSnapshotPool(3);
      pool.registerArenaChampion(checkpoint(root.resolve("champion"), 5, 10));
      pool.save(root);
      Path registry = root.resolve("decision-snapshot-pool.json");
      JsonObject json = JsonParser.parseString(Files.readString(registry)).getAsJsonObject();
      json.addProperty("version", 2);
      JsonObject entry = json.getAsJsonArray("snapshots").get(0).getAsJsonObject();
      entry.remove("id");
      entry.remove("iteration");
      entry.addProperty("version", 5);
      entry.addProperty("candidateId", "initial/00005");
      Files.writeString(registry, json.toString());
      var restored = EpsilonDecisionSnapshotPool.load(root, 3);
      Assert.assertEquals(restored.sampleOpponentIdsForMacro(17, 0.7)[0][0], 5L);
      Assert.assertEquals(restored.snapshot(5).iteration(), 5);
      restored.registerMacroCheckpoint(checkpoint(root.resolve("macro_010"), 6, 20));
      Assert.assertEquals(restored.snapshot(6).createdStep(), 20);
      restored.save(root);
      Assert.assertEquals(
          EpsilonDecisionSnapshotPool.load(root, 3).snapshot(5), restored.snapshot(5));
    } finally {
      deleteTree(root);
    }
  }

  private static Path checkpoint(Path path, int iteration, int step) throws Exception {
    Files.createDirectories(path);
    var manifest =
        new EpsilonDecisionCheckpointBundle(step, iteration, 29, 16, EpsilonUtilityProfile.TENHOU);
    Files.writeString(path.resolve("manifest.json"), new Gson().toJson(manifest));
    Files.writeString(path.resolve("architecture.id"), manifest.architecture);
    Files.write(path.resolve(manifest.modelFile), new byte[] {1, 2, 3});
    return path;
  }

  private static void deleteTree(Path directory) throws Exception {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}

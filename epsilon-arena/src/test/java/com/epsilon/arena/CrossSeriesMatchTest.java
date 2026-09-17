package com.epsilon.arena;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.major.EpsilonModelProvider;
import com.epsilon.nano.NanoModelProvider;
import com.epsilon.pico.PicoModelProvider;
import com.epsilon.runtime.MeasurementStatus;
import com.epsilon.spi.PolicyExecutionContext;
import com.google.gson.GsonBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 各系列の実チェックポイント・エンコーダー・ネットワークを共有対戦環境へ接続し、半荘を最後まで実行する。 */
public class CrossSeriesMatchTest {
  @Test(groups = "native", timeOut = 900000)
  public void allSeriesPairsAndFourMixedSeatsCompleteWithSharedRuntimes() throws Exception {
    Path target = Path.of("target").toAbsolutePath();
    Files.createDirectories(target);
    Path checkpoints = Files.createTempDirectory(target, "cross-series-checkpoints-");
    Path reportPath = target.resolve("cross-series-match.json");
    List<Map<String, Object>> completedMatches = new ArrayList<>();
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("complete", false);
    report.put("device", "cpu");
    report.put("hidden", 32);
    report.put("utilityProfile", "TOP");
    report.put("modelSeeds", List.of(1701, 1702, 1703));
    report.put("matches", completedMatches);
    try {
      createCheckpoints(checkpoints);
      Map<String, String> options =
          Map.of(
              "device",
              "cpu",
              "epsilon.decision.inference.maxBatch",
              "32",
              "epsilon.decision.inference.multiTransitionMaxBatch",
              "32",
              "epsilon.decision.inference.computePrecision",
              "FLOAT32");
      List<ModelSpec> specs =
          List.of(
              new ModelSpec("epsilon-nano", checkpoints.resolve("nano"), options),
              new ModelSpec("epsilon-pico", checkpoints.resolve("pico"), options),
              new ModelSpec("epsilon", checkpoints.resolve("major"), options));
      RunSettings settings = new RunSettings(4, 4, 2, 98117L, 0);
      try (PolicyExecutionContext execution =
              new PolicyExecutionContext(
                  settings.workers(), settings.slotsPerDevice(), settings.readyBatchesPerDevice());
          ModelRegistry registry =
              new ModelRegistry(
                  List.of(
                      new NanoModelProvider(), new PicoModelProvider(), new EpsilonModelProvider()),
                  execution)) {
        List<Participant> models = new ArrayList<>(3);
        for (ModelSpec spec : specs) models.add(registry.open(spec));
        for (int index = 0; index < specs.size(); index++)
          Assert.assertSame(registry.open(specs.get(index)).policy(), models.get(index).policy());

        for (int candidate = 0; candidate < 3; candidate++) {
          for (int opponent = candidate; opponent < 3; opponent++) {
            Participant first = models.get(candidate);
            Participant other = models.get(opponent);
            ArenaResult result =
                new ArenaRunner(List.of(first, other, other, other), settings).run();
            verifyCompleteMatch(result, candidate == opponent ? 1 : 2);
            completedMatches.add(
                Map.of(
                    "name",
                    specs.get(candidate).series() + " vs " + specs.get(opponent).series(),
                    "result",
                    result));
            writeReport(reportPath, report);
          }
        }
        Participant nano = models.get(0);
        Participant pico = models.get(1);
        Participant major = models.get(2);
        ArenaResult mixed = new ArenaRunner(List.of(nano, pico, major, major), settings).run();
        verifyCompleteMatch(mixed, 3);
        completedMatches.add(Map.of("name", "Nano / Pico / Epsilon / Epsilon", "result", mixed));
      }
      Assert.assertEquals(completedMatches.size(), 7);
      report.put("complete", true);
    } catch (Exception | AssertionError failure) {
      report.put("failure", failure.toString());
      throw failure;
    } finally {
      try {
        writeReport(reportPath, report);
      } finally {
        try (var paths = Files.walk(checkpoints)) {
          for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
        }
      }
    }
  }

  private static void verifyCompleteMatch(ArenaResult result, int expectedRuntimes) {
    Assert.assertEquals(result.players().size(), 4);
    Assert.assertEquals(result.paired().walls(), 1);
    Assert.assertEquals(result.inference().size(), expectedRuntimes);
    double totalAverageScore = 0;
    int firstPlaces = 0;
    for (var player : result.players()) {
      Assert.assertEquals(player.first() + player.second() + player.third() + player.fourth(), 4);
      Assert.assertTrue(player.averageRank() >= 1 && player.averageRank() <= 4);
      totalAverageScore += player.averageScore();
      firstPlaces += player.first();
    }
    Assert.assertEquals(firstPlaces, 4);
    Assert.assertEquals(totalAverageScore, 100000.0, 1e-9, "終局時には残った供託も配分される");
    for (var inference : result.inference()) {
      var json = new GsonBuilder().create().toJsonTree(inference).getAsJsonObject();
      Assert.assertEquals(
          json.getAsJsonObject("providerTimings").get("status").getAsString(), "available");
      Assert.assertTrue(inference.rows() > 0);
      Assert.assertTrue(inference.batches() > 0);
      Assert.assertTrue(inference.rowsPerSecond() > 0);
      Assert.assertEquals(inference.providerTimings().status(), MeasurementStatus.AVAILABLE);
      Assert.assertEquals(inference.providerTimings().completedBatches(), inference.batches());
      Assert.assertTrue(inference.providerTimings().meanEncoderQueueMillis() >= 0);
      Assert.assertTrue(inference.providerTimings().meanEncodingMillis() >= 0);
      Assert.assertTrue(inference.providerTimings().meanInferenceRoundTripMillis() >= 0);
    }
  }

  private static void createCheckpoints(Path directory) throws Exception {
    Engine engine = Engine.getEngine("PyTorch");
    engine.setRandomSeed(1701);
    try (Model nano =
        com.epsilon.nano.ai.network.NetworkFactory.createDecisionModel(
            Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
      com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager.saveInitial(
          nano, directory.resolve("nano"));
    }
    engine.setRandomSeed(1702);
    try (Model pico =
        com.epsilon.pico.ai.network.NetworkFactory.createDecisionModel(
            Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
      com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager.saveInitial(
          pico, directory.resolve("pico"));
    }
    engine.setRandomSeed(1703);
    try (Model major =
        com.epsilon.major.ai.network.NetworkFactory.createDecisionModel(
            Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
      com.epsilon.major.ai.decision.training.EpsilonDecisionCheckpointManager.saveInitial(
          major, directory.resolve("major"));
    }
  }

  private static void writeReport(Path path, Map<String, Object> report) throws Exception {
    Files.writeString(
        path,
        new GsonBuilder().setPrettyPrinting().create().toJson(report) + System.lineSeparator());
  }
}

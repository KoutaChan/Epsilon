package com.epsilon.major.ai.decision.duel;

import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.arena.EpsilonDecisionDuelArena;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionInferenceServer.Prediction;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 学習用の探索設定を変えても評価対局が最大確率の行動を選ぶことを検証する。 */
public class DecisionEvaluationSelectionTest {
  @DataProvider
  public Object[][] trainingModes() {
    return new Object[][] {{"FULL_SUPPORT"}, {"POLICY_SAMPLE"}};
  }

  @Test(dataProvider = "trainingModes")
  public void selfEvaluationKeepsTheGreedyReplayUnderTrainingSettings(String mode)
      throws Exception {
    JsonObject expected = replay(settings("POLICY_GREEDY"));
    Assert.assertEquals(replay(settings(mode)), expected);
  }

  @Test(dataProvider = "trainingModes")
  public void versusAndWallEvaluationKeepGreedySeatRotationsUnderTrainingSettings(String mode) {
    var expected = duel(settings("POLICY_GREEDY"));
    var actual = duel(settings(mode));
    Assert.assertEquals(actual.rotationOutcomes(), expected.rotationOutcomes());
    Assert.assertEquals(actual.wallOutcomes(), expected.wallOutcomes());
    Assert.assertEquals(actual.rotationOutcomes().size(), 4);
    List<DuelEvaluation.WallOutcome> streamed = new ArrayList<>();
    var continuous =
        EpsilonDecisionDuelArena.evaluateDuelStreaming(
            Path.of("candidate"),
            Path.of("opponent"),
            evaluator(true),
            evaluator(false),
            4,
            193L,
            0L,
            1,
            streamed::add,
            settings(mode));
    Assert.assertEquals(streamed, expected.wallOutcomes());
    Assert.assertEquals(continuous.result().games(), 4);
  }

  private static EpsilonDecisionDuelArena.Evaluation duel(SettingsLoader settings) {
    return EpsilonDecisionDuelArena.evaluateDuel(
        Path.of("candidate"),
        Path.of("opponent"),
        evaluator(true),
        evaluator(false),
        4,
        193L,
        0L,
        1,
        settings);
  }

  private static JsonObject replay(SettingsLoader settings) throws Exception {
    Path directory = Files.createTempDirectory("epsilon-greedy-evaluation-");
    try {
      var result =
          EpsilonDecisionEvaluationRunner.evaluate(evaluator(true), 1, directory, settings);
      Assert.assertEquals(result.games(), 1);
      Assert.assertEquals(result.tenhouLogFiles().size(), 1);
      var replay =
          JsonParser.parseString(
                  Files.readAllLines(result.tenhouLogFiles().getFirst()).stream()
                      .filter(line -> line.startsWith("{"))
                      .findFirst()
                      .orElseThrow())
              .getAsJsonObject();
      replay.remove("title");
      replay.remove("ref");
      return replay;
    } finally {
      try (var files = Files.list(directory)) {
        for (Path file : files.toList()) Files.delete(file);
      }
      Files.delete(directory);
    }
  }

  private static SettingsLoader settings(String mode) {
    return EpsilonSettings.of(Map.of("epsilon.decision.rollout.selectionMode", mode));
  }

  private static EpsilonDecisionEvaluator evaluator(boolean preferLast) {
    return batch -> {
      List<Prediction> predictions = new ArrayList<>();
      for (int row = 0; row < batch.size(); row++) {
        int actions = batch.legalActionCount(row);
        float[] logProbabilities = new float[actions];
        Arrays.fill(logProbabilities, -1.0f);
        logProbabilities[preferLast ? actions - 1 : 0] = 0.0f;
        predictions.add(new Prediction(logProbabilities, 0.0f));
      }
      return predictions;
    };
  }
}

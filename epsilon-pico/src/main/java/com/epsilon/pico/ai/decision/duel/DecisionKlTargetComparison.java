package com.epsilon.pico.ai.decision.duel;

import com.epsilon.ai.decision.duel.EpsilonDecisionFixedSampleConfidenceInterval;
import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator;
import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.ai.decision.data.DecisionJsonFiles;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** 同じ固定対戦相手・牌山・4席で、目標KLを変えた二つの学習結果の差を直接測る。 */
public final class DecisionKlTargetComparison {

  private static final Gson GSON = new GsonBuilder().serializeNulls().setPrettyPrinting().create();

  private DecisionKlTargetComparison() {}

  public static Result evaluate(
      Path a,
      Path b,
      Path opponent,
      Path reports,
      long seed,
      DecisionChampionDuelSettings plan,
      DecisionExecutionContext context)
      throws IOException {
    return evaluate(a, b, opponent, reports, seed, plan, context, EpsilonSettings.defaults());
  }

  public static Result evaluate(
      Path a,
      Path b,
      Path opponent,
      Path reports,
      long seed,
      DecisionChampionDuelSettings plan,
      DecisionExecutionContext context,
      SettingsLoader config)
      throws IOException {
    Files.createDirectories(reports);
    var bounds =
        EpsilonDecisionWallDuelEvaluator.utilityBounds(
            config.bind(com.epsilon.pico.config.settings.DecisionSettings.class).utilityProfile());
    var differences =
        new EpsilonDecisionFixedSampleConfidenceInterval(
            bounds.lower() - bounds.upper(), bounds.upper() - bounds.lower());
    int[] looks = {plan.interimWallSeeds(), plan.maximumWallSeeds()};
    double[] alphas = {plan.interimAlpha(), plan.finalLookAlpha()};
    Result result = null;
    for (int look = 0; look < looks.length; look++) {
      int firstWall = differences.samples();
      int walls = looks[look] - firstWall;
      // インデックスにより対応付ける。非同期対局実行環境の完了順を対応関係として使わない。
      double[] aOutcomes = evaluateArm(a, opponent, seed, firstWall, walls, context, config);
      double[] bOutcomes = evaluateArm(b, opponent, seed, firstWall, walls, context, config);
      for (int i = 0; i < walls; i++) {
        differences.add(bOutcomes[i] - aOutcomes[i]);
      }
      var estimate = differences.estimate(alphas[look]);
      result =
          new Result(
              "kl-target-paired-comparison-v1",
              seed,
              looks[look],
              look + 1,
              plan.alpha(),
              alphas[look],
              estimate.mean(),
              estimate.standardError(),
              estimate.lower(),
              estimate.upper(),
              decide(estimate),
              "FIXED_LOOK_GAUSSIAN_APPROXIMATION",
              a.toString(),
              b.toString(),
              opponent.toString());
      DecisionJsonFiles.write(reports.resolve("look-" + (look + 1) + ".json"), result, GSON);
      if (result.decision() != Decision.UNRESOLVED) break;
    }
    return result;
  }

  private static double[] evaluateArm(
      Path candidate,
      Path opponent,
      long seed,
      int firstWall,
      int walls,
      DecisionExecutionContext context,
      SettingsLoader config)
      throws IOException {
    double[] outcomes = new double[walls];
    try (var session = EpsilonDecisionDuelSession.open(candidate, opponent, context, config)) {
      session.evaluateWalls(
          walls,
          seed,
          firstWall,
          config.bind(DecisionEvalVsSettings.class).gamesInFlight(),
          outcome ->
              outcomes[Math.toIntExact(outcome.wallIndex() - firstWall)] =
                  outcome.pairedConfiguredUtilityDelta());
    }
    return outcomes;
  }

  static Decision decide(EpsilonDecisionFixedSampleConfidenceInterval.Estimate estimate) {
    if (estimate.lower() > 0.0) return Decision.LOWER_TARGET_BETTER;
    if (estimate.upper() < 0.0) return Decision.CURRENT_TARGET_BETTER;
    return Decision.UNRESOLVED;
  }

  /** 対応付き信頼区間が示す目標KLの比較結果。 */
  public enum Decision {
    LOWER_TARGET_BETTER,
    CURRENT_TARGET_BETTER,
    UNRESOLVED
  }

  public record Result(
      String schema,
      long seed,
      int wallSeeds,
      int completedLooks,
      double totalAlpha,
      double decisionAlpha,
      double meanDifference,
      double standardError,
      double lower,
      double upper,
      Decision decision,
      String confidenceMethod,
      String a,
      String b,
      String opponent) {}
}

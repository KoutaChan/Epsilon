package com.epsilon.cfr.training;

import static com.epsilon.cfr.training.CfrIterationRunner.reportName;

import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.evaluation.CfrDuelSession;
import com.epsilon.io.AtomicFiles;
import com.google.gson.Gson;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/** 固定対戦相手による評価と、再開可能な昇格判断を担う。 */
final class CfrEvaluationRunner {
  private static final Gson REPORT_JSON =
      CfrConfig.JSON
          .newBuilder()
          .registerTypeHierarchyAdapter(
              Path.class,
              (JsonSerializer<Path>) (path, type, context) -> new JsonPrimitive(path.toString()))
          .create();

  private CfrEvaluationRunner() {}

  static void evaluate(CfrRunStore store, CfrConfig config, CfrRunState state) throws IOException {
    Path transaction =
        store.resolve(
            "evaluations/wall-duel/" + state.solveId() + "-" + state.iteration() + ".json");
    if (Files.exists(transaction)) {
      EvaluationDecision decision =
          CfrConfig.JSON.fromJson(Files.readString(transaction), EvaluationDecision.class);
      if (decision == null
          || decision.decision() == null
          || !state.averageCheckpoint().equals(decision.candidate()))
        throw new IOException("Evaluation candidate identity mismatch");
      if (decision.decision() == EpsilonDecisionWallDuelEvaluator.Decision.PROMOTED)
        store.publishChampion(decision.candidate(), state.solveId(), state.iteration());
      return;
    }
    CfrRunStore.Champion baseline = store.champion();
    var settings = config.evaluation();
    var duel = settings.championDuel();
    var plan =
        EpsilonDecisionWallDuelEvaluator.EvaluationPlan.twoLook(
            duel.interimWallSeeds(),
            duel.maximumWallSeeds(),
            duel.alpha(),
            duel.interimAlphaFraction());
    long sequence =
        Math.addExact(
            Math.multiplyExact((long) state.solveIndex(), config.iterations()), state.iteration());
    EpsilonDecisionWallDuelEvaluator.Result result;
    try (CfrDuelSession session =
        CfrDuelSession.open(
            store.resolve(state.averageCheckpoint()),
            store.resolve(baseline.checkpoint()),
            config)) {
      result =
          EpsilonDecisionWallDuelEvaluator.evaluate(
              session,
              plan,
              settings.evalVs().seedBase(),
              sequence,
              0,
              0,
              config.utilityProfile(),
              settings.evalVs());
    }
    store.report(reportName(state, "champion-duel"), REPORT_JSON.toJsonTree(result));
    // 昇格判断を保存してから採用モデルへの参照を更新する。再開時には、保存した判断を一度適用する。
    AtomicFiles.write(
        transaction,
        CfrConfig.JSON
            .toJson(
                new EvaluationDecision(
                    baseline.checkpoint(), state.averageCheckpoint(), result.decision()))
            .getBytes(StandardCharsets.UTF_8));
    if (result.decision() == EpsilonDecisionWallDuelEvaluator.Decision.PROMOTED)
      store.publishChampion(state.averageCheckpoint(), state.solveId(), state.iteration());
  }

  private record EvaluationDecision(
      String baseline, String candidate, EpsilonDecisionWallDuelEvaluator.Decision decision) {}
}

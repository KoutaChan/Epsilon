package com.epsilon.pico.workflow.benchmark;

import com.epsilon.pico.ai.decision.arena.EpsilonDecisionArena;
import com.epsilon.pico.ai.decision.benchmark.EpsilonDecisionCollectionBenchmark;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;
import java.util.Locale;

/** Decision の自己対局データを収集する速度を測定するコマンド。 */
public final class SelfPlayBenchmarkWorkflow
    implements CommandWorkflow<EpsilonDecisionCollectionBenchmark.Report> {

  private static final String DEFAULT_CHECKPOINT = "checkpoints/epsilon-pico";

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.BENCHMARK_DECISION_SELFPLAY;
  }

  @Override
  public EpsilonDecisionCollectionBenchmark.Report execute(WorkflowArguments arguments)
      throws Exception {
    Path checkpoint = Path.of(arguments.optionalText(0, DEFAULT_CHECKPOINT));
    Path scratch =
        Path.of(arguments.optionalText(3, checkpoint.resolve("benchmark-inflight").toString()));
    return EpsilonDecisionCollectionBenchmark.run(
        checkpoint,
        arguments.optionalInteger(1, "games", 2_048),
        arguments.optionalLong(2, "seedBase", 79_000_000L),
        scratch,
        arguments.optionalInteger(4, "warmupGames", 2_048),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionCollectionBenchmark.Report report) {
    EpsilonDecisionArena.Metrics metrics = report.arenaMetrics();
    return "decisionsPerSecond="
        + String.format(Locale.ROOT, "%.3f", report.decisionsPerSecond())
        + " games="
        + report.games()
        + " warmupGames="
        + report.warmupGames()
        + " configuredGamesInFlight="
        + report.configuredGamesInFlight()
        + " effectiveGamesInFlight="
        + report.effectiveGamesInFlight()
        + " samples="
        + report.samples()
        + " elapsedMillis="
        + report.elapsedMillis()
        + " gamesPerSecond="
        + String.format(Locale.ROOT, "%.3f", report.gamesPerSecond())
        + " computePrecision="
        + report.computePrecision()
        + " actorDevices="
        + report.actorDevices()
        + " opponentDevices="
        + report.opponentDevices()
        + " inferenceBatches="
        + metrics.inferenceBatchCount()
        + " inferenceRequests="
        + metrics.inferenceRequestCount()
        + " averageInferenceBatch="
        + String.format(Locale.ROOT, "%.3f", metrics.averageInferenceBatchSize())
        + " maximumInferenceBatch="
        + metrics.maxInferenceBatchSize();
  }
}

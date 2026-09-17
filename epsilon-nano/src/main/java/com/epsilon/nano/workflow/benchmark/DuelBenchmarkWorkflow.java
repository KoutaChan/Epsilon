package com.epsilon.nano.workflow.benchmark;

import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionDuelBenchmark;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** 牌山の乱数シードを固定し、Decision の対戦評価の処理速度を測定するコマンド。 */
public final class DuelBenchmarkWorkflow
    implements CommandWorkflow<EpsilonDecisionDuelBenchmark.Report> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.BENCHMARK_DECISION_DUEL;
  }

  @Override
  public EpsilonDecisionDuelBenchmark.Report execute(WorkflowArguments arguments) throws Exception {
    return EpsilonDecisionDuelBenchmark.run(
        arguments.path(0, "candidateCheckpoint"),
        arguments.path(1, "parentCheckpoint"),
        arguments.optionalInteger(2, "warmupWallSeeds", 64),
        arguments.optionalInteger(3, "measuredWallSeeds", 512),
        arguments.optionalLong(4, "seedBase", 98_200_000L),
        arguments.optionalInteger(
            5,
            "gamesInFlight",
            arguments.settings().bind(DecisionEvalVsSettings.class).gamesInFlight()),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionDuelBenchmark.Report report) {
    return "candidate="
        + report.candidateCheckpoint()
        + " parent="
        + report.parentCheckpoint()
        + " warmupWallSeeds="
        + report.warmupWallSeeds()
        + " measuredWallSeeds="
        + report.measuredWallSeeds()
        + " games="
        + report.games()
        + " elapsedMillis="
        + (report.elapsedNanos() / 1_000_000.0)
        + " gamesPerSecond="
        + report.gamesPerSecond()
        + " wallsPerSecond="
        + report.wallsPerSecond()
        + " inferenceBatches="
        + report.inferenceBatches()
        + " inferenceRequests="
        + report.inferenceRequests()
        + " averageInferenceBatch="
        + report.averageInferenceBatch()
        + " maximumInferenceBatch="
        + report.maximumInferenceBatch();
  }
}

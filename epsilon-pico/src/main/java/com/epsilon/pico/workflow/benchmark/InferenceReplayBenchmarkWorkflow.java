package com.epsilon.pico.workflow.benchmark;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.ai.decision.benchmark.EpsilonDecisionInferenceReplayBenchmark;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;

/** 固定した入力データで、Decision 推論の再現性と処理速度を測定するコマンド。 */
public final class InferenceReplayBenchmarkWorkflow
    implements CommandWorkflow<EpsilonDecisionInferenceReplayBenchmark.Report> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.BENCHMARK_DECISION_INFERENCE_REPLAY;
  }

  @Override
  public EpsilonDecisionInferenceReplayBenchmark.Report execute(WorkflowArguments arguments)
      throws Exception {
    SettingsLoader snapshot = arguments.settings();
    Path checkpoint = arguments.path(0, "checkpoint");
    Path reportFile = arguments.path(1, "reportFile");
    Path corpusFile = arguments.path(2, "corpusFile");
    EpsilonDecisionInferenceReplayBenchmark.CorpusMode corpusMode =
        EpsilonDecisionInferenceReplayBenchmark.CorpusMode.parse(arguments.text(3, "corpusMode"));
    long seedBase = arguments.optionalLong(4, "seedBase", 202_608_310_001L);
    int repetitions = arguments.optionalInteger(5, "repetitions", 3);
    double targetRowsPerSecond = arguments.optionalDouble(6, "targetRowsPerSecond", 181_000.0);
    if (arguments.size() == 8) {
      return EpsilonDecisionInferenceReplayBenchmark.run(
          checkpoint,
          reportFile,
          corpusFile,
          corpusMode,
          seedBase,
          repetitions,
          targetRowsPerSecond,
          EpsilonDecisionInferenceReplayBenchmark.parseBatchRowsCsv(arguments.text(7, "batchRows")),
          snapshot);
    }
    return EpsilonDecisionInferenceReplayBenchmark.run(
        checkpoint,
        reportFile,
        corpusFile,
        corpusMode,
        seedBase,
        repetitions,
        targetRowsPerSecond,
        snapshot);
  }

  @Override
  public String summarize(EpsilonDecisionInferenceReplayBenchmark.Report report) {
    return "classification="
        + report.verdict().classification()
        + " reason="
        + report.verdict().reason();
  }
}

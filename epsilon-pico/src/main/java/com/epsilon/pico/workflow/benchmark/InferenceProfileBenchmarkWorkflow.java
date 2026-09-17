package com.epsilon.pico.workflow.benchmark;

import com.epsilon.pico.ai.decision.benchmark.EpsilonDecisionInferenceProfileBenchmark;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;

/** 固定した入力の用途と形状を明示し、演算ごとの所要時間を全体の処理速度とは別に測定する。 */
public final class InferenceProfileBenchmarkWorkflow
    implements CommandWorkflow<EpsilonDecisionInferenceProfileBenchmark.Report> {

  private static final WorkflowDefinition DEFINITION =
      WorkflowDefinition.range(
          "profile-decision-inference",
          WorkflowKind.BENCHMARK,
          "profile-decision-inference <checkpoint> <corpus.bin> <outputDir>"
              + " <ACTOR|OPPONENT> <rows> <transitionCapacity> [repetitions=3] [sections=all]",
          6,
          8);

  @Override
  public WorkflowDefinition definition() {
    return DEFINITION;
  }

  @Override
  public EpsilonDecisionInferenceProfileBenchmark.Report execute(WorkflowArguments arguments)
      throws Exception {
    return EpsilonDecisionInferenceProfileBenchmark.run(
        arguments.path(0, "checkpoint"),
        arguments.path(1, "corpus"),
        arguments.path(2, "outputDir"),
        EpsilonDecisionInferenceProfileBenchmark.Role.parse(arguments.text(3, "role")),
        arguments.integer(4, "rows"),
        arguments.integer(5, "transitionCapacity"),
        arguments.optionalInteger(6, "repetitions", 3),
        EpsilonDecisionInferenceProfileBenchmark.sections(
            arguments.size() == 8 ? arguments.text(7, "sections") : "all"),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionInferenceProfileBenchmark.Report report) {
    return "role="
        + report.role()
        + " rows="
        + report.rows()
        + " sections="
        + report.samples().size();
  }
}

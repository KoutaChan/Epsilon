package com.epsilon.workflow.benchmark;

import com.epsilon.training.ParallelSimulator;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** ランダムに行動するプレイヤーで複数の半荘を並列実行し、対局処理の速度を測る。 */
public final class EngineBenchmarkWorkflow implements CommandWorkflow<Integer> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.BENCHMARK;
  }

  @Override
  public Integer execute(WorkflowArguments arguments) {
    int games = arguments.optionalInteger(0, "games", 1_000);
    ParallelSimulator simulator = new ParallelSimulator();
    try {
      simulator.benchmark(games);
    } finally {
      simulator.shutdown();
    }
    return games;
  }

  @Override
  public String summarize(Integer games) {
    return "games=" + games;
  }
}

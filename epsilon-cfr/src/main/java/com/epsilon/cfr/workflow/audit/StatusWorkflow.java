package com.epsilon.cfr.workflow.audit;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.training.CfrRunState;
import com.epsilon.cfr.training.CfrRunStore;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;

/** 公開済みの学習状態を読み取る。 */
public final class StatusWorkflow implements CommandWorkflow<CfrRunState> {
  private final PrintStream output;

  public StatusWorkflow(PrintStream output) {
    this.output = output;
  }

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.exact("status", WorkflowKind.AUDIT, "status --run DIR", 2);
  }

  @Override
  public CfrRunState execute(WorkflowArguments arguments) throws IOException {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    options.requireEmpty();
    CfrRunState state = CfrRunStore.inspect(run);
    output.println(CfrConfig.JSON.toJson(state));
    return state;
  }

  @Override
  public String summarize(CfrRunState state) {
    return "solve="
        + state.solveId()
        + " iteration="
        + state.iteration()
        + " phase="
        + state.phase();
  }
}

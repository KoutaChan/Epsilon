package com.epsilon.cfr.workflow.audit;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.data.CfrReplayStore;
import com.epsilon.cfr.io.CfrFiles;
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

/** 公開済みの実行状態が参照する保存物と、保持中の再生データを検証する。 */
public final class VerifyWorkflow implements CommandWorkflow<VerifyWorkflow.Result> {
  public record Result(
      boolean verified, String solveId, long iteration, long replayGeneration, long retainedRows) {}

  private final PrintStream output;

  public VerifyWorkflow(PrintStream output) {
    this.output = output;
  }

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.exact("verify", WorkflowKind.AUDIT, "verify --run DIR", 2);
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws IOException {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    options.requireEmpty();
    Result result;
    try (CfrRunStore store = new CfrRunStore(run)) {
      CfrRunState state = store.verify();
      long rows = 0;
      if (state.replayGeneration() > 0) {
        Path replayPath = CfrFiles.inside(run, "solves/" + state.solveId() + "/replay");
        try (CfrReplayStore replay =
            CfrReplayStore.loadGeneration(replayPath, state.solveId(), state.replayGeneration())) {
          rows = replay.verifyPayloads();
        }
      }
      result = new Result(true, state.solveId(), state.iteration(), state.replayGeneration(), rows);
    }
    output.println(CfrConfig.JSON.toJson(result));
    return result;
  }

  @Override
  public String summarize(Result result) {
    return "solve="
        + result.solveId()
        + " iteration="
        + result.iteration()
        + " retainedRows="
        + result.retainedRows();
  }
}

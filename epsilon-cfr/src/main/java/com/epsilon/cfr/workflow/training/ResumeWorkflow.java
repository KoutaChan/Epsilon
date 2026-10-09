package com.epsilon.cfr.workflow.training;

import com.epsilon.cfr.training.CfrCampaign;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.io.IOException;
import java.nio.file.Path;

/** 公開済みの状態からLCFR学習を再開する。 */
public final class ResumeWorkflow implements CommandWorkflow<Path> {
  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.exact("resume", WorkflowKind.TRAINING, "resume --run DIR", 2);
  }

  @Override
  public Path execute(WorkflowArguments arguments) throws IOException {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    options.requireEmpty();
    CfrCampaign.resume(run);
    return run;
  }

  @Override
  public String summarize(Path run) {
    return "run=" + run.toAbsolutePath().normalize();
  }
}

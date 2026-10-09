package com.epsilon.cfr.workflow.training;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.training.CfrCampaign;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.io.IOException;
import java.nio.file.Path;

/** 起動時に確定した共通設定から新しいLCFR学習を開始する。 */
public final class TrainWorkflow implements CommandWorkflow<Path> {
  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.exact("train", WorkflowKind.TRAINING, "train --run DIR", 2);
  }

  @Override
  public Path execute(WorkflowArguments arguments) throws IOException {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    options.requireEmpty();
    CfrCampaign.train(CfrConfig.load(arguments.settings()), run);
    return run;
  }

  @Override
  public String summarize(Path run) {
    return "run=" + run.toAbsolutePath().normalize();
  }
}

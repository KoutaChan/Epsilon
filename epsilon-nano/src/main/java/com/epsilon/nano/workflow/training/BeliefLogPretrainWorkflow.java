package com.epsilon.nano.workflow.training;

import com.epsilon.nano.ai.belief.EpsilonBeliefLogPretrainRunner;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;

/** 牌譜から Belief を事前学習するワークフロー。 */
public final class BeliefLogPretrainWorkflow
    implements CommandWorkflow<EpsilonBeliefLogPretrainRunner.TrainSummary> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.PRETRAIN_BELIEF_LOGS;
  }

  @Override
  public EpsilonBeliefLogPretrainRunner.TrainSummary execute(WorkflowArguments arguments)
      throws Exception {
    return EpsilonBeliefLogPretrainRunner.pretrainFromLogs(
        arguments.optionalPath(0, Path.of("checkpoints/epsilon-nano/belief")),
        arguments.optionalPath(1, Path.of("data/logs")),
        arguments.optionalInteger(2, "epochs", 1),
        arguments.optionalInteger(3, "maxFiles", 0),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonBeliefLogPretrainRunner.TrainSummary result) {
    return "files="
        + result.files()
        + " skippedFiles="
        + result.skippedFiles()
        + " samples="
        + result.samples()
        + " batches="
        + result.batches()
        + " globalStep="
        + result.globalStep()
        + " iteration="
        + result.iteration()
        + " loss="
        + result.loss()
        + " handLoss="
        + result.handLoss()
        + " waitLoss="
        + result.waitLoss()
        + " scalarLoss="
        + result.scalarLoss();
  }
}

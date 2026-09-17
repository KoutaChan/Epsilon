package com.epsilon.pico.workflow.training;

import com.epsilon.pico.ai.decision.training.EpsilonDecisionLogPretrainRunner;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;

/** 牌譜から Decision を事前学習するワークフロー。 */
public final class DecisionLogPretrainWorkflow
    implements CommandWorkflow<EpsilonDecisionLogPretrainRunner.TrainSummary> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.PRETRAIN_DECISION_LOGS;
  }

  @Override
  public EpsilonDecisionLogPretrainRunner.TrainSummary execute(WorkflowArguments arguments)
      throws Exception {
    return EpsilonDecisionLogPretrainRunner.pretrainFromLogs(
        arguments.optionalPath(0, Path.of("checkpoints/epsilon-pico")),
        arguments.optionalPath(1, Path.of("data/logs")),
        arguments.optionalInteger(2, "epochs", 1),
        arguments.optionalInteger(3, "maxFiles", 0),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionLogPretrainRunner.TrainSummary result) {
    return "samples="
        + result.samples()
        + " batches="
        + result.batches()
        + " globalStep="
        + result.globalStep()
        + " iteration="
        + result.iteration()
        + " outputCheckpoint="
        + result.outputCheckpoint()
        + " validatedCheckpoints="
        + result.validations().size()
        + " arenaPromotionApplied=false";
  }
}

package com.epsilon.pico.workflow.evaluation;

import com.epsilon.pico.ai.decision.training.EpsilonDecisionLogPretrainRunner;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** 固定牌譜で Decision 事前学習チェックポイントを再検証するワークフロー。 */
public final class DecisionPretrainValidationWorkflow
    implements CommandWorkflow<EpsilonDecisionLogPretrainRunner.ValidationSummary> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.VALIDATE_DECISION_PRETRAIN;
  }

  @Override
  public EpsilonDecisionLogPretrainRunner.ValidationSummary execute(WorkflowArguments arguments)
      throws Exception {
    return EpsilonDecisionLogPretrainRunner.validateCheckpoint(
        arguments.path(0, "grpCheckpointRoot"),
        arguments.path(1, "checkpoint"),
        arguments.path(2, "logDir"),
        arguments.optionalInteger(3, "maxFiles", 0),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionLogPretrainRunner.ValidationSummary result) {
    return "checkpoint="
        + result.checkpoint()
        + " usable="
        + result.usable()
        + " assessment="
        + result.assessment()
        + " samples="
        + result.metrics().samples()
        + " policySamples="
        + result.metrics().policySamples()
        + " arenaPromotionApplied=false";
  }

  @Override
  public boolean successful(EpsilonDecisionLogPretrainRunner.ValidationSummary result) {
    return result.usable();
  }
}

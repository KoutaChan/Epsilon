package com.epsilon.nano.workflow.training;

import com.epsilon.nano.ai.decision.training.DecisionSelectedPgCampaignResult;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionSelectedPgCampaign;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;

/** 自己対局で選択した行動を使って方策を学習し、対戦評価を繰り返すコマンド。 */
public final class DecisionTrainingWorkflow
    implements CommandWorkflow<DecisionSelectedPgCampaignResult> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.TRAIN_DECISION;
  }

  @Override
  public DecisionSelectedPgCampaignResult execute(WorkflowArguments arguments) throws Exception {
    return EpsilonDecisionSelectedPgCampaign.train(
        arguments.optionalPath(0, Path.of("checkpoints/epsilon-nano")),
        arguments.settings().bind(DecisionSelectedPgCampaignSettings.class),
        arguments.settings());
  }

  @Override
  public String summarize(DecisionSelectedPgCampaignResult result) {
    return "macros="
        + result.completedMacros()
        + "/"
        + result.maximumMacros()
        + " duels="
        + result.duels().size()
        + " promotions="
        + result.promotions()
        + " status="
        + result.status()
        + " completed="
        + result.completed();
  }

  @Override
  public boolean successful(DecisionSelectedPgCampaignResult result) {
    return result.completed();
  }
}

package com.epsilon.major.workflow.training;

import com.epsilon.major.ai.decision.training.DecisionKlTrainingPlan;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** 停止中の学習状態に対して、方策更新の目標 KL ダイバージェンスとその推移を変更する。 */
public final class DecisionKlPlanWorkflow implements CommandWorkflow<String> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.PLAN_KL_DECISION;
  }

  @Override
  public String execute(WorkflowArguments args) throws Exception {
    return DecisionKlTrainingPlan.replan(
        args.path(0, "checkpointRoot"),
        args.longValue(1, "endActorOptimizerStep"),
        args.floatValue(2, "endTargetKl"),
        args.settings());
  }

  @Override
  public String summarize(String result) {
    return result;
  }
}

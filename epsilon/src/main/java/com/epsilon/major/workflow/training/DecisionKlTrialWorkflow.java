package com.epsilon.major.workflow.training;

import com.epsilon.major.ai.decision.training.DecisionKlTargetTrial;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** 専用の実験ディレクトリで、目標 KL ダイバージェンスを K と K/2 にした学習結果を対戦比較する。 */
public final class DecisionKlTrialWorkflow implements CommandWorkflow<String> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.TRIAL_KL_DECISION;
  }

  @Override
  public String execute(WorkflowArguments args) throws Exception {
    return DecisionKlTargetTrial.run(
        args.path(0, "sourceRoot"),
        args.path(1, "trialRoot"),
        args.integer(2, "totalMacrosPerArm"),
        args.longValue(3, "trainSeed"),
        args.longValue(4, "evalSeed"),
        args.settings());
  }

  @Override
  public String summarize(String result) {
    return result;
  }
}

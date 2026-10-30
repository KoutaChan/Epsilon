package com.epsilon.prism;

import com.epsilon.prism.luck.LuckFitWorkflow;
import com.epsilon.prism.run.FitWorkflow;
import com.epsilon.prism.workflow.AuditGrpWorkflow;
import com.epsilon.prism.workflow.EvaluateWorkflow;
import com.epsilon.prism.workflow.TrainWorkflow;
import com.epsilon.workflow.WorkflowRegistry;
import java.util.List;

final class Workflows {
  private Workflows() {}

  /** コマンドラインから実行できるコマンドを、ヘルプの表示順に登録する。 */
  public static WorkflowRegistry create() {
    return WorkflowRegistry.of(
        List.of(
            new TrainWorkflow(),
            new EvaluateWorkflow(),
            new FitWorkflow(),
            new LuckFitWorkflow(),
            new AuditGrpWorkflow()));
  }
}

package com.epsilon.cfr;

import com.epsilon.cfr.workflow.audit.StatusWorkflow;
import com.epsilon.cfr.workflow.audit.VerifyWorkflow;
import com.epsilon.cfr.workflow.checkpoint.ExportWorkflow;
import com.epsilon.cfr.workflow.checkpoint.InitConfigWorkflow;
import com.epsilon.cfr.workflow.training.ResumeWorkflow;
import com.epsilon.cfr.workflow.training.TrainWorkflow;
import com.epsilon.workflow.WorkflowRegistry;
import java.io.PrintStream;
import java.util.List;

final class Workflows {
  private Workflows() {}

  /** コマンドラインから実行できるコマンドを、ヘルプの表示順に登録する。 */
  public static WorkflowRegistry create(PrintStream output) {
    return WorkflowRegistry.of(
        List.of(
            new TrainWorkflow(),
            new ResumeWorkflow(),
            new InitConfigWorkflow(output),
            new StatusWorkflow(output),
            new VerifyWorkflow(output),
            new ExportWorkflow(output)));
  }
}

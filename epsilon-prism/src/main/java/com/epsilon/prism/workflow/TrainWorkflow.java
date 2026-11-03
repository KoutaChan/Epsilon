package com.epsilon.prism.workflow;

import com.epsilon.prism.run.Lineage;
import com.epsilon.prism.run.StopSignal;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.nio.file.Path;

/**
 * run を始めるか再開する（網の系譜 {@link Lineage}）。新しい run は設定の [epsilon.prism] から始め、再開は run の config.json を読む。SIGTERM を
 * 受けると次の安全点で止まって 0 で終わる（{@link StopSignal#onTerm}。合図は引数を読む前に登録する）。
 */
public final class TrainWorkflow implements CommandWorkflow<Integer> {
  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "train", WorkflowKind.TRAINING, "train --run DIR [--stop-after N]", 2, 4);
  }

  @Override
  public Integer execute(WorkflowArguments arguments) throws Exception {
    StopSignal stop = StopSignal.onTerm();
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    int stopAfter =
        Integer.parseInt(options.optionalText("stop-after", Integer.toString(Integer.MAX_VALUE)));
    options.requireEmpty();
    return Lineage.run(run, stopAfter, arguments.settings(), stop);
  }

  @Override
  public String summarize(Integer status) {
    return "status=" + status;
  }

  @Override
  public boolean successful(Integer status) {
    return status == 0;
  }
}

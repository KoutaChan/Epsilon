package com.epsilon.cfr.workflow.checkpoint;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.runtime.CfrDeployment;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;

/** 学習済み平均方策を配布形式へ書き出す。 */
public final class ExportWorkflow implements CommandWorkflow<ExportWorkflow.Result> {
  public record Result(Path directory, CfrDeployment.Manifest manifest) {}

  private final PrintStream output;

  public ExportWorkflow(PrintStream output) {
    this.output = output;
  }

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "export",
        WorkflowKind.CHECKPOINT,
        "export --run DIR --out DIR [--selection champion|average]",
        4,
        6);
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws IOException {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    Path destination = options.path("out");
    String selection = options.optionalText("selection", "champion");
    options.requireEmpty();
    CfrDeployment.Manifest manifest = CfrDeployment.export(run, destination, selection);
    output.println(CfrConfig.JSON.toJson(manifest));
    output.println(destination.toAbsolutePath().normalize());
    return new Result(destination, manifest);
  }

  @Override
  public String summarize(Result result) {
    return "deployment=" + result.directory().toAbsolutePath().normalize();
  }
}

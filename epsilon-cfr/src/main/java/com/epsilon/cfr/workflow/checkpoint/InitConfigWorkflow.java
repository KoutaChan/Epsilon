package com.epsilon.cfr.workflow.checkpoint;

import com.epsilon.cfr.config.settings.EpsilonSettings;
import com.epsilon.config.settings.SettingsFiles;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;

/** 学習設定の雛形を作成する。 */
public final class InitConfigWorkflow implements CommandWorkflow<Path> {
  private final PrintStream output;

  public InitConfigWorkflow(PrintStream output) {
    this.output = output;
  }

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.exact(
        "init-config", WorkflowKind.CHECKPOINT, "init-config --out FILE", 2);
  }

  @Override
  public Path execute(WorkflowArguments arguments) throws IOException {
    WorkflowOptions options = arguments.namedOptions();
    Path destination = options.path("out");
    options.requireEmpty();
    if (Files.exists(destination)) throw new FileAlreadyExistsException(destination.toString());
    SettingsFiles.createDefault(destination, EpsilonSettings.DEFAULT_RESOURCE);
    output.println(destination.toAbsolutePath().normalize());
    return destination;
  }

  @Override
  public String summarize(Path destination) {
    return "config=" + destination.toAbsolutePath().normalize();
  }
}

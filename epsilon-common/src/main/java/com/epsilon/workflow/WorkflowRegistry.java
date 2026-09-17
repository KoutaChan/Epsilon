package com.epsilon.workflow;

import com.epsilon.config.settings.SettingsLoader;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Epsilon CLI 処理を一意なコマンド名で登録し、共通手順で実行する。 */
public final class WorkflowRegistry {

  private static final Logger log = LoggerFactory.getLogger(WorkflowRegistry.class);

  private final Map<String, CommandWorkflow<?>> workflows;

  private WorkflowRegistry(List<? extends CommandWorkflow<?>> workflows) {
    LinkedHashMap<String, CommandWorkflow<?>> indexed = new LinkedHashMap<>();
    for (CommandWorkflow<?> workflow : workflows) {
      Objects.requireNonNull(workflow, "workflow");
      String command = workflow.definition().command();
      if (indexed.putIfAbsent(command, workflow) != null) {
        throw new IllegalArgumentException("Duplicate workflow command: " + command);
      }
    }
    this.workflows = Collections.unmodifiableMap(indexed);
  }

  /** 起動処理から渡されたコマンド一覧を登録する。コマンド名が重複している場合は例外を送出する。 */
  public static WorkflowRegistry of(List<? extends CommandWorkflow<?>> workflows) {
    return new WorkflowRegistry(workflows);
  }

  public boolean handles(String command) {
    return workflows.containsKey(command);
  }

  public List<WorkflowDefinition> definitions() {
    return workflows.values().stream().map(CommandWorkflow::definition).toList();
  }

  public List<String> usageLines() {
    return definitions().stream().map(WorkflowDefinition::usageLine).toList();
  }

  /**
   * コマンドを実行し、処理固有の成功条件を返す。
   *
   * @throws Exception 引数の変換またはコマンドの実行に失敗した場合
   */
  public boolean execute(String command, String[] rawArguments, SettingsLoader settings)
      throws Exception {
    CommandWorkflow<?> workflow = workflows.get(command);
    if (workflow == null) {
      throw new IllegalArgumentException("Unknown workflow command: " + command);
    }
    WorkflowDefinition definition = workflow.definition();
    definition.validateArgumentCount(rawArguments.length);
    WorkflowArguments arguments = new WorkflowArguments(command, rawArguments, settings);
    log.info(
        "{} workflow started: command={} arguments={}",
        definition.kind().displayName(),
        command,
        arguments.size());
    long started = System.nanoTime();
    WorkflowExecution execution = executeTyped(workflow, arguments);
    long elapsedMillis = (System.nanoTime() - started) / 1_000_000L;
    log.info(
        "{} workflow complete: command={} successful={} elapsedMs={} {}",
        definition.kind().displayName(),
        command,
        execution.successful(),
        elapsedMillis,
        execution.summary());
    return execution.successful();
  }

  private static <R> WorkflowExecution executeTyped(
      CommandWorkflow<R> workflow, WorkflowArguments arguments) throws Exception {
    R result = Objects.requireNonNull(workflow.execute(arguments), "workflow result");
    String summary = Objects.requireNonNull(workflow.summarize(result), "workflow summary");
    if (summary.isBlank()) {
      throw new IllegalStateException("Workflow summary must not be blank");
    }
    return new WorkflowExecution(workflow.successful(result), summary);
  }

  private record WorkflowExecution(boolean successful, String summary) {}
}

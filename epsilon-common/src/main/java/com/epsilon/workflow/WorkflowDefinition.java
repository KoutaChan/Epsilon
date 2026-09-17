package com.epsilon.workflow;

import java.util.Objects;

/** コマンド名、処理の分類、引数の数、使用方法を定義する。 */
public record WorkflowDefinition(
    String command, WorkflowKind kind, String usage, int minimumArguments, int maximumArguments) {

  public WorkflowDefinition {
    command = requireText(command, "command");
    kind = Objects.requireNonNull(kind, "kind");
    usage = requireText(usage, "usage");
    if (!usage.startsWith(command)) {
      throw new IllegalArgumentException("Workflow usage must start with its command: " + usage);
    }
    if (minimumArguments < 0 || maximumArguments < minimumArguments) {
      throw new IllegalArgumentException(
          "Invalid workflow argument range: " + minimumArguments + ".." + maximumArguments);
    }
  }

  public static WorkflowDefinition exact(
      String command, WorkflowKind kind, String usage, int arguments) {
    return new WorkflowDefinition(command, kind, usage, arguments, arguments);
  }

  public static WorkflowDefinition range(
      String command, WorkflowKind kind, String usage, int minimumArguments, int maximumArguments) {
    return new WorkflowDefinition(command, kind, usage, minimumArguments, maximumArguments);
  }

  public static WorkflowDefinition atLeast(
      String command, WorkflowKind kind, String usage, int minimumArguments) {
    return new WorkflowDefinition(command, kind, usage, minimumArguments, Integer.MAX_VALUE);
  }

  void validateArgumentCount(int actual) {
    if (actual >= minimumArguments && actual <= maximumArguments) {
      return;
    }
    String expectation;
    if (minimumArguments == maximumArguments) {
      expectation = "requires exactly " + minimumArguments;
    } else if (maximumArguments == Integer.MAX_VALUE) {
      expectation = "requires at least " + minimumArguments;
    } else if (minimumArguments == 0) {
      expectation = "accepts at most " + maximumArguments;
    } else {
      expectation = "accepts between " + minimumArguments + " and " + maximumArguments;
    }
    throw new IllegalArgumentException(
        command + " " + expectation + " argument(s), but received " + actual + ". Usage: " + usage);
  }

  String usageLine() {
    return "  " + usage;
  }

  private static String requireText(String value, String name) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }
}

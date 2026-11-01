package com.epsilon.workflow;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 名前付き引数を取り出し、コマンドで使用しなかったオプションを検出する。 */
public final class WorkflowOptions {
  private final String command;
  private final Map<String, String> remaining;

  private WorkflowOptions(String command, Map<String, String> remaining) {
    this.command = command;
    this.remaining = remaining;
  }

  static WorkflowOptions parse(String command, List<String> arguments) {
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < arguments.size(); i += 2) {
      String option = arguments.get(i);
      if (!option.startsWith("--") || option.length() == 2 || option.substring(2).isBlank()) {
        throw new IllegalArgumentException(command + " expected --name value, received: " + option);
      }
      if (i + 1 >= arguments.size()
          || arguments.get(i + 1).startsWith("--")
          || arguments.get(i + 1).isBlank()) {
        throw new IllegalArgumentException(command + " requires a value for " + option);
      }
      String name = option.substring(2);
      if (values.putIfAbsent(name, arguments.get(i + 1)) != null) {
        throw new IllegalArgumentException(command + " repeats option " + option);
      }
    }
    return new WorkflowOptions(command, values);
  }

  /** 必須のパス引数を取り出す。 */
  public Path path(String name) {
    String value = remaining.remove(name);
    if (value == null) {
      throw new IllegalArgumentException(command + " requires --" + name);
    }
    return Path.of(value);
  }

  /** オプションを取り出す。指定されなかった場合は既定値を返す。 */
  public String optionalText(String name, String fallback) {
    String value = remaining.remove(name);
    return value == null ? fallback : value;
  }

  /** 処理を開始する前に、未使用のオプションがないことを確認する。 */
  public void requireEmpty() {
    if (!remaining.isEmpty()) {
      throw new IllegalArgumentException(command + " has unknown options: " + remaining.keySet());
    }
  }
}

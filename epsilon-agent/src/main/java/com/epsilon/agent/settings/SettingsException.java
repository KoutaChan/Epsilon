package com.epsilon.agent.settings;

import java.util.List;

/** 設定のエラー。エラーは 1 行に 1 つずつ英語で書く。record に結び付けるときは、見つけたエラーを全て集めてから投げる。 */
public final class SettingsException extends IllegalArgumentException {

  private final List<String> problems;

  /** @param problems エラーメッセージ（1 つ以上） */
  public SettingsException(List<String> problems) {
    super(String.join("\n", problems));
    if (problems.isEmpty()) {
      throw new IllegalArgumentException("A settings error needs at least one problem");
    }
    this.problems = List.copyOf(problems);
  }

  /** エラーメッセージ。見つけた順に並ぶ。 */
  public List<String> problems() {
    return problems;
  }
}

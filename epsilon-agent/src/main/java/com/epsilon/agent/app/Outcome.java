package com.epsilon.agent.app;

import java.util.Objects;

/**
 * コマンドがどう終わったか。
 *
 * @param exit 終了の種類
 * @param summary 終わりのログの行に足す短い要約（英語）
 */
public record Outcome(Exit exit, String summary) {

  public Outcome {
    Objects.requireNonNull(exit, "exit");
    if (summary == null || summary.isBlank()) {
      throw new IllegalArgumentException("An outcome needs a summary");
    }
  }

  /** 頼まれた処理を終えたときの結果を作る。 */
  public static Outcome ok(String summary) {
    return new Outcome(Exit.OK, summary);
  }
}

package com.epsilon.agent.run;

import java.time.Instant;
import java.util.Objects;

/**
 * 反復を始めたときに固定した設定の行と jar（work の中の ITERATION.json）。途中で止めて続けるときも、反復は最後までこの行で回る。
 *
 * @param iteration 反復の番号
 * @param settingsRow 固定した設定の行
 * @param jar 固定した jar の ID
 * @param started 反復を初めて固定した時刻
 */
public record IterationFile(int iteration, int settingsRow, String jar, Instant started) {

  public IterationFile {
    if (iteration < 1 || settingsRow < 1) {
      throw new IllegalArgumentException(
          "An iteration and its settings row start at 1: iteration " + iteration + ", row " + settingsRow);
    }
    Objects.requireNonNull(jar, "jar");
    Objects.requireNonNull(started, "started");
  }
}

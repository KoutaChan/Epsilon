package com.epsilon.workflow;

import java.util.Locale;

/** CLIコマンドの処理の種類。 */
public enum WorkflowKind {
  CHECKPOINT,
  TRAINING,
  VALIDATION,
  EVALUATION,
  PLAY,
  BENCHMARK,
  PROBE,
  AUDIT,
  PROMOTION;

  String displayName() {
    return name().toLowerCase(Locale.ROOT);
  }
}

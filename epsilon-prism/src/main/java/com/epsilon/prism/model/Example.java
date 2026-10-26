package com.epsilon.prism.model;

import com.epsilon.nano.ai.decision.input.DecisionHostBatch;

/** 学習の 1 行（{@link LabelExample}・{@link DistillExample}）。入力と、損失（{@link Objective}）が読む教師。 */
public interface Example {
  /** 1 行の入力（保存した payload を復号したもの）。 */
  DecisionHostBatch host();
}

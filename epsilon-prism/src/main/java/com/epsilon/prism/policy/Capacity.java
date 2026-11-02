package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import java.util.List;

/** 受付のバッチ区分。遷移を複数持つ合法手（鳴き）を含む行は入力が大きいので行数の上限を分ける。 */
public enum Capacity {
  SINGLE_TRANSITION,
  MULTIPLE_TRANSITIONS;

  public static Capacity of(List<Action> legal) {
    return DecisionBatchBuilder.selectInferenceBucket(legal).actionTransitionCapacity() > 1
        ? MULTIPLE_TRANSITIONS
        : SINGLE_TRANSITION;
  }
}

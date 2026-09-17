package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.DecisionLearningRole;
import java.util.AbstractList;
import java.util.Objects;
import java.util.RandomAccess;

/** 行動を局面へ反映した結果を保持する、エンジン所有の再利用バッファ。 */
final class EngineOutcomeBuffer extends AbstractList<EngineDecisionOutcome>
    implements RandomAccess {

  private static final int MAX_OUTCOMES = 3;

  private final EngineDecisionOutcome[] outcomes = new EngineDecisionOutcome[MAX_OUTCOMES];
  private int size;

  EngineOutcomeBuffer() {
    for (int index = 0; index < outcomes.length; index++) {
      outcomes[index] = new EngineDecisionOutcome();
    }
  }

  void append(
      long decisionId,
      int player,
      Action selectedAction,
      boolean selectedActionWasExecuted,
      DecisionLearningRole learningRole) {
    if (size == outcomes.length) {
      throw new IllegalStateException("outcome capacity exceeded: " + outcomes.length);
    }
    outcomes[size++].bind(
        decisionId, player, selectedAction, selectedActionWasExecuted, learningRole);
  }

  @Override
  public EngineDecisionOutcome get(int index) {
    Objects.checkIndex(index, size);
    return outcomes[index];
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public void clear() {
    for (int index = 0; index < size; index++) {
      outcomes[index].clear();
    }
    size = 0;
  }
}

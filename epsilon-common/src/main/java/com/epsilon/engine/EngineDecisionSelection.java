package com.epsilon.engine;

import com.epsilon.core.Action;

/**
 * 返却済みの {@link EngineDecisionPoint} に対して選択した行動。
 *
 * <dl>
 *   <dt>{@code decisionId}
 *   <dd>選択元判断時点のID
 *   <dt>{@code action}
 *   <dd>その時点の合法集合から選んだ行動
 * </dl>
 */
public final class EngineDecisionSelection {

  private long decisionId;
  private Action action;

  EngineDecisionSelection() {}

  public EngineDecisionSelection(long decisionId, Action action) {
    bind(decisionId, action);
  }

  void bind(long decisionId, Action action) {
    this.decisionId = decisionId;
    this.action = action;
  }

  void clear() {
    decisionId = 0L;
    action = null;
  }

  public long decisionId() {
    return decisionId;
  }

  public Action action() {
    return action;
  }
}

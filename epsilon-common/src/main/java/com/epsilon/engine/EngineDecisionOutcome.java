package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.core.GameState;

/**
 * 選択した行動の実行結果と、その判断を方策・価値の学習に使うかを表す。エンジンが再利用するため、次の行動反映を超えて保持してはならない。
 *
 * <p>実際に行動が実行されなかった場合でも、別の合法手を選べば応答結果を変えられたなら方策学習の対象となる。例えば他家のロンによってポンが実行されなかった場合でも、自家もロンを選べたなら、この判断は方策学習の対象となる。
 *
 * <dl>
 *   <dt>{@code decisionId}
 *   <dd>対応する判断時点の一意ID
 *   <dt>{@code player}
 *   <dd>選択した席番号（0-3）
 *   <dt>{@code selectedAction}
 *   <dd>プレイヤーが提出した行動
 *   <dt>{@code selectedActionWasExecuted}
 *   <dd>選択行動自体が応答優先順位の解決後に実行されたならtrue
 *   <dt>{@code learningRole}
 *   <dd>この判断を方策と価値の学習にどう使うか
 * </dl>
 */
public final class EngineDecisionOutcome {

  private long decisionId;
  private int player;
  private Action selectedAction;
  private boolean selectedActionWasExecuted;
  private DecisionLearningRole learningRole;

  EngineDecisionOutcome() {}

  void bind(
      long decisionId,
      int player,
      Action selectedAction,
      boolean selectedActionWasExecuted,
      DecisionLearningRole learningRole) {
    if (decisionId <= 0L) {
      throw new IllegalArgumentException("decisionId must be positive");
    }
    if (player < 0 || player >= GameState.NUM_PLAYERS) {
      throw new IllegalArgumentException("player must be 0-3");
    }
    if (learningRole == null) {
      throw new IllegalArgumentException("learningRole must not be null");
    }
    this.decisionId = decisionId;
    this.player = player;
    this.selectedAction = selectedAction;
    this.selectedActionWasExecuted = selectedActionWasExecuted;
    this.learningRole = learningRole;
  }

  void clear() {
    decisionId = 0L;
    player = -1;
    selectedAction = null;
    selectedActionWasExecuted = false;
    learningRole = null;
  }

  public long decisionId() {
    return decisionId;
  }

  public int player() {
    return player;
  }

  public Action selectedAction() {
    return selectedAction;
  }

  public boolean selectedActionWasExecuted() {
    return selectedActionWasExecuted;
  }

  public DecisionLearningRole learningRole() {
    return learningRole;
  }
}

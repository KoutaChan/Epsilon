package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.replay.ReplayDecision.Choice;
import com.epsilon.replay.ReplayEvent.None;
import java.util.List;

/**
 * 観測者への出口。局面の改訂を数え、牌譜の手を合法手で確かめてから判断を渡す。改訂は通知する出来事ごと（明示の見送りを除く）と、出来事の通知の外の局面の変化（{@link #changed}）で
 * 1 つずつ進む。
 */
final class ReplayFeed {
  private final ReplayObserver observer;
  private final GameState state;
  private int revision;

  ReplayFeed(ReplayObserver observer, GameState state) {
    this.observer = observer;
    this.state = state;
  }

  /** 出来事を通知する。None は局面を変えないので改訂を進めない。 */
  void applied(int index, ReplayEvent event) {
    if (!(event instanceof None)) revision++;
    observer.onEventApplied(index, revision, event, state);
  }

  /** 出来事の通知の外で局面が変わった（ロンの見逃しの一時振聴、加槓の窓を閉じた）。 */
  void changed() {
    revision++;
  }

  /** @throws IllegalStateException action が legal に無い（観測者へ渡す前に落とす） */
  void recorded(
      int seat,
      EngineDecisionKind kind,
      Action action,
      List<Action> legal,
      int causeIndex,
      int resolveIndex) {
    decide(
        seat,
        kind,
        Choice.RECORDED_ACTION,
        slotOf(seat, action, legal),
        legal,
        causeIndex,
        resolveIndex);
  }

  void passed(
      int seat, EngineDecisionKind kind, List<Action> legal, int causeIndex, int resolveIndex) {
    decide(
        seat,
        kind,
        Choice.CONFIRMED_PASS,
        slotOf(seat, Action.pass(), legal),
        legal,
        causeIndex,
        resolveIndex);
  }

  void unobserved(
      int seat, EngineDecisionKind kind, List<Action> legal, int causeIndex, int resolveIndex) {
    decide(seat, kind, Choice.UNOBSERVED, -1, legal, causeIndex, resolveIndex);
  }

  private void decide(
      int seat,
      EngineDecisionKind kind,
      Choice choice,
      int chosenSlot,
      List<Action> legal,
      int causeIndex,
      int resolveIndex) {
    observer.onDecision(
        new ReplayDecision(seat, kind, choice, chosenSlot, causeIndex, resolveIndex, revision),
        state,
        legal);
  }

  private static int slotOf(int seat, Action action, List<Action> legal) {
    int id = action.toIndex();
    for (int slot = 0; slot < legal.size(); slot++)
      if (legal.get(slot).toIndex() == id) return slot;
    throw new IllegalStateException(
        "Replay action is not legal: player=" + seat + " action=" + action);
  }
}

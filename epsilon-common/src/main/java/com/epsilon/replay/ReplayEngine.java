package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.replay.DecisionPoint.ChoiceKind;
import com.epsilon.replay.ReplayEvent.*;
import java.util.Arrays;
import java.util.List;

/** 正規化済みの一対局を再生する。局面と応答の受け付け状態を管理し、用途に依存しない同期通知を送る。 */
public final class ReplayEngine {
  private final List<ReplayEvent> events;
  private final ReplayObserver observer;
  private final ReplayState state;
  private final ResponseWindow responseWindow = new ResponseWindow();
  private final ResponseChoices responseChoices = new ResponseChoices();
  private ResponseWindow pendingWindow;
  private int currentEventIndex;
  private int stateId;
  private int turnCauseEventIndex = -1;

  private ReplayEngine(ReplayRecord record, ReplayObserver observer) {
    events = record.events();
    this.observer = observer;
    state = new ReplayState(record.walls());
  }

  /** イベントと牌山を読み取り専用で借用して再生する。通知中にだけ借用状態を読める。 返却する最終得点配列は所有権を呼び出し側へ移し、以後変更しない。 */
  public static int[] replay(ReplayRecord record, ReplayObserver observer) {
    ReplayEngine engine = new ReplayEngine(record, observer);
    engine.replayEvents();
    return engine.state.finalScores();
  }

  private void replayEvents() {
    for (int index = 0; index < events.size(); ) {
      if (pendingWindow != null) {
        index = resolveResponses(index);
        continue;
      }
      currentEventIndex = index;
      applyEvent(events.get(index));
      notifyEvent(index);
      index++;
    }
    if (pendingWindow != null) {
      currentEventIndex = events.size();
      finishUnresolvedResponses(responseChoices.reset(), false);
      closeResponses(false);
    }
  }

  private void notifyEvent(int index) {
    observer.onEventApplied(index, stateId, events.get(index), state.borrowedState());
  }

  private void applyEvent(ReplayEvent event) {
    switch (event) {
      case StartGame ignored -> state.startMatch();
      case StartKyoku start -> {
        state.startRound(start);
        turnCauseEventIndex = currentEventIndex;
      }
      case Tsumo draw -> {
        state.applyDraw(draw);
        turnCauseEventIndex = currentEventIndex;
      }
      case Dahai discard -> {
        Action action = state.discardAction(discard);
        emitRecordedAction(
            discard.actor(), action, state.discardActions(discard.actor()), turnCauseEventIndex);
        state.applyDiscard(discard, action);
        openResponses(discard.actor(), false);
      }
      case Ankan kan -> {
        emitRecordedAction(
            kan.actor(),
            Action.ankan(Tile.typeOf(kan.consumedPhysicalTileIds()[0])),
            state.turnActions(kan.actor()),
            turnCauseEventIndex);
        state.applyClosedKan(kan);
      }
      case Kakan kan -> {
        emitRecordedAction(
            kan.actor(),
            Action.kakan(Tile.typeOf(kan.addedPhysicalTileId())),
            state.turnActions(kan.actor()),
            turnCauseEventIndex);
        state.applyAddedKan(kan);
        openResponses(kan.actor(), true);
      }
      case Reach reach -> state.applyRiichiDeclaration(reach);
      case ReachAccepted accepted -> state.applyRiichiPayment(accepted);
      case Dora dora -> state.revealDora(dora);
      case Hora win -> {
        if (win.actor() != win.target())
          throw new IllegalStateException("Ron event outside response window: " + win);
        emitRecordedAction(
            win.actor(), Action.tsumoAgari(), state.turnActions(win.actor()), turnCauseEventIndex);
        state.settleWin(win);
      }
      case Ryukyoku draw -> {
        if ("NINE_TERMINALS".equals(draw.reason())) {
          int player = state.borrowedState().currentPlayer();
          emitRecordedAction(
              player, Action.kyushuKyuhai(), state.turnActions(player), turnCauseEventIndex);
        }
        state.settleDraw(draw);
      }
      case EndKyoku ignored -> state.finishRound();
      case EndGame end -> state.finishMatch(end);
      case None ignored -> {
        return;
      }
      case Chi ignored -> throw responseOutsideWindow(event);
      case Pon ignored -> throw responseOutsideWindow(event);
      case Daiminkan ignored -> throw responseOutsideWindow(event);
    }
    stateId++;
  }

  private void openResponses(int sourcePlayer, boolean addedKan) {
    pendingWindow =
        addedKan
            ? responseWindow.openKakan(sourcePlayer, currentEventIndex)
            : responseWindow.openDahai(sourcePlayer, currentEventIndex);
    state.collectResponses(pendingWindow);
    if (pendingWindow.isEmpty()) closeResponses(false);
  }

  private int resolveResponses(int index) {
    currentEventIndex = index;
    ReplayEvent current = events.get(index);
    if (pendingWindow.preservesWindow(current)) {
      applyEvent(current);
      notifyEvent(index);
      return index + 1;
    }
    if (!pendingWindow.accepts(current)) {
      finishUnresolvedResponses(responseChoices.reset(), true);
      closeResponses(false);
      return index;
    }

    ResponseChoices choices = responseChoices.reset();
    int end = index;
    while (end < events.size() && pendingWindow.accepts(events.get(end))) {
      ReplayEvent response = events.get(end);
      choices.record(pendingWindow.actorOf(response), pendingWindow.isExplicitPass(response));
      end++;
    }
    boolean confirmedWithoutAction = end < events.size() && !choices.hasSelectedAction();
    finishUnresolvedResponses(choices, confirmedWithoutAction);

    boolean robbedByRon = false;
    for (int i = index; i < end; i++) {
      currentEventIndex = i;
      ReplayEvent response = events.get(i);
      if (pendingWindow.isExplicitPass(response)) {
        notifyEvent(i);
        continue;
      }
      robbedByRon |= pendingWindow.isRon(response);
      applyResponse(response);
      if (!(response instanceof Hora)) notifyEvent(i);
    }
    // 複数ロンは全員の適用前判断を通知してから、個々の得点増減を一度ずつ適用する。
    for (int i = index; i < end; i++) {
      if (events.get(i) instanceof Hora win) {
        state.settleWin(win);
        stateId++;
        notifyEvent(i);
      }
    }
    closeResponses(robbedByRon);
    return end;
  }

  private void finishUnresolvedResponses(ResponseChoices choices, boolean inferUnobservedPasses) {
    for (int player = 0; player < 4; player++) {
      if (!pendingWindow.hasPendingPlayer(player)) continue;
      boolean confirmedPass =
          choices.explicitlyPassed(player)
              || inferUnobservedPasses && !choices.selectedAction(player);
      if (pendingWindow.hasLegalDecision(player)) {
        List<Action> legal = pendingWindow.legalActions(player);
        if (confirmedPass) {
          emitDecision(
              player,
              Action.pass(),
              legal,
              pendingWindow.causeEventIndex(),
              ChoiceKind.CONFIRMED_PASS);
        } else if (!choices.selectedAction(player)) {
          observer.onDecision(
              new DecisionPoint(
                  currentEventIndex,
                  stateId,
                  pendingWindow.causeEventIndex(),
                  -1,
                  ChoiceKind.UNOBSERVED),
              state.borrowedState(),
              player,
              legal);
        }
      }
      if (pendingWindow.createsFuritenWhenDeclined(player) && !choices.selectedAction(player)) {
        enterTemporaryFuriten(player);
      }
    }
  }

  private void applyResponse(ReplayEvent event) {
    int actor = pendingWindow.actorOf(event);
    boolean declinedRon =
        !pendingWindow.isRon(event) && pendingWindow.createsFuritenWhenDeclined(actor);
    List<Action> legal = pendingWindow.legalActions(actor);
    int cause = pendingWindow.causeEventIndex();
    switch (event) {
      case Hora ignored -> {
        emitRecordedAction(actor, Action.ronAgari(), legal, cause);
        return;
      }
      case Chi chi -> {
        emitRecordedAction(actor, ReplayState.chiAction(chi), legal, cause);
        if (declinedRon) enterTemporaryFuriten(actor);
        state.applyChi(chi);
      }
      case Pon pon -> {
        emitRecordedAction(actor, ReplayState.ponAction(pon), legal, cause);
        if (declinedRon) enterTemporaryFuriten(actor);
        state.applyPon(pon);
      }
      case Daiminkan kan -> {
        emitRecordedAction(
            actor, Action.daiminkan(Tile.typeOf(kan.calledPhysicalTileId())), legal, cause);
        if (declinedRon) enterTemporaryFuriten(actor);
        state.applyOpenKan(kan);
      }
      default -> throw new IllegalStateException("Unexpected response event: " + event);
    }
    turnCauseEventIndex = currentEventIndex;
    stateId++;
  }

  private void enterTemporaryFuriten(int player) {
    if (state.enterTemporaryFuriten(player)) stateId++;
  }

  private void closeResponses(boolean robbedByRon) {
    if (state.completeResponses(pendingWindow, robbedByRon)) stateId++;
    pendingWindow = null;
  }

  private void emitRecordedAction(int player, Action action, List<Action> legal, int cause) {
    emitDecision(player, action, legal, cause, ChoiceKind.RECORDED_ACTION);
  }

  private void emitDecision(
      int player, Action action, List<Action> legal, int cause, ChoiceKind kind) {
    int actionId = action.toIndex();
    for (int slot = 0; slot < legal.size(); slot++) {
      if (legal.get(slot).toIndex() == actionId) {
        observer.onDecision(
            new DecisionPoint(currentEventIndex, stateId, cause, slot, kind),
            state.borrowedState(),
            player,
            legal);
        return;
      }
    }
    throw new IllegalStateException(
        "Replay action is not legal: player=" + player + " action=" + action);
  }

  private static IllegalStateException responseOutsideWindow(ReplayEvent event) {
    return new IllegalStateException("Response event outside response window: " + event);
  }

  private static final class ResponseChoices {
    private enum Choice {
      UNOBSERVED,
      PASS,
      ACTION
    }

    private final Choice[] byPlayer = new Choice[4];

    private ResponseChoices reset() {
      Arrays.fill(byPlayer, Choice.UNOBSERVED);
      return this;
    }

    private void record(int player, boolean pass) {
      byPlayer[player] = pass ? Choice.PASS : Choice.ACTION;
    }

    private boolean explicitlyPassed(int player) {
      return byPlayer[player] == Choice.PASS;
    }

    private boolean selectedAction(int player) {
      return byPlayer[player] == Choice.ACTION;
    }

    private boolean hasSelectedAction() {
      for (Choice choice : byPlayer) if (choice == Choice.ACTION) return true;
      return false;
    }
  }
}

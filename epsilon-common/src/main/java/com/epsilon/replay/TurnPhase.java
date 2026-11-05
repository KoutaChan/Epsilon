package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.Meld;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.EndKyoku;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.Reach;
import com.epsilon.replay.ReplayEvent.ReachAccepted;
import com.epsilon.replay.ReplayEvent.Ryukyoku;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import java.util.List;

/** 窓が開いていない間の出来事を 1 つずつ適用する。手番の判断はその出来事で局面を変える前に出し、打牌と加槓は応答の窓を開く。 */
final class TurnPhase {
  private final List<ReplayEvent> events;
  private final ReplayTable table;
  private final LegalActions legal;
  private final ReplayFeed feed;
  private final ResponsePhase responses;

  /** 次の出来事（その席の嶺上牌の打牌）の後に開く大明槓・加槓の新ドラ。その打牌が開いて消す。無ければ null。 */
  private Dora kanDoraAfterDiscard;

  TurnPhase(
      List<ReplayEvent> events,
      ReplayTable table,
      LegalActions legal,
      ReplayFeed feed,
      ResponsePhase responses) {
    this.events = events;
    this.table = table;
    this.legal = legal;
    this.feed = feed;
    this.responses = responses;
  }

  /** @return 次に読む出来事の位置 */
  int apply(int index) {
    ReplayEvent event = events.get(index);
    ReplayEvent applied = event;
    switch (event) {
      case StartGame ignored -> table.startMatch();
      case StartKyoku start -> table.startRound(index, start);
      case Tsumo draw -> table.draw(index, draw);
      case Dahai discard -> discard(index, discard);
      case Ankan kan -> {
        turnDecision(index, kan.actor(), RecordedActions.ankan(kan));
        table.ankan(kan);
      }
      case Kakan kan -> {
        turnDecision(index, kan.actor(), RecordedActions.kakan(kan));
        responses.open(table.declareKakan(kan), index);
      }
      case Reach reach -> table.declareRiichi(reach.actor());
      case ReachAccepted accepted -> table.payRiichi(accepted.actor());
      case Dora dora -> {
        if (waitsForNextDiscard(index)) kanDoraAfterDiscard = dora;
        else table.revealDora(dora);
      }
      case Hora win -> {
        if (win.isRon())
          throw new IllegalStateException("Ron event outside response window: " + win);
        turnDecision(index, win.actor(), Action.tsumoAgari());
        applied = table.settleWin(win);
      }
      case Ryukyoku ryukyoku -> {
        if (ryukyoku.isNineTerminals())
          turnDecision(index, table.state().currentPlayer(), Action.kyushuKyuhai());
        table.settleRyukyoku(ryukyoku);
      }
      case EndKyoku ignored -> {}
      case EndGame end -> table.endMatch(end);
      case None ignored -> {}
      case Chi ignored -> throw outsideWindow(event);
      case Pon ignored -> throw outsideWindow(event);
      case Daiminkan ignored -> throw outsideWindow(event);
    }
    feed.applied(index, applied);
    return index + 1;
  }

  private void discard(int index, Dahai discard) {
    int seat = discard.actor();
    Action action = RecordedActions.discard(discard, table.riichiDeclared(seat));
    Meld call = table.pendingCall();
    if (call != null)
      feed.recorded(
          seat,
          EngineDecisionKind.POST_CALL_DAHAI,
          action,
          legal.afterCall(seat, call),
          table.turnCause(),
          index);
    else
      feed.recorded(
          seat,
          EngineDecisionKind.TURN,
          action,
          legal.discard(seat, table.requireDraw(index), table.riichiDeclared(seat)),
          table.turnCause(),
          index);
    TurnEvent.Discard origin = table.discard(discard, action);
    if (kanDoraAfterDiscard != null) {
      table.revealDora(kanDoraAfterDiscard);
      kanDoraAfterDiscard = null;
    }
    responses.open(origin, index);
  }

  /** 位置 index のドラ表示が、次の出来事である嶺上牌の打牌の後に開く大明槓・加槓の新ドラか（その打牌の判断は開く前を、打牌の通知と応答は開いた後を見る）。 */
  private boolean waitsForNextDiscard(int index) {
    return index + 1 < events.size()
        && events.get(index + 1) instanceof Dahai next
        && table.kanDoraWaitsForDiscard(next.actor());
  }

  private void turnDecision(int index, int seat, Action action) {
    feed.recorded(
        seat,
        EngineDecisionKind.TURN,
        action,
        legal.turn(seat, table.requireDraw(index)),
        table.turnCause(),
        index);
  }

  private static IllegalStateException outsideWindow(ReplayEvent event) {
    return new IllegalStateException("Response event outside response window: " + event);
  }
}

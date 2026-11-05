package com.epsilon.reviewer.engine.scenario;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.TurnEvent;
import com.epsilon.replay.ReplayDecision;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.replay.ReplayObserver;
import com.epsilon.replay.ReplayRecord;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.LocatedDecision.Kind;
import com.epsilon.reviewer.engine.view.TableProjector;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * 解析の 1 判断を探す。牌譜のその局だけを通常の全情報の再生で進め（局は開始点を持つので、局だけの再生でも状態は同じ）、席と出来事の位置の
 * 合う最初の判断を写す。
 */
public final class DecisionLocator {
  private DecisionLocator() {}

  /**
   * @param round 局の位置
   * @param eventIndex 判断が解く出来事の位置
   * @throws ReviewerException decision_not_found
   */
  public static LocatedDecision find(LoadedRecord record, int round, int eventIndex, int seat) {
    if (round < 0 || round >= record.roundStarts().length) throw notFound();
    int start = record.roundStarts()[round];
    int end = record.roundEnd(round);
    if (eventIndex <= start || eventIndex > end) throw notFound();
    ReplayRecord replay = record.replay();
    int[][] walls =
        replay.walls() != null && round < replay.walls().length
            ? new int[][] {replay.walls()[round]}
            : null;
    var search = new Search(eventIndex - start, seat);
    ReplayEngine.replay(
        new ReplayRecord(
            replay.events().subList(start, end), replay.metadata(), walls, replay.completion()),
        search);
    if (search.table == null) throw notFound();
    return new LocatedDecision(
        start,
        end,
        eventIndex,
        seat,
        search.kind,
        search.table,
        search.recorded,
        search.concealed,
        search.drawn);
  }

  private static ReviewerException notFound() {
    return new ReviewerException(ErrorCode.DECISION_NOT_FOUND);
  }

  /** 局の再生から判断を写し、判断の席の手を記録の物理牌で追う（再生の手は同じ牌の写しを区別しない）。 */
  private static final class Search implements ReplayObserver {
    private final int target;
    private final int seat;

    /** 判断の席の手にある物理牌（手に入った順）。 */
    private final Set<Integer> hand = new LinkedHashSet<>();

    private Kind kind;
    private TableState table;
    private Action recorded;
    private List<Integer> concealed;
    private int drawn = -1;

    Search(int target, int seat) {
      this.target = target;
      this.seat = seat;
    }

    @Override
    public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
      if (table != null || decision.seat() != seat || decision.resolveIndex() != target) return;
      kind =
          switch (decision.kind()) {
            case RESPONSE, CHANKAN -> Kind.RESPONSE;
            case TURN, POST_CALL_DAHAI -> Kind.TURN;
          };
      table = TableProjector.table(state);
      recorded = decision.chosenSlot() < 0 ? null : legal.get(decision.chosenSlot());
      if (table.players().get(seat).draw() != null)
        drawn = ((TurnEvent.Draw) state.turnEvent()).physicalTileId();
      concealed = hand.stream().filter(id -> id != drawn).toList();
    }

    @Override
    public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
      if (table != null) return;
      switch (event) {
        case StartKyoku start -> {
          for (int id : start.initialHandPhysicalTileIds()[seat]) hand.add(id);
        }
        case Tsumo draw when draw.actor() == seat -> hand.add(draw.physicalTileId());
        case Dahai discard when discard.actor() == seat -> hand.remove(discard.physicalTileId());
        case Chi call when call.actor() == seat -> remove(call.consumedPhysicalTileIds());
        case Pon call when call.actor() == seat -> remove(call.consumedPhysicalTileIds());
        case Daiminkan call when call.actor() == seat -> remove(call.consumedPhysicalTileIds());
        case Ankan kan when kan.actor() == seat -> remove(kan.consumedPhysicalTileIds());
        case Kakan kan when kan.actor() == seat -> hand.remove(kan.addedPhysicalTileId());
        default -> {}
      }
    }

    private void remove(int[] physicalTileIds) {
      for (int physicalTileId : physicalTileIds) hand.remove(physicalTileId);
    }
  }
}

package com.epsilon.reviewer.engine.review;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.replay.DecisionPoint;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayObserver;
import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.view.EventKind;
import com.epsilon.reviewer.engine.view.OutcomeProjector;
import com.epsilon.reviewer.engine.view.TableProjector;
import com.epsilon.reviewer.engine.view.TableState;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.concurrent.CancellationException;

/** 牌譜の全情報の再生を観測し、全ての判断を評価して手順を文書へ流す。推論の失敗を作った確率で隠さない。 */
public final class Analysis {
  private Analysis() {}

  /**
   * 牌譜を再生し全ての判断を評価して書き、文書を閉じる（下の流れは閉じない）。start_game と最初の局より前の出来事は手順にしない。
   *
   * @throws CancellationException 取り消されたか、スレッドが割り込まれた
   */
  public static void run(
      LoadedRecord record, ReviewModel model, JobControl control, ResultWriter writer)
      throws IOException {
    var observer = new Observer(record.replay().events().size(), model, control, writer);
    try {
      ReplayEngine.replay(record.replay(), observer);
    } catch (UncheckedIOException failure) {
      throw failure.getCause();
    }
    observer.requireNotCancelled();
    writer.finish(observer.valued ? Double.valueOf(model.temperature()) : null);
  }

  private static final class Observer implements ReplayObserver {
    private final int events;
    private final ReviewModel model;
    private final JobControl control;
    private final ResultWriter writer;
    private boolean inRound;

    /** 最後に書いた手順の局面の改訂。 */
    private int stateId = -1;

    private boolean valued;

    Observer(int events, ReviewModel model, JobControl control, ResultWriter writer) {
      this.events = events;
      this.model = model;
      this.control = control;
      this.writer = writer;
    }

    @Override
    public void onDecision(DecisionPoint point, GameState state, int player, List<Action> legal) {
      requireNotCancelled();
      EvaluatedDecision decision =
          DecisionEvaluator.evaluate(model, state, player, legal, point.chosenSlot());
      valued = decision.valued();
      try {
        writer.decision(
            player,
            point.causeEventIndex(),
            point.eventIndex(),
            decision,
            changed(point.stateId(), state));
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
    }

    @Override
    public void onEventApplied(int eventIndex, int stateId, ReplayEvent event, GameState state) {
      requireNotCancelled();
      if (event instanceof StartGame) return;
      try {
        if (event instanceof StartKyoku) {
          writer.startRound(
              state.roundIndex(),
              state.honba(),
              state.dealer(),
              eventIndex,
              TableProjector.table(state));
          this.stateId = stateId;
          inRound = true;
          writer.event(eventIndex, EventKind.START_KYOKU, -1, null, null);
        } else if (inRound) {
          writer.event(
              eventIndex,
              EventKind.of(event),
              EventKind.actor(event),
              changed(stateId, state),
              OutcomeProjector.of(event, state));
        }
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
      control.progress(Math.min(99, (eventIndex + 1) * 100 / events));
    }

    /** 局面の改訂が変わった時だけ局面を写す。 */
    private TableState changed(int id, GameState state) {
      if (id == stateId) return null;
      stateId = id;
      return TableProjector.table(state);
    }

    void requireNotCancelled() {
      if (control.cancelled() || Thread.currentThread().isInterrupted())
        throw new CancellationException("Analysis was cancelled.");
    }
  }
}

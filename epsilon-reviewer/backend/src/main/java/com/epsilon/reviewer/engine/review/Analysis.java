package com.epsilon.reviewer.engine.review;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.replay.ReplayDecision;
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
    private int revision = -1;

    private boolean valued;

    Observer(int events, ReviewModel model, JobControl control, ResultWriter writer) {
      this.events = events;
      this.model = model;
      this.control = control;
      this.writer = writer;
    }

    @Override
    public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
      requireNotCancelled();
      EvaluatedDecision evaluated =
          DecisionEvaluator.evaluate(model, state, decision.seat(), legal, decision.chosenSlot());
      valued = evaluated.valued();
      try {
        writer.decision(
            decision.seat(),
            decision.causeIndex(),
            decision.resolveIndex(),
            evaluated,
            changed(decision.revision(), state));
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
    }

    @Override
    public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
      requireNotCancelled();
      if (event instanceof StartGame) return;
      try {
        if (event instanceof StartKyoku) {
          writer.startRound(
              state.roundIndex(),
              state.honba(),
              state.dealer(),
              index,
              TableProjector.table(state));
          this.revision = revision;
          inRound = true;
          writer.event(index, EventKind.START_KYOKU, -1, null, null);
        } else if (inRound) {
          writer.event(
              index,
              EventKind.of(event),
              EventKind.actor(event),
              changed(revision, state),
              OutcomeProjector.of(event, state));
        }
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
      control.progress(Math.min(99, (index + 1) * 100 / events));
    }

    /** 局面の改訂が変わった時だけ局面を写す。 */
    private TableState changed(int revision, GameState state) {
      if (revision == this.revision) return null;
      this.revision = revision;
      return TableProjector.table(state);
    }

    void requireNotCancelled() {
      if (control.cancelled() || Thread.currentThread().isInterrupted())
        throw new CancellationException("Analysis was cancelled.");
    }
  }
}

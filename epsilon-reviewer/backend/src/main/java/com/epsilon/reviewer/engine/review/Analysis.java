package com.epsilon.reviewer.engine.review;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.view.EventKind;
import com.epsilon.reviewer.engine.view.OutcomeProjector;
import com.epsilon.reviewer.engine.view.RecordTable;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CancellationException;

/**
 * 牌譜の手のとおりにエンジンで局を進めて全ての判断を評価し、手順を結果の文書に書き出す。推論に渡す O は、判断する席から見える
 * 卓だけで作る。表示用の卓は他家の伏せた牌まで持つが、それは推論に渡さない。推論が失敗したときに、代わりの確率を作ってごまかす
 * ことはしない。
 *
 * <p>手順の位置は、読み込んだ牌譜のインデックスのとおりに振る。判断は、それを決めるイベントの前に書く。判断を決めるイベントとは、
 * 判断の時点でエンジンがまだ適用していない牌譜のイベントのうち、最初のもの。
 */
public final class Analysis {
  /** O の作業領域として初めに用意する long の数。足りなければ広げる。 */
  private static final int INITIAL_LONGS = 256;

  private final LoadedRecord record;
  private final ReviewModel model;
  private final JobControl control;
  private final ResultWriter writer;
  private final Variant variant;
  private final ReplayDriver driver;
  private final RuleWorkspace ws;

  /** 解析ごとに作る。スレッドセーフではない。 */
  private final ObservationWriter observations;

  private final ObservationBatch batch;
  private final Observation observation;

  /** 次に書く手順の位置。 */
  private int step;

  private boolean valued;

  private Analysis(LoadedRecord record, ReviewModel model, JobControl control, ResultWriter writer) {
    this.record = record;
    this.model = model;
    this.control = control;
    this.writer = writer;
    variant = record.game().variant();
    driver = new ReplayDriver(variant);
    ws = new RuleWorkspace(variant);
    observations = new ObservationWriter(variant);
    batch = new ObservationBatch(variant, INITIAL_LONGS);
    observation = new Observation(variant);
  }

  /**
   * 牌譜を再生して全ての判断を評価し、書き終えたら結果の文書を閉じる。書き出し先のストリームは閉じない。
   *
   * @throws ReviewerException モデルが牌譜のルール（四人打ちか三人打ち）に対応していないとき（model_unavailable）
   * @throws CancellationException 取り消されたか、スレッドが割り込まれた
   */
  public static void run(
      LoadedRecord record, ReviewModel model, JobControl control, ResultWriter writer)
      throws IOException {
    if (model.variant() != record.game().variant())
      throw new ReviewerException(
          ErrorCode.MODEL_UNAVAILABLE,
          "The model reads " + model.variant() + " records, not " + record.game().variant() + ".");
    var analysis = new Analysis(record, model, control, writer);
    List<RecordedRound> rounds = record.rounds();
    for (int round = 0; round < rounds.size(); round++) analysis.round(round, rounds.get(round));
    analysis.requireNotCancelled();
    writer.event(analysis.step, EventKind.END_GAME, -1, null, null);
    writer.finish(analysis.valued ? Double.valueOf(model.temperature()) : null);
  }

  private void round(int index, RecordedRound round) throws IOException {
    requireNotCancelled();
    var table = new RecordTable(round);
    step = record.firstStep(index);
    writer.startRound(round.start().round(), round.start().honba(), table.dealer(), step, table.state());
    event(EventKind.START_KYOKU, -1);
    List<RecordEvent> events = round.events();
    int applied = 0;
    Boundary boundary = driver.start(round, ws);
    while (boundary == Boundary.DECISIONS) {
      int cursor = driver.cursor();
      for (; applied < cursor; applied++) apply(table, events.get(applied));
      decisions(record.step(index, cursor));
      boundary = driver.step(ws);
    }
    for (; applied < events.size(); applied++) apply(table, events.get(applied));
    outcome(table, round.result(), driver.engine().roundEnd());
    event(EventKind.END_KYOKU, -1);
  }

  /** この判断の時点にある全ての判断を、判断する席の順に評価して書く。 */
  private void decisions(int resolve) throws IOException {
    Engine engine = driver.engine();
    Decisions decisions = engine.decisions();
    for (int slot = 0; slot < decisions.size(); slot++) {
      requireNotCancelled();
      int seat = decisions.seat(slot);
      SeatView view = engine.view(seat);
      batch.clear();
      observation.bind(batch, observations.capture(view, decisions, slot, batch));
      int choice = driver.choice(slot);
      EvaluatedDecision evaluated =
          DecisionEvaluator.evaluate(
              model,
              observation,
              view,
              decisions,
              slot,
              choice == ReplayDriver.UNOBSERVED ? -1 : choice);
      valued = evaluated.valued();
      writer.decision(seat, step - 1, resolve, evaluated);
    }
  }

  /** 牌譜のイベントを表示用の卓に適用して書く。立直の宣言牌の打牌なら、その前に立直の宣言を書く。 */
  private void apply(RecordTable table, RecordEvent event) throws IOException {
    requireNotCancelled();
    int actor = EventKind.actor(event);
    if (EventKind.declaresRiichi(event)) event(EventKind.REACH, actor);
    table.apply(event);
    writer.event(step++, EventKind.of(event), actor, table.state(), null);
    progress();
  }

  /** 局の結果を書く。和了なら和了ごとに hora を 1 つ、流局なら ryukyoku を 1 つ書く。点の移動は表示用の卓に足す。 */
  private void outcome(RecordTable table, RoundResult result, RoundEnd end) throws IOException {
    if (result.win()) {
      for (int index = 0; index < result.wins().size(); index++) {
        table.settleWin(end, index);
        writer.event(
            step++,
            EventKind.HORA,
            result.wins().get(index).seat(),
            table.state(),
            OutcomeProjector.win(table, result, end, index));
      }
    } else {
      table.settleDraw(end);
      writer.event(step++, EventKind.RYUKYOKU, -1, table.state(), OutcomeProjector.draw());
    }
  }

  /** 局面を変えない手順を書く。 */
  private void event(EventKind kind, int actor) throws IOException {
    writer.event(step++, kind, actor, null, null);
    progress();
  }

  private void progress() {
    control.progress(Math.min(99, step * 100 / record.stepCount()));
  }

  private void requireNotCancelled() {
    if (control.cancelled() || Thread.currentThread().isInterrupted())
      throw new CancellationException("Analysis was cancelled.");
  }
}

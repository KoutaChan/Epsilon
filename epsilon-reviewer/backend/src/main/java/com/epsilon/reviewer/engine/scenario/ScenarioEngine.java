package com.epsilon.reviewer.engine.scenario;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Legal;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatTable;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.review.DecisionEvaluator;
import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.engine.scenario.LocatedDecision.Kind;
import com.epsilon.reviewer.engine.view.RecordTable;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;

/**
 * 保存した解析の 1 つの判断を、局面を変えて評価し直す（もしも）。局面は、変更した局を判断する席から見える卓でもう一度再生して作る。
 * 他家の応答の受付、見逃しの振聴、槓ドラをめくる時点は、席から見える卓のルールに従う。
 */
public final class ScenarioEngine {
  /** O の作業領域として初めに用意する long の数。足りなければ広げる。 */
  private static final int INITIAL_LONGS = 256;

  private ScenarioEngine() {}

  /** @throws ReviewerException invalid_request（席の番号が牌譜の席の数以上） */
  public static void requireSeat(LoadedRecord record, int seat) {
    int seats = record.game().variant().seats();
    if (seat >= seats)
      throw new ReviewerException(
          ErrorCode.INVALID_REQUEST, "seat must be between 0 and " + (seats - 1) + ".");
  }

  /**
   * 見つけた判断を評価する。牌譜の手がまだ合法なら、それを選んだ手とする。鳴きを指定すると、判断する席がそのチー・ポンをした後の
   * 打牌を評価し、鳴いた後の局面も返す。この打牌は牌譜に無い判断なので、選んだ手は付けない。大明槓は嶺上牌が分からないので扱わない。
   * 立直の宣言牌を鳴くと、鳴いた時に立直が成立する。その立直棒で供託か点が O に書ける範囲を外れるなら断る。
   *
   * @param call 鳴いたとするチー・ポンの行動の番号。鳴かなければ null
   * @throws ReviewerException invalid_request・invalid_edit・model_unavailable
   */
  public static ScenarioAnswer evaluate(
      LoadedRecord record,
      LocatedDecision decision,
      ScenarioEdits edits,
      Integer call,
      ReviewModel model) {
    Variant variant = record.game().variant();
    if (model.variant() != variant)
      throw new ReviewerException(
          ErrorCode.MODEL_UNAVAILABLE,
          "The model reads " + model.variant() + " records, not " + variant + ".");
    RewrittenRound round =
        RoundRewriter.apply(record.rounds().get(decision.round()), decision, edits);
    int seat = decision.seat();
    var table = new SeatTable(variant, seat);
    replay(table, round, edits, new RuleWorkspace(variant));
    Decisions decisions = table.decisions();
    if (decisions.size() > 0 && Kind.of(decisions.kind(0)) != decision.kind())
      throw new IllegalStateException(
          "Seat " + seat + " reaches a " + decisions.kind(0) + " decision, not " + decision.kind());
    if (call == null) {
      if (decisions.size() == 0) {
        if (decision.kind() != Kind.RESPONSE)
          throw new IllegalStateException("Seat " + seat + " has no turn at the decision");
        return new ScenarioAnswer(DecisionEvaluator.passOnly(model, decision.recorded()), null);
      }
      return new ScenarioAnswer(evaluate(table, model, indexOf(decisions, decision.recorded())), null);
    }
    if (decision.kind() != Kind.RESPONSE)
      throw invalidEdit("call: the decision is not a response to a discard.");
    if (decisions.size() == 0 || indexOf(decisions, call) < 0)
      throw invalidEdit("call: the call is not a legal response at the decision.");
    Action.Type type = Action.of(call).type();
    if (type != Action.Type.CHI && type != Action.Type.PON)
      throw invalidEdit("call: only a chi or a pon is followed by a discard.");
    table.call(seat, Legal.callMeld(table.view(), call));
    requireWritable(table.view());
    return new ScenarioAnswer(
        evaluate(table, model, -1), RecordTable.afterCall(table.view(), round.table()));
  }

  /**
   * 変更した局を、席から見える卓で判断の時点まで再生する。変更のせいで記録の手が合法でなくなれば invalid_edit を投げる。変更が無いのに
   * 再生できなければ、元の例外をそのまま投げる。
   */
  private static void replay(
      SeatTable table, RewrittenRound round, ScenarioEdits edits, RuleWorkspace ws) {
    try {
      table.startScenario(
          round.round(),
          round.honba(),
          round.kyotaku(),
          round.scores(),
          round.hand(),
          round.indicator());
      for (RecordEvent event : round.events()) event.applyTo(table, ws);
    } catch (IllegalArgumentException | IllegalStateException failure) {
      if (edits.empty()) throw failure;
      throw invalidEdit(
          "edits: the recorded play before the decision is not legal after the edits ("
              + failure.getMessage()
              + ").");
    }
  }

  /**
   * 鳴いた後の供託と点が、O に書ける範囲にあることを確かめる。変更を確かめるのは判断の時点の値で、鳴いた時に成立する立直の立直棒は
   * まだ入っていない。
   */
  private static void requireWritable(SeatView view) {
    Variant variant = view.variant();
    int max = ScenarioLimits.maxKyotaku(variant);
    if (view.kyotaku() > max)
      throw invalidEdit("call: the riichi deposit at the call puts kyotaku above " + max + ".");
    for (int seat = 0; seat < variant.seats(); seat++)
      if (view.score(seat) < ScenarioLimits.MIN_SCORE)
        throw invalidEdit(
            "call: the riichi deposit at the call puts the score of seat "
                + seat
                + " out of range.");
  }

  /** 席から見える卓の判断を評価する。O の作業領域はスレッドセーフではないので、もしもごとに作る。 */
  private static EvaluatedDecision evaluate(SeatTable table, ReviewModel model, int chosen) {
    Variant variant = table.view().variant();
    var batch = new ObservationBatch(variant, INITIAL_LONGS);
    int index = new ObservationWriter(variant).capture(table.view(), table.decisions(), 0, batch);
    Observation observation = new Observation(variant).bind(batch, index);
    return DecisionEvaluator.evaluate(
        model, observation, table.view(), table.decisions(), 0, chosen);
  }

  /** 指定の行動の番号が、判断する席の合法手の何番目か。無ければ -1。 */
  private static int indexOf(Decisions decisions, int code) {
    if (code < 0) return -1;
    for (int index = 0; index < decisions.legalCount(0); index++)
      if (decisions.legal(0, index) == code) return index;
    return -1;
  }

  private static ReviewerException invalidEdit(String message) {
    return new ReviewerException(ErrorCode.INVALID_EDIT, message);
  }
}

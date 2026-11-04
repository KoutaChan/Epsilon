package com.epsilon.reviewer.engine.scenario;

import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.review.DecisionEvaluator;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.engine.view.TableProjector;
import com.epsilon.reviewer.error.ReviewerException;

/**
 * 保存した解析の 1 判断を、局面を変えて評価し直す（もしも）。判断は牌譜のその局の通常の再生で探し、局面は変更した局の出来事を判断の席の目で
 * 再生し直して作る。
 */
public final class ScenarioEngine {
  private ScenarioEngine() {}

  /**
   * 牌譜の手がまだ合法なら選んだ手にする。鳴きを指せば、判断の席がそのチー・ポンをした後の打牌を評価し、鳴いた後の局面を返す（記録に無い判断
   * なので選んだ手は付けない）。
   *
   * @throws ReviewerException invalid_edit・decision_not_found
   */
  public static ScenarioAnswer evaluate(
      LoadedRecord record, ScenarioQuery query, ReviewModel model) {
    LocatedDecision decision =
        DecisionLocator.find(record, query.round(), query.eventIndex(), query.seat());
    RewrittenRound round = RoundRewriter.apply(record.replay().events(), decision, query.edits());
    if (query.call() == null) {
      SeatReplay.Decision position = SeatReplay.replay(round.events(), decision);
      return new ScenarioAnswer(
          DecisionEvaluator.evaluate(
              model,
              position.state(),
              decision.seat(),
              position.legal(),
              position.slotOf(decision.recorded())),
          null);
    }
    SeatReplay.Decision position = SeatReplay.afterCall(round.events(), decision, query.call());
    return new ScenarioAnswer(
        DecisionEvaluator.evaluate(model, position.state(), decision.seat(), position.legal(), -1),
        TableProjector.afterCall(position.state(), round.table(), decision.seat()));
  }
}

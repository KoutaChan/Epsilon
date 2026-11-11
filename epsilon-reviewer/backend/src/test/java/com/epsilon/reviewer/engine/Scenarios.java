package com.epsilon.reviewer.engine;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.engine.scenario.DecisionLocator;
import com.epsilon.reviewer.engine.scenario.LocatedDecision;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEngine;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;

/**
 * もしものテストで、判断を結果の文書での位置（局、判断を決める手順の位置、席）で指す。その局を牌譜の手のとおりに再生して、指した判断が
 * その席の局の中で何番目の判断かを数え、本番と同じく {@link DecisionLocator#findNth} で探す。候補が文書と同じかは確かめない。
 */
final class Scenarios {
  private Scenarios() {}

  /** 指した判断で、もしもを評価する。失敗の種類は本番のもしもと同じ。 */
  static ScenarioAnswer evaluate(LoadedRecord record, ScenarioQuery query, ReviewModel model) {
    ScenarioEngine.requireSeat(record, query.seat());
    LocatedDecision decision = find(record, query.round(), query.eventIndex(), query.seat());
    return ScenarioEngine.evaluate(record, decision, query.edits(), query.call(), model);
  }

  /**
   * 指した判断を探す。見つからなければ decision_not_found で断る。
   *
   * @param eventIndex 判断を決める手順の位置（文書の resolve）
   */
  static LocatedDecision find(LoadedRecord record, int round, int eventIndex, int seat) {
    int ordinal = ordinal(record, round, eventIndex, seat);
    return DecisionLocator.findNth(record, round, seat, ordinal, null);
  }

  /** 指した判断が、その席が局の始めから数えて何番目にした判断か（0 から）。指した位置にその席の判断が無ければ -1。 */
  private static int ordinal(LoadedRecord record, int round, int eventIndex, int seat) {
    if (round < 0 || round >= record.rounds().size()) return -1;
    RecordedRound recorded = record.rounds().get(round);
    int position = -1;
    for (int index = 0; index <= recorded.events().size(); index++)
      if (record.step(round, index) == eventIndex) position = index;
    if (position < 0) return -1;
    Variant variant = record.game().variant();
    var driver = new ReplayDriver(variant);
    var ws = new RuleWorkspace(variant);
    int seen = 0;
    for (Boundary boundary = driver.start(recorded, ws);
        boundary == Boundary.DECISIONS && driver.cursor() <= position;
        boundary = driver.step(ws)) {
      if (driver.engine().decisions().slotOf(seat) < 0) continue;
      if (driver.cursor() == position) return seen;
      seen++;
    }
    return -1;
  }
}

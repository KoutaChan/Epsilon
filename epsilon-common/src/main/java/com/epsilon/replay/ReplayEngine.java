package com.epsilon.replay;

import java.util.List;

/** 正規化した 1 対局を先頭から再生し、判断と出来事を観測者へ同期で渡す。 */
public final class ReplayEngine {
  private ReplayEngine() {}

  /**
   * 牌譜を最後まで再生する。出来事と牌山は読むだけで変えない。観測者が投げた例外はそのまま抜ける。
   *
   * @return 最後の局面の点（EndGame に点があればその点）。呼び出し側が持つ新しい配列
   * @throws IllegalStateException 牌譜の手が合法手に無い（その判断は観測者へ届かない）か、出来事の並びを再生できない
   */
  public static int[] replay(ReplayRecord record, ReplayObserver observer) {
    List<ReplayEvent> events = record.events();
    ReplayTable table = new ReplayTable(record.walls());
    ReplayFeed feed = new ReplayFeed(observer, table.state());
    LegalActions legal = new LegalActions(table.state());
    ResponsePhase responses = new ResponsePhase(events, table, legal, feed);
    TurnPhase turns = new TurnPhase(events, table, legal, feed, responses);
    int index = 0;
    while (index < events.size())
      index = responses.isOpen() ? responses.resolve(index) : turns.apply(index);
    if (responses.isOpen()) responses.endOfRecord(events.size());
    return table.finalScores();
  }
}

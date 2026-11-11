package com.epsilon.reviewer.engine.scenario;

import java.util.List;

/**
 * 判断の局面に加える変更。点・本場・供託は、判断の局面で見えている値で指定する。省いた項目は元のまま（リストは空、数は null）。
 * 点の数が牌譜のルールの席の数と合うかは、変更を適用するときに確かめる。
 *
 * @param scores 席ごとの持ち点。空なら変えない
 */
public record ScenarioEdits(
    List<RiverEdit> rivers,
    List<DoraEdit> dora,
    List<HandEdit> hand,
    List<Integer> scores,
    Integer honba,
    Integer kyotaku) {
  public ScenarioEdits {
    rivers = rivers == null ? List.of() : rivers;
    dora = dora == null ? List.of() : dora;
    hand = hand == null ? List.of() : hand;
    scores = scores == null ? List.of() : scores;
  }

  /** 何も変えないか。 */
  public boolean empty() {
    return rivers.isEmpty()
        && dora.isEmpty()
        && hand.isEmpty()
        && scores.isEmpty()
        && honba == null
        && kyotaku == null;
  }
}

package com.epsilon.mahjong.rules;

import static com.epsilon.mahjong.rules.RoundOutcome.NO_SEAT;

/** テスト用のヘルパー。卓と局の結果を作り、精算から次の局へ進めるまでを 1 回の呼び出しで行う。 */
final class RoundFixtures {

  private RoundFixtures() {}

  /** 局の始まりの卓を作る。値は {@code RoundStart} の不変条件で確かめる。 */
  static TableState table(Variant variant, int round, int honba, int kyotaku, int... scores) {
    TableState table = new TableState(variant);
    table.set(new RoundStart(variant, round, honba, kyotaku, scores));
    return table;
  }

  static RoundOutcome tsumo(Variant variant, int base, int winner, int pao, int riichiMask) {
    RoundOutcome outcome = new RoundOutcome(variant);
    outcome.setTsumo(base, winner, pao, riichiMask);
    return outcome;
  }

  static RoundOutcome ron(Variant variant, int base, int winner, int loser, int riichiMask) {
    RoundOutcome outcome = new RoundOutcome(variant);
    outcome.setRon(base, winner, loser, NO_SEAT, riichiMask);
    return outcome;
  }

  static RoundOutcome exhaustive(Variant variant, int tenpaiMask, int nagashiMask, int riichiMask) {
    RoundOutcome outcome = new RoundOutcome(variant);
    outcome.setExhaustive(tenpaiMask, nagashiMask, riichiMask);
    return outcome;
  }

  static RoundOutcome abortive(Variant variant, AbortiveDraw draw, int riichiMask) {
    RoundOutcome outcome = new RoundOutcome(variant);
    outcome.setAbortive(draw, riichiMask);
    return outcome;
  }

  /**
   * 局を終えたときの点の移動を返す。
   *
   * @param start 局の始まりの卓
   * @param outcome 局の結果
   */
  static int[] settle(TableState start, RoundOutcome outcome) {
    int[] delta = new int[start.variant().seats()];
    Settlement.apply(start.variant(), start, outcome, delta);
    return delta;
  }

  /**
   * 局を終えた後の卓（次の局の始まりか、終局した卓）を返す。
   *
   * @param start 局の始まりの卓
   * @param outcome 局の結果
   */
  static TableState advance(TableState start, RoundOutcome outcome) {
    TableState next = new TableState(start.variant());
    Progression.next(start, outcome, settle(start, outcome), next);
    return next;
  }

  static int[] scores(TableState table) {
    int[] scores = new int[table.variant().seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = table.score(seat);
    }
    return scores;
  }
}

package com.epsilon.prism.core;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.score.RiichiState;

/**
 * 公開情報から見た脅威。判断する席から見える情報（立直の宣言と晒した面子の数）だけで決め、ほかの席の伏せた牌は見ない。卓の席のビューから
 * 求めても、観測 O から求めても同じ値になる。
 *
 * <p>脅威とみなすのは、立直している相手か、晒した面子が 2 つ以上の相手。
 */
public final class Threats {
  private Threats() {}

  /** 立直しているか、晒した面子が 2 つ以上なら true を返す。 */
  private static boolean isThreat(RiichiState riichi, int melds) {
    return riichi != RiichiState.NONE || melds >= 2;
  }

  /** ビューの席から見た、脅威の相手の数を返す。 */
  public static int count(SeatView view) {
    int seat = view.seat(), count = 0;
    for (int other = 0; other < view.variant().seats(); other++)
      if (other != seat && isThreat(view.riichi(other), view.meldCount(other))) count++;
    return count;
  }

  /**
   * 相手が、O を観測した席にとって脅威なら true を返す。
   *
   * @param o 判断の席の観測
   * @param facts 同じ O を読み込んだ、席ごとの公開情報
   * @param seat 相手の相対席（0 は自分で、いつも false）
   */
  public static boolean isThreat(Observation o, SeatFacts facts, int seat) {
    return seat != 0 && isThreat(o.riichi(seat), facts.meldCount(seat));
  }

  /** O を観測した席から見た、脅威の相手の数を返す。 */
  public static int count(Observation o, SeatFacts facts) {
    int count = 0;
    for (int seat = 1; seat < o.variant().seats(); seat++) if (isThreat(o, facts, seat)) count++;
    return count;
  }
}

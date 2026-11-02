package com.epsilon.prism.core;

import com.epsilon.core.GameState;
import com.epsilon.core.PublicObservation;

/**
 * 公開の脅威。判断の席から見える情報（立直の宣言と副露の数）だけで決め、ほかの席の伏せた牌を見ない。
 *
 * <p>脅威の相手: 立直しているか、副露が 2 つ以上の相手。
 */
public final class Threats {
  private Threats() {}

  /** 相手 other が席 seat にとって脅威か。 */
  public static boolean isThreat(GameState state, int seat, int other) {
    return other != seat && isThreat(state.isRiichi(other), state.hand(other).meldCount());
  }

  /** 立直しているか副露が 2 つ以上なら脅威。 */
  private static boolean isThreat(boolean riichi, int melds) {
    return riichi || melds >= 2;
  }

  /** 脅威の相手の数。 */
  public static int count(GameState state, int seat) {
    int count = 0;
    for (int other = 0; other < GameState.NUM_PLAYERS; other++)
      if (isThreat(state, seat, other)) count++;
    return count;
  }

  /** 観測した席から見た脅威の相手の数（{@link #count(GameState, int)} と同じ値。状態のない牌譜解析で使う）。 */
  public static int count(PublicObservation observation) {
    int seat = observation.observer(), count = 0;
    for (int other = 0; other < GameState.NUM_PLAYERS; other++)
      if (other != seat
          && isThreat(observation.isRiichi(other), observation.hand(other).meldCount())) count++;
    return count;
  }

  /** 立直している相手の数。 */
  public static int riichiOpponents(GameState state, int seat) {
    int count = 0;
    for (int other = 0; other < GameState.NUM_PLAYERS; other++)
      if (other != seat && state.isRiichi(other)) count++;
    return count;
  }
}

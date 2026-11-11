package com.epsilon.prism.core;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.SeatView;

/** 判断の席の手が門前か副露しているか。暗槓は門前を崩さないので、暗槓だけの手は門前に数える。 */
public enum Openness {
  /** 門前（晒した面子が無いか、暗槓だけ）。 */
  CLOSED,
  /** 副露している（チー・ポン・大明槓・加槓の面子がある）。 */
  OPEN;

  private static final Openness[] VALUES = values();

  public static final int COUNT = VALUES.length;

  public static Openness of(int ordinal) {
    return VALUES[ordinal];
  }

  /** このビューで見ている席の手が、門前か副露しているかを返す。 */
  public static Openness of(SeatView view) {
    int seat = view.seat();
    for (int index = 0; index < view.meldCount(seat); index++)
      if (!Meld.of(view.meld(seat, index)).type().concealed()) return OPEN;
    return CLOSED;
  }
}

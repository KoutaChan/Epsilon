package com.epsilon.prism.luck;

import com.epsilon.core.TurnEvent;

/**
 * 運の網 Ω が評価する偶然の出来事の種類。GRP を外す道は作らないので、配牌と半荘末の出来事は持たない。
 *
 * <p>結果はどの種類も牌の種と赤（{@link LuckSchema#OUTCOMES} 通り）で、出来事の前のプール（全員から見えない牌）の上で一様に出る。
 */
public enum LuckEvent {
  /** 生きた山のツモ。行為者はツモった席。 */
  LIVE_DRAW,
  /** 嶺上のツモ。行為者は槓をした席。 */
  RINSHAN,
  /** 槓ドラの表示。行為者はその表示を生んだ槓の作者（表示の時点の手番の席）。 */
  KAN_DORA;

  /**
   * ツモの種類を返す。
   *
   * @param draw ツモ
   * @return 嶺上なら {@link #RINSHAN}、ほかは {@link #LIVE_DRAW}
   */
  public static LuckEvent of(TurnEvent.Draw draw) {
    return draw.isRinshanDraw() ? RINSHAN : LIVE_DRAW;
  }
}

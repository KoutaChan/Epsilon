package com.epsilon.calculate.scoring;

import com.epsilon.core.TurnEvent;

/** 和了を発生させた局内イベント。 */
public enum WinSource {
  /** 通常の自摸または打牌に対する栄和。 */
  NORMAL,

  /** 嶺上牌による自摸和。 */
  RINSHAN,

  /** 加槓宣言牌に対する搶槓。 */
  CHANKAN,

  /** 海底牌の自摸または河底牌への栄和。 */
  LAST_TILE;

  /** 局イベントと王牌を除く牌山牌の枯渇状態から和了原因を分類する。 */
  public static WinSource from(TurnEvent event, boolean wallExhausted) {
    if (event instanceof TurnEvent.Draw draw && draw.isRinshanDraw()) {
      return RINSHAN;
    }
    if (event instanceof TurnEvent.KanAttempt kan && kan.kanKind() == TurnEvent.KanKind.KAKAN) {
      return CHANKAN;
    }
    return wallExhausted ? LAST_TILE : NORMAL;
  }
}

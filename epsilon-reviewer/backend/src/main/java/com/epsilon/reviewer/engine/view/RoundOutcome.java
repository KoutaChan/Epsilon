package com.epsilon.reviewer.engine.view;

/** 局の終わり方。hora と ryukyoku のイベントだけが持つ。点の移動と流局の理由は持たない。 */
public sealed interface RoundOutcome {
  /**
   * 和了。
   *
   * @param target 放銃した席。ツモなら {@code actor} と同じ
   * @param winningTile 和了牌の記号（ツモ牌、打牌、槍槓された加槓の牌、または抜いた北）
   * @param details 採点の内訳
   */
  record Hora(int actor, int target, String winningTile, WinDetails details)
      implements RoundOutcome {}

  record Ryukyoku() implements RoundOutcome {}
}

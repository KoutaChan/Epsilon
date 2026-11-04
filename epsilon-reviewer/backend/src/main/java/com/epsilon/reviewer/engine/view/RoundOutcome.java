package com.epsilon.reviewer.engine.view;

import com.epsilon.replay.WinDetails;

/** 局の終わり（hora・ryukyoku の出来事だけが持つ）。点の移動と流局の理由は持たない。 */
public sealed interface RoundOutcome {
  /**
   * 和了。
   *
   * @param target ロンされた席。ツモは actor と同じ
   * @param winningTile 適用した時の物理牌の記号（ツモ牌・打牌・槍槓の加槓の牌）
   * @param details 牌譜の採点の内訳。無ければ null
   */
  record Hora(int actor, int target, String winningTile, WinDetails details)
      implements RoundOutcome {}

  record Ryukyoku() implements RoundOutcome {}
}

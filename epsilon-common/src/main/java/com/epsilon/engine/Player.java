package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import java.util.List;

/** AIプレイヤーのインターフェース。 */
public interface Player {

  /**
   * 合法アクションから1つを選択する。
   *
   * @param state 自家の手牌と公開情報を参照できる、判断中だけ有効な観測
   * @param playerIndex このプレイヤーのインデックス (0-3)
   * @param legalActions 選択可能なアクションのリスト
   * @return 選択したアクション
   */
  Action selectAction(GameState state, int playerIndex, List<Action> legalActions);

  /**
   * 局終了後、次局情報まで確定した精算結果を通知する。
   *
   * @param settlement 得点移動、最終得点、次局遷移を含むエンジン所有の借用した結果。コールバック終了後に保持しない
   */
  default void onRoundSettled(RoundSettlement settlement) {}
}

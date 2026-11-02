package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.TurnEvent;
import java.util.List;

/** エンジンが行動の選択を待つ場面の種類。 */
public enum EngineDecisionKind {
  /** 自摸後の通常手番判断。 */
  TURN,
  /** 他家打牌に対するRON・PASS・副露判断。 */
  RESPONSE,
  /** 槓宣言に対する槍槓RON判断。 */
  CHANKAN,
  /** CHI・PON成立直後の強制打牌判断。 */
  POST_CALL_DAHAI;

  /**
   * 公開の観測と合法手だけで決まる判断の種類。GameEngine のない牌譜（Reviewer・SnapshotScorer）でも使える。エンジンの判断点の
   * {@link EngineDecisionPoint#kind()} と常に等しい。
   *
   * <p>他家の打牌・槓の宣言への応答は RESPONSE・CHANKAN。CHI・PON の直後は手番の出来事が消え（{@link TurnEvent.None}）手番が鳴いた
   * 席に移るので、直前の出来事がなく自分の番で打牌だけが合法なら POST_CALL_DAHAI。ほかは TURN（通常の手番は自摸の出来事を持つ）。
   *
   * @param o 判断する席の公開の観測（呼び出しの間だけ借りる）
   * @param legal その席の合法手
   * @return 判断の種類
   */
  public static EngineDecisionKind of(PublicObservation o, List<Action> legal) {
    TurnEvent last = o.turnEvent();
    if (last instanceof TurnEvent.ResponseSource source && source.player() != o.observer()) {
      return source instanceof TurnEvent.KanAttempt ? CHANKAN : RESPONSE;
    }
    if (last == TurnEvent.None.INSTANCE
        && o.currentPlayer() == o.observer()
        && onlyDiscards(legal)) {
      return POST_CALL_DAHAI;
    }
    return TURN;
  }

  private static boolean onlyDiscards(List<Action> legal) {
    for (int action = 0, actions = legal.size(); action < actions; action++) {
      if (legal.get(action).type() != Action.Type.DAHAI) {
        return false;
      }
    }
    return true;
  }
}

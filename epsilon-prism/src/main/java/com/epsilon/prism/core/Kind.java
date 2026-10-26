package com.epsilon.prism.core;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import java.util.List;

/** 判断の種別。合法手の集合から決め、複数に当てはまるときは先に並ぶ種別を採る。 */
public enum Kind {
  /** TSUMO_AGARI か RON_AGARI が合法（槍槓を含む）。 */
  WIN,
  /** 九種九牌が合法。 */
  KYUSHU,
  /** 立直宣言の打牌が合法。 */
  RIICHI,
  /** 暗槓・加槓・大明槓のいずれかが合法。 */
  KAN,
  /** 応答で CHI か PON が合法。 */
  CALL,
  /** 鳴いた直後の打牌。 */
  POSTCALL,
  /** 上のどれにも当たらない手番の打牌。 */
  DISCARD;

  private static final Kind[] VALUES = values();

  public static Kind of(int ordinal) {
    return VALUES[ordinal];
  }

  /** 収集・評価・学習で共通の分類。合法手が1つの判断にも種別は付くが、強制手は分岐しない。 */
  public static Kind classify(EngineDecisionKind decision, List<Action> legal) {
    boolean riichi = false, kan = false, call = false, kyushu = false;
    for (int i = 0, n = legal.size(); i < n; i++) {
      switch (legal.get(i).type()) {
        case TSUMO_AGARI, RON_AGARI -> {
          return WIN;
        }
        case KYUSHU_KYUHAI -> kyushu = true;
        case RIICHI_DAHAI -> riichi = true;
        case ANKAN, KAKAN, DAIMINKAN -> kan = true;
        case CHI, PON -> call = true;
        default -> {}
      }
    }
    if (kyushu) return KYUSHU;
    if (riichi) return RIICHI;
    if (kan) return KAN;
    if (call) return CALL;
    if (decision == EngineDecisionKind.POST_CALL_DAHAI) return POSTCALL;
    return DISCARD;
  }

  /**
   * エンジンの判断の種別が来ない判断（牌譜解析）の種別。{@link #classify(EngineDecisionKind, List)} と同じ値になる。CHI・PON の直後は手番の出来事が
   * 消え（{@link TurnEvent.None}）、手番は鳴いた席になる（通常の手番は自摸の出来事を持つ）。種別の判断で要るのは鳴いた直後の打牌とほかの打牌の
   * 区別だけで、ほかの種別は合法手で決まる。
   */
  public static Kind classify(PublicObservation observation, List<Action> legal) {
    boolean postCall =
        observation.turnEvent() == TurnEvent.None.INSTANCE
            && observation.currentPlayer() == observation.observer()
            && legal.stream().allMatch(action -> action.type() == Action.Type.DAHAI);
    return classify(postCall ? EngineDecisionKind.POST_CALL_DAHAI : EngineDecisionKind.TURN, legal);
  }
}

package com.epsilon.prism.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;

/** 推論の受付でバッチを分ける区分。遷移を複数持つ合法手（チー・ポン）を含む行は入力が大きいので、行数の上限を別にする。 */
public enum Capacity {
  SINGLE_TRANSITION,
  MULTIPLE_TRANSITIONS;

  /** 観測の合法手にチーかポンがあれば {@link #MULTIPLE_TRANSITIONS}、無ければ {@link #SINGLE_TRANSITION} を返す。 */
  public static Capacity of(Observation observation) {
    for (int action = 0, actions = observation.legalActions(); action < actions; action++) {
      Action.Type type = observation.action(action).type();
      if (type == Action.Type.CHI || type == Action.Type.PON) {
        return MULTIPLE_TRANSITIONS;
      }
    }
    return SINGLE_TRANSITION;
  }
}

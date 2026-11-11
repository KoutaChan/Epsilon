package com.epsilon.agent.policy;

import com.epsilon.agent.observe.Observation;

/** 対局を進めるスレッドで、その場で答える方策。 */
public non-sealed interface InlinePolicy extends Policy {

  /** 記録の判断で選んだ手（合法手の何番目か）を返す。メモリを確保しない。 */
  int act(Observation observation);
}

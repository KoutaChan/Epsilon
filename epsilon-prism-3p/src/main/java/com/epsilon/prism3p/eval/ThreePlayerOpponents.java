package com.epsilon.prism3p.eval;

import com.epsilon.agent.policy.Series;
import com.epsilon.prism.eval.OpponentRules;
import java.util.List;

/** 三人打ちの評価の卓で使える相手。ルールベースの打ち手だけで、チェックポイントを開かないので既定のデバイスは cpu。 */
public enum ThreePlayerOpponents implements OpponentRules {
  RULES;

  private static final List<Series> SERIES = List.of(Series.RULE_BOT);

  @Override
  public List<Series> series() {
    return SERIES;
  }

  @Override
  public String defaultDevice() {
    return "cpu";
  }
}

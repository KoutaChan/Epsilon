package com.epsilon.prism4p.eval;

import com.epsilon.agent.policy.Series;
import com.epsilon.prism.eval.OpponentRules;
import java.util.List;

/** 四人打ちの評価の卓で相手にできる打ち手。PRISM のバージョンのほかに、ルールベースの打ち手と classic の Nano・Major（GPU 0 で打つ）を使える。 */
public enum FourPlayerOpponents implements OpponentRules {
  RULES;

  private static final List<Series> SERIES = List.of(Series.RULE_BOT, Series.NANO, Series.MAJOR);

  @Override
  public List<Series> series() {
    return SERIES;
  }

  @Override
  public String defaultDevice() {
    return "gpu0";
  }
}

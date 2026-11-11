package com.epsilon.agent.eval;

/** 決闘を止めた理由。 */
public enum Stopped {
  /** 良い・悪い・互角が出た。早く止まった決闘ほど、この時点の平均は真の差より絶対値が大きく出やすい。 */
  DECIDED,
  /** 標準誤差が目標以下になった。 */
  PRECISE,
  /** 計画の牌山を全て打った。 */
  LIMIT
}

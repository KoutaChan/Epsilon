package com.epsilon.agent.eval;

/** 評価の卓で使う牌山の選び方。 */
public enum WallSet {
  /** 全ての run とバージョンで同じ牌山を使う（{@link Arena#EVALUATION_SEED}）。 */
  FIXED,
  /** ジョブごとに新しい牌山を使う（{@link Arena#freshSeed}）。 */
  FRESH
}

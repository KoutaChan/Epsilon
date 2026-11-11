package com.epsilon.prism.hip;

/** FFN を通すトークンの組。並びは native の TokenGroup と同じ。局・席と牌は、トークンの軸の上でそれぞれ連続している。 */
public enum TokenGroup {
  /** 型で分けない、全てのトークン。 */
  ALL,
  /** 局と席のトークン（トークンの軸の先頭から (1 + 席の数) × 行の数）。 */
  CONTEXT,
  /** 牌のトークン（トークンの軸で局・席の後に続く、牌の数 × 行の数）。 */
  TILE
}

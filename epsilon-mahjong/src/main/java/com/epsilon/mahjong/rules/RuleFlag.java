package com.epsilon.mahjong.rules;

/** ルールごとに有無が分かれる項目。 */
public enum RuleFlag {
  CHI,
  /** 北抜き。抜いた北は河に置かず、補充の牌を王牌から引く。 */
  NUKI,
  /** 北をポン・槓できる。北抜きのあるルールだけで意味を持つ。 */
  NORTH_CALLS
}

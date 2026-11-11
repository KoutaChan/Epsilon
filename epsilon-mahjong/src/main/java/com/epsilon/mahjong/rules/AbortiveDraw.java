package com.epsilon.mahjong.rules;

/** 途中流局の種類。どれも親が連荘し、本場が 1 つ増える。 */
public enum AbortiveDraw {
  /** 九種九牌。誰も鳴き・暗槓・抜きをしていない第一ツモで、么九牌が 9 種以上あるとき。 */
  NINE_TERMINALS,
  FOUR_WINDS,
  /** 四家立直。4 人目の立直の宣言牌が通ったとき。 */
  FOUR_RIICHI,
  /** 四槓散了。2 人以上で槓が合わせて 4 つになり、その後の打牌が通ったとき。 */
  FOUR_KANS,
  /** 三家和。同じ打牌か加槓に 3 人がロンしたとき。 */
  TRIPLE_RON;

  /** 三家和のあるルールでは、ロンした人数がこの数になると途中流局になる。 */
  public static final int TRIPLE_RON_WINNERS = 3;
}

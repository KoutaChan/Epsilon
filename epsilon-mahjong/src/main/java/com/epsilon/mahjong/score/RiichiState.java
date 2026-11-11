package com.epsilon.mahjong.score;

/** 立直の状態。定数の名前と順は PRISM の語彙が使うので変えない。 */
public enum RiichiState {
  NONE,
  RIICHI,
  /** ダブル立直。第一巡で誰の鳴き・暗槓・抜きもまだ無いうちに、自分の最初の打牌で立直した。 */
  DOUBLE_RIICHI
}

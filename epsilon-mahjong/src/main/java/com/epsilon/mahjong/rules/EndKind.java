package com.epsilon.mahjong.rules;

public enum EndKind {
  TSUMO,
  /** ダブロンを含む。 */
  RON,
  /** 荒牌の流局。流し満貫を含む。 */
  EXHAUSTIVE,
  ABORTIVE;

  /** ツモかロンの和了なら true。和了なら、場の供託は最初の和了者が受け取る。 */
  public boolean win() {
    return this == TSUMO || this == RON;
  }
}

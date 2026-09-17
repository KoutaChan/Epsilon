package com.epsilon.calculate.scoring;

/** 現在の局面または候補行動を適用した後の手牌に対応する、リーチの状態。 */
public enum RiichiState {
  /** 未立直。 */
  NONE,
  /** 通常の立直。 */
  RIICHI,
  /** 一巡目かつ鳴きのない状態で成立したダブル立直。 */
  DOUBLE_RIICHI
}

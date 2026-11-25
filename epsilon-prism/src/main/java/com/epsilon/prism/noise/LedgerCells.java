package com.epsilon.prism.noise;

/** 帳簿の係数を当てはめる区分。行が足りない区分では、判断のクラスごとの係数、全体の係数、0 の順に代わりに使う。 */
public enum LedgerCells {
  /** 判断のクラスごと。 */
  CLASS,
  /** 判断のクラス、局の段階、順位の組ごと（区分の数と境は、帳簿の当てはめがルールの局と席から決める）。 */
  CLASS_STAGE_RANK
}

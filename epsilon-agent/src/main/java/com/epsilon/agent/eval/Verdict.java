package com.epsilon.agent.eval;

/** 決闘の判定。区間は止め方で決まる区間で、値は候補が良いほど大きくなる向き。 */
public enum Verdict {
  /** 良い。区間の下端 > 0。 */
  BETTER,
  /** 悪い。区間の上端 < 0。 */
  WORSE,
  /** 互角。良いでも悪いでもなく、区間が互角の幅の内側にある。 */
  EVEN,
  /** 計画の牌山を全て打っても、どれにも決まらない。 */
  UNDECIDED
}

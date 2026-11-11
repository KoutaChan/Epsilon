package com.epsilon.prism.core;

/** 中立ロールインの逸脱 d を引く手の集合。 */
public enum DeviationTarget {
  /** 候補集合 C(o) の全ての手。d が実際に打った手と同じなら、写しを作らない。 */
  ANY,
  /** 候補集合 C(o) から、実際に打った手を除いた手。いつも写しを作る。 */
  OTHER
}

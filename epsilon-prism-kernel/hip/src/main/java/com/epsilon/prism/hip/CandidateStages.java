package com.epsilon.prism.hip;

/** 候補層の段の組み方。並びは native の CandidateStages と同じ。 */
public enum CandidateStages {
  /** 遷移ごとの Cross-Attention と FFN の 1 段だけ。B は K6 が書く。 */
  ONE,
  /** 1 段目の後に 2 段目を持つ。2 段目は Attention だけで、遷移ごとに局・席・牌とイベントを読み直す。B は K9 が書く。 */
  REREAD
}

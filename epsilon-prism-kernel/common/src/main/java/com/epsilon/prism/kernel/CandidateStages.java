package com.epsilon.prism.kernel;

/** 候補層の段の組み方。並びは native の CandidateStages と同じ。 */
public enum CandidateStages {
  /** 遷移ごとの Cross-Attention と FFN の 1 段だけ。打牌の値は候補層が書く。 */
  ONE,
  /** 1 段目の後に 2 段目を持つ。2 段目は Attention だけで、遷移ごとに局・席・牌とイベントを読み直す。打牌の値はイベントの再読後に書く。 */
  REREAD
}

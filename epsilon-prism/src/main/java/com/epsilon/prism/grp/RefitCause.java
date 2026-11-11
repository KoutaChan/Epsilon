package com.epsilon.prism.grp;

/** GRP を学び直した理由（refits.jsonl の cause）。 */
public enum RefitCause {

  /** grp-refit で要求された。 */
  MANUAL,

  /** 反復ごとの予定どおりに学び直した。 */
  SCHEDULE
}

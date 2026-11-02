package com.epsilon.prism.grp;

/** GRP を学習した理由（refits.jsonl の cause）。 */
public enum RefitCause {

  /** grp-refit の要求。 */
  MANUAL,

  /** grpRefitSchedule の間隔。 */
  SCHEDULE,

  /** ずれの警報による前倒し。 */
  DRIFT
}

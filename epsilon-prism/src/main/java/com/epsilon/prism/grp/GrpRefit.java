package com.epsilon.prism.grp;

/** GRP をいつ学び直すか。 */
public enum GrpRefit {

  /** grp-refit で要求されたときだけ学び直す。 */
  MANUAL,

  /** 反復ごとに学び直す。コホート n が確定したら M(n) を学ぶ（{@link GrpLifecycle}）。 */
  SCHEDULE
}

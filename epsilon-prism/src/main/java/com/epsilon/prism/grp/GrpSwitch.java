package com.epsilon.prism.grp;

/** 関門に合格した候補をどう使い始めるか。 */
public enum GrpSwitch {

  /** 候補を refits.jsonl に READY として書くだけにする。 */
  PROPOSE,

  /** M(n) がコホート n ＋ 1 で関門に合格したら、反復 n ＋ 2 から使う（{@link GrpLifecycle}）。 */
  AUTO
}

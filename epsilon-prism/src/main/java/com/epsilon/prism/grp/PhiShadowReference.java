package com.epsilon.prism.grp;

/** 裏の採点で、今使っている Φ と並べて採点する影の Φ の選び方。 */
public enum PhiShadowReference {

  /** READY になった自己対局の候補だけを採点する。 */
  SELF_ONLY,

  /** 自己対局の候補に加えて、g000 の元の GRP（温度と較正の表を重ねたもの）も採点する。 */
  WITH_ORIGINAL
}

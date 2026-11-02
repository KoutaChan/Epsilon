package com.epsilon.prism.grp;

/** 裏の採点で使用中の Φ の横に並べる影の Φ（設定 {@code epsilon.prism.phiShadowReference}）。 */
public enum PhiShadowReference {

  /** READY の自己対局の候補だけ。 */
  SELF_ONLY,

  /** 加えて grpCheckpoint の元の GRP（温度と較正の表を重ねたもの）。 */
  WITH_ORIGINAL
}

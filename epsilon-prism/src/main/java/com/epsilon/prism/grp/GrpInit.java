package com.epsilon.prism.grp;

/** 使用中の Φ が GRP のときの学習の始め（設定 {@code epsilon.prism.grpInit}）。使用中が点数だけの Φ なら常に白紙。 */
public enum GrpInit {

  /** 使用中の GRP の重みから。 */
  WARM,

  /** 白紙の GRP から。 */
  SCRATCH
}

package com.epsilon.prism.noise;

/** 分岐の牌山の作り方。 */
public enum WallSharing {
  /** 分岐は親の牌山をそのまま写す。兄弟の分岐は同じ牌山で打つ。 */
  SHARED,
  /** 分岐ごとに、見えない牌を並べ直す。 */
  INDEPENDENT
}

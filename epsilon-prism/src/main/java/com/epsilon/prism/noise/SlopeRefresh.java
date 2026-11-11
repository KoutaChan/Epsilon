package com.epsilon.prism.noise;

/** 帳簿の傾きの列（点に対する Φ の傾き）を作り直す時点。 */
public enum SlopeRefresh {
  /** 局で 1 回だけ作る。 */
  HAND,
  /** 立直の供託で卓が変わったら、次に帳簿を足す判断の時点で作り直す。 */
  DEPOSIT
}

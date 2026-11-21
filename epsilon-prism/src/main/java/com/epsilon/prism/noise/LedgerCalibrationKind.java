package com.epsilon.prism.noise;

/** 帳簿の補正率 h(o) の形。 */
public enum LedgerCalibrationKind {
  /**
   * 帳簿の全ての列に共通の率。自席の順位ごとに、最も近い他家との点差と局の番号の格子に係数を置き、局面の値を双線形に補間する。
   */
  COMMON_SCALE
}

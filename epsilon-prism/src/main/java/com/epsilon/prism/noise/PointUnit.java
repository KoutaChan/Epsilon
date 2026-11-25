package com.epsilon.prism.noise;

/** 運の精算の帳簿で、点の組の列が出来事（ツモの和了・放銃・栄和）に付ける値の単位。 */
public enum PointUnit {
  /** 動く点（千点）。 */
  THOUSANDS,
  /** その出来事で局が終わった卓の Φ と、立直した席だけが聴牌の荒牌流局で終わった卓の Φ の差（効用）。 */
  PHI
}

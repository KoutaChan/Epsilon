package com.epsilon.prism.noise;

/**
 * 運の精算の帳簿で、点の組の列が出来事（ツモの和了・放銃・栄和）に付ける値の単位。Φ で付ける値は、その出来事で局が終わった卓の Φ と、
 * 立直した席だけが聴牌の荒牌流局で終わった卓の Φ の差（効用）。
 */
public enum PointUnit {
  /** 全ての列を、動く点（千点）で付ける。 */
  THOUSANDS(false, false),
  /** 全ての列を、収集が局末の値に使う Φ で付ける。 */
  PHI(true, true),
  /** 全ての列を Φ で付ける。Φ は、終盤の数え上げをせず、葉のネットワークだけで求める。 */
  PHI_LEAF(true, true),
  /** ツモの和了の 2 列だけを、収集が局末の値に使う Φ で付ける。放銃と栄和の列は千点で付ける。 */
  PHI_DRAWS(true, false);

  private final boolean phiDraws;
  private final boolean phiRons;

  PointUnit(boolean phiDraws, boolean phiRons) {
    this.phiDraws = phiDraws;
    this.phiRons = phiRons;
  }

  /** ツモの和了の列を Φ で付けるなら true。 */
  public boolean phiDraws() {
    return phiDraws;
  }

  /** 放銃と栄和の列を Φ で付けるなら true。 */
  public boolean phiRons() {
    return phiRons;
  }
}

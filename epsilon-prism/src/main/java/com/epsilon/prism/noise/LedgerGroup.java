package com.epsilon.prism.noise;

import java.util.EnumSet;
import java.util.Set;

/** 運の精算の帳簿の列をまとめた組。組ごとに、計算するかと、教師の補正に使うかを決める。 */
public enum LedgerGroup {
  /** 自分と相手の手を進める牌を引いたか。 */
  IMPROVE,
  /** ツモ・放銃・危険・相手のツモと機会の点（千点）。 */
  POINTS,
  /** 自分と相手のドラの数。 */
  DORA,
  /** 局末の裏ドラの数。 */
  URA,
  /** 自分と相手が方策から手を引いたときの運（効用）。 */
  POLICY,
  /** 点の動きを Φ の傾きで効用に直した値。 */
  SLOPE,
  /** 局を終える結果ごとの、次局の卓の Φ と今の卓の Φ の差。 */
  EXACT,
  /** 引いた牌のドラ・赤・聴牌の待ちの枚数。 */
  DRAW,
  /** 判断の時点の乱数で決めた、Y と関係のない列。 */
  FAKE;

  /** 行の帳簿の 17 列が属する組を返す。 */
  public static Set<LedgerGroup> computed() {
    return EnumSet.of(IMPROVE, POINTS, DORA, URA, POLICY);
  }
}

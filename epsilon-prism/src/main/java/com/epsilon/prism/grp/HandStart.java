package com.epsilon.prism.grp;

import com.epsilon.ai.grp.EpsilonGrpFeature;

/**
 * 局頭の公開の整数（GRP の 1 歩の入力。{@link EpsilonGrpFeature#fromProgress}）。局末の記録では精算後の次局の値。
 *
 * @param kyokuIndex 局番号（東 1 = 0）
 * @param honba 本場
 * @param kyotaku 供託の本数
 * @param scores 4 席の点数（書き換えない）
 */
public record HandStart(int kyokuIndex, int honba, int kyotaku, int[] scores) {

  /** 半荘の始め（東 1 局 0 本場、25,000 点ずつ）。 */
  public static final HandStart INITIAL = new HandStart(0, 0, 0, new int[] {25000, 25000, 25000, 25000});

  public HandStart {
    if (scores.length != 4) throw new IllegalArgumentException("Four scores required");
  }

  /** GRP の 1 歩の特徴。 */
  public float[] feature() {
    return EpsilonGrpFeature.fromProgress(kyokuIndex, honba, kyotaku, scores);
  }
}

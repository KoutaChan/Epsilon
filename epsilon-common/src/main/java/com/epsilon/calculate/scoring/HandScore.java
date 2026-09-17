package com.epsilon.calculate.scoring;

/** 和了の評価結果を保存する不変の値。配列や変更可能な子オブジェクトを保持しない。 */
public record HandScore(
    boolean isMenzen,
    long yakuBits,
    int han,
    int fu,
    int omoteDoraCount,
    int akaDoraCount,
    int uraDoraCount,
    int yakumanMultiplier,
    int basePoints) {

  public boolean hasYaku(ScoringYaku yaku) {
    return YakuBits.contains(yakuBits, yaku);
  }

  public boolean hasYakuman(ScoringYaku yaku) {
    return yaku.isYakuman() && hasYaku(yaku);
  }

  public boolean isYakuman() {
    return yakumanMultiplier > 0;
  }
}

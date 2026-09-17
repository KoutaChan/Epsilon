package com.epsilon.calculate.scoring;

/** 親子・支払い方法に依存しない手の基本点。 */
public final class ScoreMath {
  private ScoreMath() {}

  public static int normalBasePoints(int han, int fu) {
    if (han < 0 || fu < 0) throw new IllegalArgumentException("negative han or fu");
    if (han >= 13) return 8000;
    if (han >= 11) return 6000;
    if (han >= 8) return 4000;
    if (han >= 6) return 3000;
    if (han >= 5) return 2000;
    return Math.min(2000, fu << (han + 2));
  }
}

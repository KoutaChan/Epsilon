package com.epsilon.calculate.scoring;

/** 親子・支払い方法に依存しない手の基本点。 */
public final class ScoreMath {
  /** 符なし、または符を使わない役満を表す符コード。 */
  public static final int FU_CODE_NONE = 0;

  private ScoreMath() {}

  /** 翻・符・役満倍率から基本点を求める。役満倍率が指定された場合は翻・符を使用しない。 */
  public static int basePoints(int han, int fu, int yakumanMultiplier) {
    if (yakumanMultiplier < 0) throw new IllegalArgumentException("negative yakuman multiplier");
    if (yakumanMultiplier > 0) return 8000 * yakumanMultiplier;
    return normalBasePoints(han, fu);
  }

  public static int normalBasePoints(int han, int fu) {
    if (han < 0 || fu < 0) throw new IllegalArgumentException("negative han or fu");
    if (han >= 13) return 8000;
    if (han >= 11) return 6000;
    if (han >= 8) return 4000;
    if (han >= 6) return 3000;
    if (han >= 5) return 2000;
    return Math.min(2000, fu << (han + 2));
  }

  /** 疎な符の値を、0から11までの連続したコードへ変換する。 */
  public static int encodeFuCode(int fu) {
    if (fu == 0) return FU_CODE_NONE;
    if (fu == 20) return 1;
    if (fu == 25) return 2;
    if (fu >= 30 && fu <= 110 && fu % 10 == 0) return fu / 10;
    throw new IllegalArgumentException("unsupported fu: " + fu);
  }

  /** {@link #encodeFuCode(int)}で符号化した値を符へ戻す。 */
  public static int decodeFuCode(int code) {
    if (code == FU_CODE_NONE) return 0;
    if (code == 1) return 20;
    if (code == 2) return 25;
    if (code >= 3 && code <= 11) return code * 10;
    throw new IllegalArgumentException("unsupported fu code: " + code);
  }
}

package com.epsilon.prism.grp;

/**
 * 見張りと関門で局ごとに分ける区分（終わった局の局番号。{@code PhiCalibration} の段と audit-grp の CSV の kyoku と同じ。南 4 と西入は
 * まとめる）。
 */
public enum HandGroup {
  E1,
  E2,
  E3,
  E4,
  S1,
  S2,
  S3,
  S4_PLUS;

  private static final HandGroup[] VALUES = values();

  /**
   * 局番号の区分。
   *
   * @param kyokuIndex 東 1 = 0、…、南 4 = 7、西入はそれ以上
   */
  public static HandGroup of(int kyokuIndex) {
    if (kyokuIndex < 0) throw new IllegalArgumentException("Negative kyoku index " + kyokuIndex);
    return VALUES[Math.min(kyokuIndex, VALUES.length - 1)];
  }
}

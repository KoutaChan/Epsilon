package com.epsilon.prism.core;

/** 局の段階。残りの山の枚数だけで決める公開情報。 */
public enum Stage {
  /** 残り50枚以上。 */
  EARLY,
  /** 残り25〜49枚。 */
  MID,
  /** 残り24枚以下。 */
  LATE;

  private static final Stage[] VALUES = values();

  public static Stage of(int wallRemaining) {
    return wallRemaining >= 50 ? EARLY : wallRemaining >= 25 ? MID : LATE;
  }

  public static Stage of(byte ordinal) {
    return VALUES[ordinal];
  }
}

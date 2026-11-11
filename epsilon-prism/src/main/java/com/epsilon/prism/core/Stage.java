package com.epsilon.prism.core;

import com.epsilon.mahjong.rules.Variant;

/**
 * 局の段階。山の残り枚数だけで決まるので、公開情報から求められる。しきい値は山の長さに合わせて、四人打ちと三人打ちで変える（四人打ちは
 * 50 枚・25 枚、三人打ちは 40 枚・20 枚）。
 */
public enum Stage {
  /** 山の残りが序盤のしきい値（四人打ち 50 枚、三人打ち 40 枚）以上。 */
  EARLY,
  /** 山の残りが中盤のしきい値（四人打ち 25 枚、三人打ち 20 枚）以上で、序盤のしきい値より少ない。 */
  MID,
  /** 山の残りが中盤のしきい値より少ない。 */
  LATE;

  private static final Stage[] VALUES = values();

  /**
   * 段階を返す。
   *
   * @param variant 四人打ちか三人打ちか
   * @param wallRemaining 王牌を除いた山の残り枚数
   * @return 段階
   */
  public static Stage of(Variant variant, int wallRemaining) {
    return wallRemaining >= earlyWall(variant) ? EARLY : wallRemaining >= midWall(variant) ? MID : LATE;
  }

  /**
   * 序盤にする、残りの山の枚数の下限。
   *
   * @param variant 四人打ちか三人打ちか
   * @return 四人打ちは 50 枚、三人打ちは 40 枚
   */
  public static int earlyWall(Variant variant) {
    return switch (variant) {
      case TENHOU_4P -> 50;
      case TENHOU_3P -> 40;
    };
  }

  /**
   * 中盤にする、残りの山の枚数の下限。
   *
   * @param variant 四人打ちか三人打ちか
   * @return 四人打ちは 25 枚、三人打ちは 20 枚
   */
  public static int midWall(Variant variant) {
    return switch (variant) {
      case TENHOU_4P -> 25;
      case TENHOU_3P -> 20;
    };
  }

  public static Stage of(byte ordinal) {
    return VALUES[ordinal];
  }
}

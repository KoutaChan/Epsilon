package com.epsilon.grp.train;

import com.epsilon.mahjong.util.SeedMixer;

/**
 * 半荘 ID のハッシュで、半荘を学習用・検証用・関門用に分ける。分け方は学習する範囲に依らないので、検証用になった半荘は、どの
 * モデルの学習にも使われない。
 */
public enum GrpSlice {
  /** 学習に使う。 */
  TRAIN,
  /** 学習には使わない。範囲の新しい方の 2 コホートにあるものを、早期終了の判定に使う（全体の 1/16）。 */
  STOP,
  /** GUARDED のとき、関門のコホートの半荘のうち関門で測る分（そのコホートの 1/4）。学習にも早期終了にも使わない。 */
  GUARD;

  private static final double STOP_SHARE = 1.0 / 16;

  /**
   * GUARDED で関門のコホートも学習に使うとき、関門で測るために残す割合。小さいほど学習に使える半荘は増えるが、関門の SE は大きく
   * なる（1/4 なら、全ての半荘で測るときの約 2 倍）。
   */
  private static final double GUARD_SHARE = 1.0 / 4;

  /** 検証用と関門用は別のソルトのハッシュで決めるので、互いに独立に決まる。 */
  private static final long STOP_SALT = 0x5D0F_6A1B_3C27_E849L;
  private static final long GUARD_SALT = 0x2B7E_1516_28AE_D2A6L;

  /** 関門のコホートでない半荘を、学習用か検証用に分ける。 */
  public static GrpSlice of(long gameId) {
    return unit(gameId, STOP_SALT) < STOP_SHARE ? STOP : TRAIN;
  }

  /** GUARDED のとき、関門のコホートの半荘を分ける。関門用にならなかった半荘は {@link #of} と同じに分ける。 */
  public static GrpSlice guarded(long gameId) {
    return unit(gameId, GUARD_SALT) < GUARD_SHARE ? GUARD : of(gameId);
  }

  /** [0, 1) に一様に散らばるハッシュ値。 */
  private static double unit(long gameId, long salt) {
    return (SeedMixer.mix64(gameId ^ salt) >>> 11) * 0x1.0p-53;
  }
}

package com.epsilon.engine;

/**
 * 試験の 64 ビットの流れのハッシュ。値を順に混ぜ、順序も値の一部にする。
 *
 * <p>式は h ← fmix64(h + v·K)（K は黄金比の奇数、fmix64 は MurmurHash3 の最後の混ぜ）で、h の初めは 0 でない定数 {@link #START}
 * （fmix64(0) = 0 なので、0 から始めると先頭に続く 0 の値が「何も足さない」と同じになる）。製品のコードの混ぜ（SeedMixer など）には頼らない
 * （製品の側を変えても {@link GoldenGames} の対局の種が変わらないように）。浮動小数は生のビットで混ぜる（NaN の区別も含めて 1 ビットも
 * 同じ値だけが同じハッシュになる）。
 */
public final class GoldenHash {
  private static final long K = 0x9E37_79B9_7F4A_7C15L;

  /** h の初め（SHA-512 の最初の初期値。0 でなければ何でもよい）。 */
  private static final long START = 0x6A09_E667_F3BC_C908L;

  private long state = START;

  public long value() {
    return state;
  }

  public GoldenHash add(long value) {
    state = mix(state + value * K);
    return this;
  }

  public GoldenHash add(int value) {
    return add((long) value);
  }

  public GoldenHash add(boolean value) {
    return add(value ? 1L : 0L);
  }

  public GoldenHash add(float value) {
    return add(Float.floatToRawIntBits(value));
  }

  public GoldenHash add(double value) {
    return add(Double.doubleToRawLongBits(value));
  }

  /** 長さと要素を混ぜる。 */
  public GoldenHash add(short[] values, int length) {
    add(length);
    for (int index = 0; index < length; index++) add(values[index]);
    return this;
  }

  /** 長さと要素を混ぜる。 */
  public GoldenHash add(float[] values, int length) {
    add(length);
    for (int index = 0; index < length; index++) add(values[index]);
    return this;
  }

  /** 長さと要素を混ぜる。 */
  public GoldenHash add(long[] values) {
    add(values.length);
    for (long value : values) add(value);
    return this;
  }

  /** 長さと要素を混ぜる（8 バイトずつ詰めず 1 バイトずつ。長さが同じでも並びの違いを拾う）。 */
  public GoldenHash add(byte[] values, int length) {
    add(length);
    for (int index = 0; index < length; index++) add(values[index]);
    return this;
  }

  /** MurmurHash3 の fmix64。 */
  public static long mix(long value) {
    value ^= value >>> 33;
    value *= 0xFF51_AFD7_ED55_8CCDL;
    value ^= value >>> 33;
    value *= 0xC4CE_B9FE_1A85_EC53L;
    return value ^ value >>> 33;
  }
}

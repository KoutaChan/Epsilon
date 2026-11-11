package com.epsilon.agent.observe.codec;

/**
 * float32 と bfloat16 のビットを変換する。転送する数値のフィールドはホストでこの形にし、ネットワークと HIP は float32 の上位 16 bit
 * として読む。
 *
 * <p>丸めは最近接偶数（下位 16 bit がちょうど半分なら、上位の最下位ビットが偶数になる方へ丸める）。フィールドの値は有限でなければ
 * ならないので、有限でない値と、丸めると有限でなくなる値は IllegalArgumentException を投げる。
 */
public final class Bfloat16 {

  private static final int HALF_SHIFT = 16;

  /** 下位 16 bit の半分から 1 を引いた値。上位の最下位ビットも足すと、ちょうど半分のときに偶数の方へ丸まる。 */
  private static final int ROUNDING_BIAS = 0x7FFF;

  private static final int EXPONENT_MASK = 0x7F80;
  private static final int BITS_MASK = 0xFFFF;

  private Bfloat16() {}

  public static short fromFloat(float value) {
    int bits = Float.floatToRawIntBits(value);
    int rounded = (bits + ROUNDING_BIAS + ((bits >>> HALF_SHIFT) & 1)) >>> HALF_SHIFT;
    if ((rounded & EXPONENT_MASK) == EXPONENT_MASK) {
      throw new IllegalArgumentException("Value is not finite in bfloat16: " + value);
    }
    return (short) rounded;
  }

  /** テストと診断に使う。 */
  public static float toFloat(short bits) {
    return Float.intBitsToFloat((bits & BITS_MASK) << HALF_SHIFT);
  }
}

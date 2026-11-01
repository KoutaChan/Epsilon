package com.epsilon.prism.codec;

/**
 * float32 と bfloat16 のビットの変換。転送の数値の欄はホストでこの形にし、網と HIP は float32 の上位 16 bit として読む。
 *
 * <p>丸めは最近接偶数（下位 16 bit がちょうど半分なら、上位の最下位ビットが偶数になる方へ）にする。欄の値は有限でなければならないので、有限でない
 * 値と、丸めて有限でなくなる値は例外にする。
 */
public final class Bfloat16 {

  private static final int EXPONENT_MASK = 0x7F80;

  private Bfloat16() {}

  /**
   * float32 を最近接偶数で bfloat16 のビットにする。
   *
   * @param value 有限な値
   * @return bfloat16 のビット
   * @throws IllegalArgumentException 値が有限でないか、丸めると有限でなくなるとき
   */
  public static short fromFloat(float value) {
    int bits = Float.floatToRawIntBits(value);
    int rounded = (bits + 0x7FFF + ((bits >>> 16) & 1)) >>> 16;
    if ((rounded & EXPONENT_MASK) == EXPONENT_MASK) {
      throw new IllegalArgumentException("value is not finite in bfloat16: " + value);
    }
    return (short) rounded;
  }

  /**
   * bfloat16 のビットを float32 に戻す（試験と診断のため）。
   *
   * @param bits bfloat16 のビット
   * @return 値
   */
  public static float toFloat(short bits) {
    return Float.intBitsToFloat((bits & 0xFFFF) << 16);
  }
}

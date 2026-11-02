package com.epsilon.prism.codec;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * 符号なしのバイト（u8）の列を、添字ごとの上限と比べる。網の入力の語彙と鍵の符号の確かめ（行ごとに 1 回）に使う。割り当てをしない。
 *
 * <p>8 バイトを 1 語にまとめて比べる。各バイトの最上位のビットを除いて上限から値を引き（バイトをまたぐ桁借りが起きない）、最上位のビットの
 * 桁借りの式で「値 &gt; 上限」のバイトを出す。
 */
public final class UnsignedBytes {

  private static final VarHandle WORDS =
      MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
  private static final long HIGH_BITS = 0x8080808080808080L;

  private UnsignedBytes() {}

  /**
   * 先頭の {@code length} 個の値が、同じ添字の上限以下かを返す。
   *
   * @param values 値
   * @param limits 添字ごとの上限
   * @param length 比べる数
   * @return 全ての {@code values[i] ≤ limits[i]}（符号なし）なら true
   * @throws IndexOutOfBoundsException どちらかの配列が {@code length} より短いとき
   */
  public static boolean allAtMost(byte[] values, byte[] limits, int length) {
    Objects.checkFromIndexSize(0, length, values.length);
    Objects.checkFromIndexSize(0, length, limits.length);
    long above = 0L;
    int index = 0;
    for (; index + Long.BYTES <= length; index += Long.BYTES) {
      long value = (long) WORDS.get(values, index);
      long limit = (long) WORDS.get(limits, index);
      long difference = (limit | HIGH_BITS) - (value & ~HIGH_BITS);
      above |= (~limit & value | ~(limit ^ value) & ~difference) & HIGH_BITS;
    }
    for (; index < length; index++) {
      if ((values[index] & 0xFF) > (limits[index] & 0xFF)) {
        return false;
      }
    }
    return above == 0L;
  }
}

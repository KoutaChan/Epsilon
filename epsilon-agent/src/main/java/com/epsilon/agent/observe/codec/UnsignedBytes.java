package com.epsilon.agent.observe.codec;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * 符号なしのバイト（u8）の列を、添字ごとの上限と比べる。ネットワークの入力に使うキーのコードが、語彙の中でそのキーに割り当てた
 * 範囲に収まるかを、行ごとに 1 回確かめるのに使う。メモリは確保しない。
 *
 * <p>8 バイトを 1 つの {@code long} にまとめて比べる。各バイトの最上位のビットを落としてから、上限から値を引く。こうすると
 * バイトをまたぐ桁借りは起きない。最上位のビットの桁借りの式で、値が上限より大きいバイトを見つける。
 */
public final class UnsignedBytes {

  private static final VarHandle WORDS = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
  private static final long HIGH_BITS = 0x8080808080808080L;
  private static final int BYTE_MASK = 0xFF;

  private UnsignedBytes() {}

  /**
   * 先頭から指定した数の値が全て、同じ添字の上限以下なら true（符号なしで比べる）。どちらかの配列が指定した数より短ければ
   * IndexOutOfBoundsException を投げる。
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
      if ((values[index] & BYTE_MASK) > (limits[index] & BYTE_MASK)) {
        return false;
      }
    }
    return above == 0L;
  }
}

package com.epsilon.agent.observe.codec;

import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 転送に使う数の形式を確かめる。bfloat16 へは最近接偶数に丸め、NaN と、丸めると無限大になる値は拒む。符号なし 8 bit の列を
 * 8 バイトずつまとめて上限と比べても、1 バイトずつ比べたのと同じ結果になる。
 */
public class NumberCodingTest {

  @Test
  public void bfloat16RoundsToNearestEven() {
    Assert.assertEquals(Bfloat16.fromFloat(1f), (short) 0x3F80);
    Assert.assertEquals(Bfloat16.fromFloat(-0f), (short) 0x8000);
    // 下位 16 bit がちょうど半分のときは、上位が偶数なら切り捨て、奇数なら切り上げる。
    Assert.assertEquals(Bfloat16.fromFloat(Float.intBitsToFloat(0x3F808000)), (short) 0x3F80);
    Assert.assertEquals(Bfloat16.fromFloat(Float.intBitsToFloat(0x3F818000)), (short) 0x3F82);
    Assert.assertEquals(Bfloat16.fromFloat(Float.intBitsToFloat(0x3F808001)), (short) 0x3F81);
    Assert.assertEquals(Bfloat16.fromFloat(Float.intBitsToFloat(0x3F807FFF)), (short) 0x3F80);
    Assert.assertEquals(Bfloat16.toFloat((short) 0x3F82), Float.intBitsToFloat(0x3F820000));
    Assert.assertThrows(IllegalArgumentException.class, () -> Bfloat16.fromFloat(Float.NaN));
    Assert.assertThrows(IllegalArgumentException.class, () -> Bfloat16.fromFloat(Float.MAX_VALUE));
  }

  @Test
  public void unsignedBytesCompareEveryPairInEveryLane() {
    int length = Long.BYTES + 1;
    for (int background : new int[] {0x00, 0x7F, 0x80, 0xFF}) {
      for (int lane = 0; lane < length; lane++) {
        byte[] values = new byte[length];
        byte[] limits = new byte[length];
        for (int value = 0; value < 256; value++) {
          for (int limit = 0; limit < 256; limit++) {
            // ほかの位置は値と上限を同じにして（上限以下なので通る）、桁借りが隣のバイトへ漏れないことも確かめる。
            Arrays.fill(values, (byte) background);
            Arrays.fill(limits, (byte) background);
            values[lane] = (byte) value;
            limits[lane] = (byte) limit;
            if (UnsignedBytes.allAtMost(values, limits, length) != (value <= limit)) {
              Assert.fail("lane " + lane + " value " + value + " limit " + limit + " background " + background);
            }
          }
        }
      }
    }
  }
}

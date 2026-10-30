package com.epsilon.prism.codec;

import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 転送の数の形: bfloat16 の最近接偶数の丸めと、u8 の列を 8 バイトずつ比べる確かめが、1 つずつの比べと同じになること。 */
public class NumberCodingTest {

  @Test
  public void bfloat16RoundsToNearestEven() {
    Assert.assertEquals(Bfloat16.fromFloat(1f), (short) 0x3F80);
    Assert.assertEquals(Bfloat16.fromFloat(-0f), (short) 0x8000);
    // 下位 16 bit がちょうど半分: 上位が偶数なら切り捨て、奇数なら切り上げ。
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
            // 他の添字は値 = 上限（通る）にして、桁借りが隣のバイトへ漏れないことも見る。
            Arrays.fill(values, (byte) background);
            Arrays.fill(limits, (byte) background);
            values[lane] = (byte) value;
            limits[lane] = (byte) limit;
            if (UnsignedBytes.allAtMost(values, limits, length) != (value <= limit)) {
              Assert.fail(
                  "lane " + lane + " value " + value + " limit " + limit + " background "
                      + background);
            }
          }
        }
      }
    }
  }
}

package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;

import com.epsilon.prism.hip.RmsNormKernels.NormShape;
import org.testng.annotations.Test;

/** {@link NormShape} の約束（GPU の演算の形の決まり）。 */
public class RmsNormKernelsTest {

  /** 本番の幅と上限の幅は通り、部分和の数は塊の数の切り上げ。8 の倍数でない幅・上限を越える幅・正でない ε は英語の例外で止める。 */
  @Test
  public void shapeCountsGainBlocksAndRejectsOtherWidths() {
    int block = RmsNormKernels.GAIN_ROWS;
    assertEquals(new NormShape(1, 256, 1e-5f).partials(), 1);
    assertEquals(new NormShape(block, 256, 1e-5f).partials(), 1);
    assertEquals(new NormShape(block + 1, 256, 1e-5f).partials(), 2);
    new NormShape(7, RmsNormKernels.MAX_WIDTH, 1e-5f);
    for (int width : new int[] {0, 12, 260, RmsNormKernels.MAX_WIDTH + 8}) {
      assertThrows(IllegalArgumentException.class, () -> new NormShape(4, width, 1e-5f));
    }
    assertThrows(IllegalArgumentException.class, () -> new NormShape(0, 256, 1e-5f));
    assertThrows(IllegalArgumentException.class, () -> new NormShape(4, 256, 0f));
  }
}

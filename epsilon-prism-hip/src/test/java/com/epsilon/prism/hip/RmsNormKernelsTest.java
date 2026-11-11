package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.prism.hip.RmsNormKernels.NormShape;
import org.testng.annotations.Test;

/** {@link NormShape} のテスト（GPU の演算が受け付ける形）。 */
public class RmsNormKernelsTest {

  /**
   * 本番の幅と上限の幅は通る。部分和の数は、行の数を {@code GAIN_ROWS} で割って切り上げた数になる。幅が 0 か、8 の倍数でないか、
   * 上限を越えるときと、行の数が 0 のとき、ε が 0 のときは例外で止める。
   */
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

  /** 範囲の外の値で止めるときの例外のメッセージは、渡した行の数・幅・ε を示す。 */
  @Test
  public void rejectedShapeReportsTheGivenValues() {
    String message = expectThrows(IllegalArgumentException.class, () -> new NormShape(37, 12, 0.5f)).getMessage();
    for (String given : new String[] {"rows=37", "width=12", "epsilon=0.5"}) {
      assertTrue(message.contains(given), message);
    }
  }
}

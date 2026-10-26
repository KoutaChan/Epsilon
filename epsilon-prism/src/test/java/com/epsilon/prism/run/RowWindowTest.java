package com.epsilon.prism.run;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 学習の行の範囲と間引き: 残す割合は fraction と limit / 行の数 の小さい方で、範囲の逆転と範囲外の値は拒む。 */
public class RowWindowTest {
  @Test
  public void keepIsTheSmallerOfFractionAndLimit() {
    Assert.assertEquals(Pipeline.RowWindow.ALL.keep(1000), 1.0);
    Assert.assertEquals(new Pipeline.RowWindow(0, 0, 0.5, 0).keep(1000), 0.5);
    Assert.assertEquals(new Pipeline.RowWindow(0, 0, 1, 250).keep(1000), 0.25);
    Assert.assertEquals(new Pipeline.RowWindow(0, 0, 0.1, 250).keep(1000), 0.1);
    Assert.assertEquals(new Pipeline.RowWindow(0, 0, 1, 5000).keep(1000), 1.0);
    Assert.assertThrows(IllegalArgumentException.class, () -> new Pipeline.RowWindow(5, 4, 1, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> new Pipeline.RowWindow(0, 0, 0, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> new Pipeline.RowWindow(0, 0, 1, -1));
  }
}

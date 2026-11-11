package com.epsilon.grp.train;

import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * コホートの重み 2^{−(n − c)/τ} を確かめる（新しいほど重い）。重みが 1/16 未満のコホートと、始めの 2 コホートは使わない。
 * 使うコホートの数には上限がある。
 */
public class RecencyWindowTest {

  @Test
  public void weightsHalveEveryHalfLife() {
    RecencyWindow window = new RecencyWindow(40, 1, 64);
    Assert.assertEquals(window.weight(40), 1.0);
    Assert.assertEquals(window.weight(39), 0.5);
    Assert.assertEquals(window.weight(38), 0.25);
    Assert.assertEquals(window.weight(36), 1.0 / 16);
    Assert.assertEquals(window.first(), 36);
    Assert.assertEquals(window.cohorts(), 5);
    Assert.assertEquals(window.weight(35), 0.0);
    Assert.assertEquals(window.weight(41), 0.0);

    RecencyWindow doubled = new RecencyWindow(40, 2, 64);
    Assert.assertEquals(doubled.weight(38), 0.5, 1e-15);
    Assert.assertEquals(doubled.weight(39), Math.sqrt(0.5), 1e-15);
    Assert.assertEquals(doubled.first(), 32);
    Assert.assertEquals(doubled.weight(32), 1.0 / 16, 1e-15);
    Assert.assertEquals(doubled.weight(31), 0.0);
  }

  @Test
  public void uniformWindowsStopAtTheirLength() {
    RecencyWindow window = RecencyWindow.uniform(47, 8);
    Assert.assertEquals(window.first(), 40);
    Assert.assertEquals(window.cohorts(), 8);
    for (int cohort = 40; cohort <= 47; cohort++) {
      Assert.assertEquals(window.weight(cohort), 1.0);
    }
    Assert.assertEquals(window.weight(39), 0.0);
    Assert.assertEquals(new RecencyWindow(40, 1, 3).first(), 38);
  }

  @Test
  public void theFirstTwoCohortsAreNeverLearned() {
    Assert.assertEquals(new RecencyWindow(5, 1, 64).first(), 3);
    Assert.assertEquals(RecencyWindow.uniform(4, 8).cohorts(), 2);
    RecencyWindow early = new RecencyWindow(2, 1, 8);
    Assert.assertEquals(early.cohorts(), 0);
    Assert.assertFalse(early.contains(2));
    Assert.assertEquals(early.weight(2), 0.0);
  }

  @Test
  public void windowsNeedAPositiveHalfLifeAndLength() {
    Assert.assertThrows(IllegalArgumentException.class, () -> new RecencyWindow(10, 0, 8));
    Assert.assertThrows(IllegalArgumentException.class, () -> new RecencyWindow(10, Double.NaN, 8));
    Assert.assertThrows(IllegalArgumentException.class, () -> new RecencyWindow(10, 1, 0));
  }
}

package com.epsilon.prism.luck;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 同じ帳簿を持つ候補に異なる係数を掛けたときの教師と、世代をまたぐ選択を確かめる。 */
public class RootCoefficientsTest {
  @Test
  public void poolsIndependentSingleGameBandsAroundTheirCombinedMean() {
    var first = new LedgerCalibrator.Group("band");
    var second = new LedgerCalibrator.Group("band");
    var together = new LedgerCalibrator.Group("band");
    first.add(1, 3, 3, 1);
    together.add(1, 3, 3, 1);
    second.add(2, 3, 3, 2);
    together.add(2, 3, 3, 2);
    second.add(2, 2, 2, 2);
    together.add(2, 2, 2, 2);
    var a = first.band(true);
    var b = second.band(true);
    Assert.assertNull(a.seShadowMinusRaw());
    var pooled = a.paired().plus(b.paired());
    Assert.assertEquals(pooled.se(), together.band(true).seShadowMinusRaw(), 1e-14);
    Assert.assertTrue(pooled.se() > 0);
  }
}

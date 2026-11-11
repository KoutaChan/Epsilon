package com.epsilon.prism.noise;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 裏ドラの期待値の設定をテストする。どの局末で裏ドラの期待値を求めるかを確かめる。 */
public class UraSettingsTest {

  /** 期待値で学ぶなら、乱数の値に依らず全ての局末で期待値を求める。 */
  @Test
  public void theExpectedModeAlwaysAverages() {
    UraSettings settings = new UraSettings.Expected();
    Assert.assertTrue(settings.averages(0));
    Assert.assertTrue(settings.averages(0.999));
  }

  /** 実際の値で学ぶなら、乱数の値が監査の割合より小さい局末だけで期待値を求める。 */
  @Test
  public void theRealizedModeAveragesOnlyTheAuditedShare() {
    UraSettings settings = new UraSettings.Realized(0.25);
    Assert.assertTrue(settings.averages(0));
    Assert.assertTrue(settings.averages(0.2499));
    Assert.assertFalse(settings.averages(0.25));
    Assert.assertFalse(settings.averages(0.9));
    Assert.assertFalse(new UraSettings.Realized(0).averages(0));
    Assert.assertTrue(new UraSettings.Realized(1).averages(0.999));
  }
}

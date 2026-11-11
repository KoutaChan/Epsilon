package com.epsilon.prism.settings;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 方策の記憶の η をテストする。バージョンごとに η の割合で混ぜていくと、設定どおりの重みを付けた平均になることを、スカラーの値で確かめる。 */
public class MemorySettingsTest {

  /** バージョン v の値を v として、バージョン 1..last を記憶の η で混ぜていった平均を返す。 */
  private static double accumulate(MemorySettings memory, int last) {
    double average = 1;
    for (int v = 2; v <= last; v++) {
      double eta = memory.step(v).eta();
      average = (1 - eta) * average + eta * v;
    }
    return average;
  }

  private static MemorySettings ema(double halfLife, int from) {
    return new MemorySettings.Ema(halfLife, from, MemoryNetwork.ITERATE, MemoryNetwork.ITERATE);
  }

  @Test
  public void noneKeepsTheLatestVersion() {
    var none = new MemorySettings.None();
    Assert.assertFalse(none.averages());
    Assert.assertEquals(none.step(5).eta(), 1.0);
    Assert.assertEquals(accumulate(none, 10), 10.0, 1e-12);
  }

  @Test
  public void emaHalvesTheWeightEveryHalfLife() {
    double eta = ema(2, 1).step(3).eta();
    Assert.assertEquals(eta, 1 - Math.pow(2, -0.5), 1e-12);
    // 2 バージョン後の重みは (1 − η)² = 1/2
    Assert.assertEquals((1 - eta) * (1 - eta), 0.5, 1e-12);
  }
}

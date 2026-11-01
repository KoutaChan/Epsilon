package com.epsilon.prism.run;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 方策の記憶の η: 混ぜる割合を重ねると、決め方どおりの重み付きの平均になる（スカラーの版の値で確かめる）。 */
public class AveragingTest {
  /** 版 1..last の値 value(v) = v を、Averaging の η で重ねた平均。 */
  private static double accumulate(Averaging averaging, int last) {
    double average = 1, weight = averaging.step(1, 0).totalWeight();
    for (int v = 2; v <= last; v++) {
      Averaging.Step step = averaging.step(v, weight);
      average = (1 - step.eta()) * average + step.eta() * v;
      weight = step.totalWeight();
    }
    return average;
  }

  @Test
  public void noneKeepsTheLatestVersion() {
    var none = new Averaging(Averaging.NONE, 8, 1, 1);
    Assert.assertFalse(none.enabled());
    Assert.assertEquals(none.step(5, 0).eta(), 1.0);
    Assert.assertEquals(accumulate(none, 10), 10.0, 1e-12);
  }

  @Test
  public void emaHalvesTheWeightEveryHalfLife() {
    var ema = new Averaging(Averaging.EMA, 2, 1, 1);
    double eta = ema.step(3, 0).eta();
    Assert.assertEquals(eta, 1 - Math.pow(2, -0.5), 1e-12);
    // 2 版後の重みは (1 − η)² = 1/2
    Assert.assertEquals((1 - eta) * (1 - eta), 0.5, 1e-12);
  }

  @Test
  public void powerZeroIsTheUniformMeanAndPowerOneIsLinear() {
    // p = 0: 版 1..10 の一様な平均 5.5
    Assert.assertEquals(accumulate(new Averaging(Averaging.POWER, 8, 0, 1), 10), 5.5, 1e-12);
    // p = 1: 重み k の平均 Σk·k / Σk = (10·11·21/6) / 55 = 7
    Assert.assertEquals(accumulate(new Averaging(Averaging.POWER, 8, 1, 1), 10), 7.0, 1e-12);
  }

  @Test
  public void averagingStartsAtFrom() {
    var power = new Averaging(Averaging.POWER, 8, 0, 4);
    // 版 4 までは平均しない（η = 1）、版 4 で累計 1 から始まる
    Assert.assertEquals(power.step(3, 0).eta(), 1.0);
    Assert.assertEquals(power.step(4, 0), new Averaging.Step(1, 1));
    Assert.assertEquals(power.step(5, 1).eta(), 0.5, 1e-12);
  }

  @Test
  public void rejectsInvalidSettings() {
    Assert.assertThrows(IllegalArgumentException.class, () -> new Averaging("mean", 8, 1, 1));
    Assert.assertThrows(IllegalArgumentException.class, () -> new Averaging(Averaging.EMA, 0, 1, 1));
    Assert.assertThrows(IllegalArgumentException.class, () -> new Averaging(Averaging.POWER, 8, -1, 1));
    Assert.assertThrows(IllegalArgumentException.class, () -> new Averaging(Averaging.EMA, 8, 1, 0));
  }
}

package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import java.util.Arrays;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 群逐次検定の境界のテスト。各時点に配る有意水準は {@link Sequential} の消費関数の増分に一致し、その和は全体の有意水準になる。臨界値は、格子を細かくして収束させた値
 * とも、設計の表の値とも合う。判定の時点が 1 回だけなら、臨界値は固定の水準と同じになる。
 */
public class SequentialTest {

  /** 有意水準 0.01・消費関数の指数 3・判定 8 回の臨界値。Simpson の公式（6,001 点）で 1e-11 まで収束させ、小数 4 桁に丸めた。 */
  private static final double[] CONVERGED_RHO3_8 = {4.2702, 3.8075, 3.5263, 3.3115, 3.1325, 2.9765, 2.8366, 2.7087};

  /** 両側 0.01 の固定の臨界値 z(0.995)。 */
  private static final double FIXED_99 = 2.5758293035489004;

  @DataProvider
  public static Object[][] plans() {
    return new Object[][] {
      {0.01, 1.0, 8}, {0.01, 2.0, 8}, {0.01, 3.0, 4}, {0.01, 3.0, 8}, {0.01, 3.0, 16}, {0.05, 3.0, 8},
      {0.2, 0.5, 32}, {0.01, 4.0, 2}, {0.01, 3.0, 1}
    };
  }

  @Test(dataProvider = "plans")
  public void spentAlphaFollowsThePowerSpendingAndAddsUpToAlpha(double alpha, double spending, int looks) {
    double[] critical = new Sequential(alpha, spending, 1).boundaries(looks);
    double[] crossing = Sequential.crossings(critical);
    double spent = 0;
    for (int k = 0; k < looks; k++) {
      double planned = alpha * (Math.pow((k + 1.0) / looks, spending) - Math.pow((double) k / looks, spending));
      assertEquals(crossing[k], planned, 1e-6, "look " + k);
      spent += crossing[k];
    }
    assertEquals(spent, alpha, 1e-6);
  }

  /** 判定の間隔は時点の数を決めるだけで、臨界値は計画の牌山の数に依らない。 */
  @Test
  public void theCheckIntervalOnlySetsTheNumberOfLooks() {
    Sequential sequential = new Sequential(0.01, 3, 512);
    assertEquals(sequential.looks(4096), 8);
    double[] critical = sequential.boundaries(4096);
    double[] perWall = new Sequential(0.01, 3, 1).boundaries(8);
    for (int k = 0; k < critical.length; k++) {
      assertEquals(critical[k], perWall[k], 0, "look " + k);
    }
  }

  @Test
  public void criticalValuesMatchTheConvergedReference() {
    double[] critical = new Sequential(0.01, 3, 1).boundaries(8);
    for (int k = 0; k < critical.length; k++) {
      assertEquals(critical[k], CONVERGED_RHO3_8[k], 1e-4, "look " + k);
    }
  }

  /** 設計の表は格子が粗いので、後の時点ほど値が少し大きめに出る（最大 0.002）。 */
  @Test
  public void criticalValuesMatchTheDesignTable() {
    assertTable(1, 3.227, 3.184, 3.134, 3.089, 3.047, 3.010, 2.977, 2.946);
    assertTable(2, 3.781, 3.482, 3.304, 3.167, 3.053, 2.954, 2.865, 2.785);
    assertTable(3, 3.781, 3.254, 2.919, 2.655);
    assertTable(3, 4.270, 3.808, 3.526, 3.312, 3.133, 2.977, 2.837, 2.710);
    assertTable(
        3, 4.713, 4.296, 4.051, 3.867, 3.717, 3.587, 3.473, 3.370, 3.275, 3.187, 3.106, 3.028, 2.955, 2.886, 2.820,
        2.756);
  }

  @Test
  public void criticalValuesFallLookByLookAndEndAboveTheFixedLevel() {
    for (double spending : new double[] {1, 2, 3}) {
      double[] critical = new Sequential(0.01, spending, 1).boundaries(16);
      for (int k = 1; k < critical.length; k++) {
        assertTrue(critical[k] < critical[k - 1], "rho " + spending + " look " + k);
      }
      assertTrue(critical[critical.length - 1] > FIXED_99, "rho " + spending);
    }
  }

  @Test
  public void oneLookIsTheFixedLevel() {
    assertEquals(new Sequential(0.01, 3, 4096).boundaries(4096)[0], FIXED_99, 1e-9);
    assertEquals(new Sequential(0.05, 1, 7).boundaries(7)[0], 1.959963984540054, 1e-9);
  }

  /** 固定の 99% 区間を等間隔に何度も見ると、どこかで 0 を外れる確率は 8 回で約 0.042、16 回で約 0.058 になり、有意水準を超える。 */
  @Test
  public void repeatedFixedIntervalsSpendMoreThanAlpha() {
    assertEquals(sum(Sequential.crossings(constant(8, FIXED_99))), 0.042, 0.002);
    assertEquals(sum(Sequential.crossings(constant(16, FIXED_99))), 0.058, 0.002);
    assertEquals(sum(Sequential.crossings(constant(1, FIXED_99))), 0.01, 1e-9);
  }

  @Test
  public void invalidPlansAreRejected() {
    assertThrows(IllegalArgumentException.class, () -> new Sequential(0, 3, 512));
    assertThrows(IllegalArgumentException.class, () -> new Sequential(1, 3, 512));
    assertThrows(IllegalArgumentException.class, () -> new Sequential(Double.NaN, 3, 512));
    assertThrows(IllegalArgumentException.class, () -> new Sequential(0.01, 0, 512));
    assertThrows(IllegalArgumentException.class, () -> new Sequential(0.01, Double.POSITIVE_INFINITY, 512));
    assertThrows(IllegalArgumentException.class, () -> new Sequential(0.01, 3, 0));
    Sequential sequential = new Sequential(0.01, 3, 512);
    assertThrows(IllegalArgumentException.class, () -> sequential.looks(4000));
    assertThrows(IllegalArgumentException.class, () -> sequential.looks(0));
    assertThrows(IllegalArgumentException.class, () -> sequential.boundaries(256));
    assertThrows(IllegalArgumentException.class, () -> Sequential.crossings(new double[] {3, 0}));
    assertThrows(IllegalArgumentException.class, () -> Sequential.crossings(new double[] {Double.NaN}));
  }

  private static void assertTable(double spending, double... expected) {
    double[] critical = new Sequential(0.01, spending, 1).boundaries(expected.length);
    for (int k = 0; k < expected.length; k++) {
      assertEquals(critical[k], expected[k], 0.0025, "rho " + spending + " with " + expected.length + " looks, look " + k);
    }
  }

  private static double[] constant(int looks, double value) {
    double[] critical = new double[looks];
    Arrays.fill(critical, value);
    return critical;
  }

  private static double sum(double[] values) {
    double sum = 0;
    for (double value : values) {
      sum += value;
    }
    return sum;
  }
}

package com.epsilon.prism.collect;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Φ の効用の空間の較正: 補正が 0 の表は Φ を変えず、低い Φ を持ち上げる表は 4 席の和を保ったまま低い席を上げる。段は終わった局（公開系列の
 * 最後から 2 つ目の局境界）で選び、区分の中心の間は線形に補う。
 */
public class PhiCalibrationTest {
  private static final double LO = -1, HI = 2.0 / 3.0;
  private static final int BINS = 24;
  private static final double[] PHI = {0.45, 0.2, -0.05, -0.6};

  /** 段 stage だけ、Φ の区分の中心 x に slope·(−x) を足す表（低い Φ ほど持ち上げる）。 */
  private static PhiCalibration calibration(int stage, double slope) {
    double[][] table = new double[9][BINS];
    for (int bin = 0; bin < BINS; bin++) {
      double center = LO + (bin + 0.5) * (HI - LO) / BINS;
      table[stage][bin] = -slope * center;
    }
    return new PhiCalibration(LO, HI, table);
  }

  private static float[] prefix(int endedKyoku, int nextKyoku) {
    float[] a = EpsilonGrpFeature.fromProgress(endedKyoku, 0, 0, new int[] {25000, 25000, 25000, 25000});
    float[] b = EpsilonGrpFeature.fromProgress(nextKyoku, 0, 0, new int[] {25000, 25000, 25000, 25000});
    float[] prefix = new float[a.length + b.length];
    System.arraycopy(a, 0, prefix, 0, a.length);
    System.arraycopy(b, 0, prefix, a.length, b.length);
    return prefix;
  }

  @Test
  public void correctionsKeepTheSumAndFollowTheEndedHand() {
    Assert.assertEquals(calibration(3, 0).apply(PHI, prefix(3, 4)), PHI, 1e-12);
    double[] moved = calibration(3, 0.05).apply(PHI, prefix(3, 4));
    double before = 0, after = 0;
    for (int seat = 0; seat < 4; seat++) {
      before += PHI[seat];
      after += moved[seat];
    }
    Assert.assertEquals(after, before, 1e-12);
    Assert.assertTrue(moved[3] > PHI[3] && moved[0] < PHI[0], "the low seat rises and the high seat falls");
    // 線形の補正（中心の間）なので、席の差は (1 − slope) 倍に縮む
    Assert.assertEquals(moved[0] - moved[3], (PHI[0] - PHI[3]) * 0.95, 1e-9);
    // 段は終わった局（3）で選ぶので、終わった局が 4 なら補正しない
    Assert.assertEquals(calibration(3, 0.05).apply(PHI, prefix(4, 5)), PHI, 1e-12);
  }
}

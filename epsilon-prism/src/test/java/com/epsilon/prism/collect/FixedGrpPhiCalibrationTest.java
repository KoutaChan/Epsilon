package com.epsilon.prism.collect;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Φ の温度の較正: 温度が空か 1 なら元のまま、T > 1 は席ごとに和 1 のまま軟らかくし、局番号は公開系列の最後の局境界から読む。 */
public class FixedGrpPhiCalibrationTest {
  private static final float[] PROBABILITIES = {
    0.7f, 0.2f, 0.07f, 0.03f, 0.1f, 0.5f, 0.3f, 0.1f, 0.1f, 0.2f, 0.4f, 0.3f, 0.1f, 0.1f, 0.23f, 0.57f
  };

  private static float[] prefix(int... kyokus) {
    float[] prefix = new float[0];
    for (int kyoku : kyokus) {
      float[] feature = EpsilonGrpFeature.fromProgress(kyoku, 0, 0, new int[] {25000, 25000, 25000, 25000});
      float[] next = Arrays.copyOf(prefix, prefix.length + feature.length);
      System.arraycopy(feature, 0, next, prefix.length, feature.length);
      prefix = next;
    }
    return prefix;
  }

  @Test
  public void temperaturesSoftenEachSeatAndFollowTheNextHand() {
    Assert.assertSame(FixedGrpPhi.calibrated(PROBABILITIES, prefix(0), new double[0]), PROBABILITIES);
    Assert.assertSame(FixedGrpPhi.calibrated(PROBABILITIES, prefix(0, 1), new double[] {2, 1}), PROBABILITIES);
    float[] soft = FixedGrpPhi.calibrated(PROBABILITIES, prefix(0, 1, 2), new double[] {1, 1, 2});
    for (int seat = 0; seat < 4; seat++) {
      double total = 0;
      for (int rank = 0; rank < 4; rank++) total += soft[seat * 4 + rank];
      Assert.assertEquals(total, 1.0, 1e-6);
    }
    Assert.assertTrue(soft[0] < PROBABILITIES[0] && soft[3] > PROBABILITIES[3]);
    // 局番号が表の長さを超えたら最後の温度を使う
    Assert.assertEquals(
        FixedGrpPhi.calibrated(PROBABILITIES, prefix(7, 9), new double[] {1, 2})[0], soft[0], 1e-6);
  }
}

package com.epsilon.grp;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 点数からなめらかに求める順位の周辺確率と、それを使う Φ を確かめる。 */
public class ScoreSmoothTest {
  private static final Variant FOUR = Variant.TENHOU_4P;

  @Test
  public void normalCdfMatchesKnownValues() {
    Assert.assertEquals(ScoreSmooth.normalCdf(0), 0.5, 1e-7);
    Assert.assertEquals(ScoreSmooth.normalCdf(1), 0.8413447460685429, 1e-7);
    Assert.assertEquals(ScoreSmooth.normalCdf(-1.96), 0.024997895148220435, 1e-7);
    Assert.assertEquals(ScoreSmooth.normalCdf(3), 0.9986501019683699, 1e-7);
  }

  @Test
  public void remainingRoundsCountsToSouthFour() {
    Assert.assertEquals(ScoreSmooth.remainingRounds(FOUR, 0), 8);
    Assert.assertEquals(ScoreSmooth.remainingRounds(FOUR, 7), 1);
    Assert.assertEquals(ScoreSmooth.remainingRounds(FOUR, 9), 1);
  }

  @Test
  public void marginalsAreDistributions() {
    Random random = new Random(7);
    double[] ranks = new double[16];
    for (int trial = 0; trial < 200; trial++) {
      GrpState state = GrpFixtures.state(FOUR, random);
      ScoreSmooth.marginals(state, ranks);
      for (int seat = 0; seat < 4; seat++) {
        double sum = 0;
        for (int r = 0; r < 4; r++) sum += ranks[4 * seat + r];
        Assert.assertEquals(sum, 1, 1e-9);
      }
      for (int r = 0; r < 4; r++) {
        double sum = 0;
        for (int seat = 0; seat < 4; seat++) sum += ranks[4 * seat + r];
        Assert.assertEquals(sum, 1, 1e-4, "rank " + r + " of " + Arrays.toString(state.scores()));
      }
    }
  }

  @Test
  public void equalScoresGiveTheMeanUtility() {
    double[] utilities = values(new int[] {25000, 25000, 25000, 25000}, 0);
    double mean = 0;
    for (int r = 0; r < 4; r++) mean += RankUtility.TENHOU.value(FOUR, r) / 4;
    for (double u : utilities) Assert.assertEquals(u, mean, 1e-6);
  }

  @Test
  public void utilityRisesWithOwnScoreAndSharpensNearTheEnd() {
    double previous = Double.NEGATIVE_INFINITY;
    for (int own = 10000; own <= 40000; own += 100) {
      int rest = (100000 - own) / 300 * 100;
      int[] scores = {own, rest, rest, 100000 - own - 2 * rest};
      double u = values(scores, 4)[0];
      Assert.assertTrue(u > previous, "not increasing at " + own);
      previous = u;
    }
    int[] lead = {28000, 24000, 24000, 24000};
    Assert.assertTrue(values(lead, 7)[0] > values(lead, 0)[0], "a lead is worth more with fewer rounds left");
    // 1 位が変わらない 1,000 点の差も値に出る（点数の順位だけで決めるなら差は 0）
    double small = values(new int[] {33000, 24000, 22000, 21000}, 2)[0];
    double big = values(new int[] {34000, 24000, 21000, 21000}, 2)[0];
    Assert.assertTrue(big > small);
  }

  @Test
  public void threePlayerIsNotMeasured() {
    Assert.assertThrows(IllegalArgumentException.class, () -> Phi.scoreSmooth(Variant.TENHOU_3P, RankUtility.TENHOU));
    GrpState three = new GrpState(Variant.TENHOU_3P, 0, 0, 0, new int[] {35_000, 35_000, 35_000});
    Assert.assertThrows(IllegalArgumentException.class, () -> ScoreSmooth.marginals(three, new double[9]));
  }

  private static double[] values(int[] scores, int round) {
    double[] out = new double[4];
    Phi.scoreSmooth(FOUR, RankUtility.TENHOU).values(new GrpState(FOUR, round, 0, 0, scores), out);
    return out;
  }
}

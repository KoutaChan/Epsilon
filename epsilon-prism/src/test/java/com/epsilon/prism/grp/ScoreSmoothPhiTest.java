package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.collect.HandValue;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ScoreSmoothPhiTest {
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;

  @Test
  public void normalCdfMatchesKnownValues() {
    Assert.assertEquals(ScoreSmoothPhi.normalCdf(0), 0.5, 1e-7);
    Assert.assertEquals(ScoreSmoothPhi.normalCdf(1), 0.8413447460685429, 1e-7);
    Assert.assertEquals(ScoreSmoothPhi.normalCdf(-1.96), 0.024997895148220435, 1e-7);
    Assert.assertEquals(ScoreSmoothPhi.normalCdf(3), 0.9986501019683699, 1e-7);
  }

  @Test
  public void remainingHandsCountsToSouthFour() {
    Assert.assertEquals(ScoreSmoothPhi.remainingHands(0), 8);
    Assert.assertEquals(ScoreSmoothPhi.remainingHands(7), 1);
    Assert.assertEquals(ScoreSmoothPhi.remainingHands(9), 1);
  }

  @Test
  public void rankProbabilitiesAreDistributions() {
    Random random = new Random(7);
    double[] ranks = new double[16];
    for (int trial = 0; trial < 200; trial++) {
      int[] scores = randomScores(random);
      ScoreSmoothPhi.rankProbabilities(scores, random.nextInt(9), ranks);
      for (int seat = 0; seat < 4; seat++) {
        double sum = 0;
        for (int r = 0; r < 4; r++) sum += ranks[4 * seat + r];
        Assert.assertEquals(sum, 1, 1e-9);
      }
      for (int r = 0; r < 4; r++) {
        double sum = 0;
        for (int seat = 0; seat < 4; seat++) sum += ranks[4 * seat + r];
        Assert.assertEquals(sum, 1, 1e-4, "rank " + r + " of " + Arrays.toString(scores));
      }
    }
  }

  @Test
  public void equalScoresGiveTheMeanUtility() {
    double[] utilities = ScoreSmoothPhi.utilities(new int[] {25000, 25000, 25000, 25000}, 0, PROFILE);
    double mean = 0;
    for (int r = 0; r < 4; r++) mean += PROFILE.utilityForRankDouble(r) / 4;
    for (double u : utilities) Assert.assertEquals(u, mean, 1e-6);
  }

  @Test
  public void utilityRisesWithOwnScoreAndSharpensNearTheEnd() {
    int[] scores = {25000, 25000, 25000, 25000};
    double previous = Double.NEGATIVE_INFINITY;
    for (int own = 10000; own <= 40000; own += 100) {
      scores[0] = own;
      double u = ScoreSmoothPhi.utilities(scores, 4, PROFILE)[0];
      Assert.assertTrue(u > previous, "not increasing at " + own);
      previous = u;
    }
    int[] lead = {28000, 24000, 24000, 24000};
    double early = ScoreSmoothPhi.utilities(lead, 0, PROFILE)[0];
    double late = ScoreSmoothPhi.utilities(lead, 7, PROFILE)[0];
    Assert.assertTrue(late > early, "a lead is worth more with fewer hands left");
    // 1 位を変えない 1,000 点の差も値に出る（点数の順位なら 0）
    double small = ScoreSmoothPhi.utilities(new int[] {33000, 24000, 22000, 21000}, 2, PROFILE)[0];
    double big = ScoreSmoothPhi.utilities(new int[] {34000, 24000, 21000, 21000}, 2, PROFILE)[0];
    Assert.assertTrue(big > small);
  }

  @Test
  public void handValueAndOfflineAgreeBitForBit() throws IOException {
    Random random = new Random(11);
    HandValue value = ScorePhi.handValue(PhiEntryKind.SCORE_SMOOTH, PROFILE);
    HandEndPhi offline = ScorePhi.offline(PhiEntryKind.SCORE_SMOOTH, PROFILE);
    for (int trial = 0; trial < 50; trial++) {
      int[] scores = randomScores(random);
      HandStart next = new HandStart(random.nextInt(10), random.nextInt(3), random.nextInt(2), scores);
      RecordedGame game = new RecordedGame(trial, List.of(next), scores);
      double[][] got = new double[1][];
      value.request(game.prefix(0), scores, utilities -> got[0] = utilities);
      Assert.assertEquals(got[0], offline.values(List.of(game.input(0))).utilities()[0]);
      Assert.assertEquals(ScoreSmoothPhi.kyokuOf(game.prefix(0)), next.kyokuIndex());
      // 同じ (点数, 次局) はメモで引き、値は同じで配列は呼び手ごとの写し
      got[0][0] = Double.NaN;
      double[][] again = new double[1][];
      value.request(game.prefix(0), scores.clone(), utilities -> again[0] = utilities);
      Assert.assertEquals(again[0], offline.values(List.of(game.input(0))).utilities()[0]);
    }
    Assert.assertEquals(value.stats().requests(), 100);
    Assert.assertEquals(value.stats().memoHits(), 50);
  }

  @Test
  public void scoreSmoothIsAScoreOnlyStartAndSource() {
    var start = RefitSchedulerTest.config("phiSource", "SELF_GRP", "phiStart", "SCORE_SMOOTH");
    start.validateTraining();
    Assert.assertTrue(start.scoreOnlyPhi());
    Assert.assertEquals(GateMode.of(PhiStart.SCORE_SMOOTH.kind()), GateMode.FIRST_REPLACEMENT);
    var source = RefitSchedulerTest.config("phiSource", "SCORE_SMOOTH");
    source.validateTraining();
    Assert.assertTrue(source.scoreOnlyPhi());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () ->
            RefitSchedulerTest.config(
                "phiSource", "SELF_GRP", "phiStart", "SCORE_SMOOTH", "grpCheckpoint", "/grp"));
  }

  private static int[] randomScores(Random random) {
    int[] scores = new int[4];
    int left = 100000;
    for (int seat = 0; seat < 3; seat++) {
      scores[seat] = 100 * random.nextInt(500);
      left -= scores[seat];
    }
    scores[3] = left;
    return scores;
  }
}

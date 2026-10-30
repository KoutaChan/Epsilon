package com.epsilon.ai.decision;

import org.testng.Assert;
import org.testng.annotations.Test;

public class EpsilonUtilityTargetsTest {
  @Test
  public void sharedDefinitionsPreserveFloatValuesAndExactDoubleRatios() {
    float[][] expected = {
      {1.5f, .5f, -.5f, -1.5f},
      {1, .2f, -.2f, -.6f},
      {.4f, .2f, 0, -1},
      {0, 0, 0, 0},
      {2f / 3, 1f / 3, 0, -1},
      {1, .2f, -.2f, -.6f},
      {1, 1f / 3, -1f / 3, -1}
    };
    for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
      Assert.assertEquals(profile.rankUtility(), expected[profile.ordinal()]);
      for (int rank = 0; rank < 4; rank++) {
        Assert.assertEquals(profile.utilityForRank(rank), expected[profile.ordinal()][rank]);
      }
      float[] floats = profile.rankUtility();
      floats[0] = Float.NaN;
      Assert.assertTrue(Float.isFinite(profile.utilityForRank(0)));
      Assert.assertTrue(Double.isFinite(profile.utilityForRankDouble(0)));
    }
    Assert.assertEquals(EpsilonUtilityProfile.TENHOU.utilityForRankDouble(0), 2.0 / 3.0);
    Assert.assertEquals(EpsilonUtilityProfile.TENHOU.utilityForRankDouble(1), 1.0 / 3.0);
  }

  /** AVERAGE_RANK の価値の定義は、Epsilon-Major の AVERAGE_RANK 版の書き出し（macro614・macro964）の manifest の値と 1 文字も違わない。 */
  @Test
  public void averageRankMatchesTheEpsilonMajorExport() {
    Assert.assertEquals(
        EpsilonDecisionHlGauss.fingerprint(EpsilonUtilityProfile.AVERAGE_RANK),
        "hl-gauss-v1-bins=101;sigma-bin=0.75;padding-sigmas=6.0;profile=AVERAGE_RANK;"
            + "utility=0x1.0p0,0x1.555556p-2,-0x1.555556p-2,-0x1.0p0;"
            + "minimum=-0x1.190b21642c859p0;maximum=0x1.190b21642c859p0;prior-floor=1.0E-8");
    Assert.assertEquals(
        EpsilonUtilityProfile.AVERAGE_RANK.ordinal(), EpsilonUtilityProfile.values().length - 1);
    Assert.assertEquals(EpsilonUtilityProfile.MLEAGUE.ordinal(), 5);
  }

  @Test
  public void doubleProjectionKeepsUtilityPrecision() {
    float[] probabilities = {0, 0, 0, 1, .1f, .2f, .3f, .4f};
    for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
      if (!profile.rankBased()) {
        Assert.assertEquals(EpsilonUtilityTargets.utilityDouble(profile, 35000, 3), .2);
        continue;
      }
      double expected = 0;
      for (int rank = 0; rank < 4; rank++) {
        expected += probabilities[4 + rank] * profile.utilityForRankDouble(rank);
        Assert.assertEquals(
            EpsilonUtilityTargets.utilityDouble(profile, 35000, rank),
            profile.utilityForRankDouble(rank));
      }
      Assert.assertEquals(
          EpsilonUtilityTargets.expectedRankUtilityTrustedDouble(profile, probabilities, 4),
          expected);
    }
  }

  @Test
  public void finalScoresUseCommonRankAndScoreDefinitionsIncludingTies() {
    int[] scores = {35000, 35000, 20000, 10000};
    int[] ranks = {0, 1, 2, 3};
    for (int seat = 0; seat < scores.length; seat++) {
      float[] expected = EpsilonUtilityTargets.fromScoresAndRanks(seat, scores, ranks);
      Assert.assertEquals(EpsilonUtilityTargets.fromFinalScores(seat, scores), expected);
      for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
        float value =
            profile == EpsilonUtilityProfile.SCORE
                ? new float[] {.2f, .2f, -.1f, -.3f}[seat]
                : profile.utilityForRank(ranks[seat]);
        Assert.assertEquals(expected[profile.ordinal()], value, 1e-7f);
        Assert.assertEquals(
            EpsilonUtilityTargets.fromFinalScores(profile, seat, scores), value, 1e-7f);
      }
    }
  }

  @Test
  public void rankProjectionNormalizesAndSupportsPackedSeatDistributions() {
    float[] probabilities = {.1f, .2f, .3f, .4f};
    float[] unnormalized = {1, 2, 3, 4};
    float[] packed = {1, 0, 0, 0, .1f, .2f, .3f, .4f, 0, 0, 0, 1};
    for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
      if (!profile.rankBased()) continue;
      float expected = 0;
      for (int rank = 0; rank < probabilities.length; rank++) {
        expected += probabilities[rank] * profile.utilityForRank(rank);
      }
      Assert.assertEquals(
          EpsilonUtilityTargets.expectedRankUtility(profile, unnormalized), expected);
      Assert.assertEquals(
          EpsilonUtilityTargets.expectedRankUtilityTrusted(profile, probabilities), expected);
      Assert.assertEquals(
          EpsilonUtilityTargets.expectedRankUtilityTrusted(profile, packed, 4), expected);
      for (int rank = 0; rank < probabilities.length; rank++) {
        Assert.assertEquals(
            EpsilonUtilityTargets.expectedRankUtility(
                profile, EpsilonUtilityTargets.oneHotRank(rank)),
            profile.utilityForRank(rank));
      }
    }
  }

  @Test
  public void invalidDistributionAndScoreProjectionAreRejected() {
    for (float[] invalid :
        new float[][] {new float[4], {1, 0, 0}, {Float.NaN, 0, 0, 1}, {-1, 1, 1, 1}}) {
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> EpsilonUtilityTargets.expectedRankUtility(EpsilonUtilityProfile.TENHOU, invalid));
    }
    Assert.expectThrows(
        IllegalArgumentException.class,
        () ->
            EpsilonUtilityTargets.expectedRankUtility(
                EpsilonUtilityProfile.SCORE, new float[] {1, 0, 0, 0}));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () ->
            EpsilonUtilityTargets.expectedRankUtilityTrusted(
                EpsilonUtilityProfile.SCORE, new float[] {1, 0, 0, 0}));
  }

  @Test
  public void storedProfileMustMatchTrainingObjective() {
    for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
      Assert.assertSame(EpsilonUtilityTargets.profile(profile.ordinal()), profile);
      EpsilonUtilityTargets.requireTrainingProfile(profile.ordinal(), profile);
    }
    Assert.expectThrows(IllegalArgumentException.class, () -> EpsilonUtilityTargets.profile(-1));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> EpsilonUtilityTargets.profile(EpsilonUtilityProfile.values().length));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () ->
            EpsilonUtilityTargets.requireTrainingProfile(
                EpsilonUtilityProfile.TENHOU.ordinal(), EpsilonUtilityProfile.TOP));
    Assert.expectThrows(IllegalArgumentException.class, () -> EpsilonUtilityTargets.oneHotRank(4));
  }
}

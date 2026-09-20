package com.epsilon.engine;

import com.epsilon.ai.decision.DecisionBranchOutcome;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRankPredictor;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DecisionBranchOutcomeTest {
  @Test
  public void hanchanEndUsesActualRankWithoutCallingGrp() {
    var settlement =
        new RoundSettlement(
            null, new int[] {10000, 40000, 30000, 20000}, RoundTransition.HanchanFinished.INSTANCE);
    EpsilonGrpRankPredictor unused =
        (sequence, seat) -> {
          throw new AssertionError("Terminal result must not use GRP");
        };
    Assert.assertEquals(
        DecisionBranchOutcome.utility(
                settlement, new float[0], 0, EpsilonUtilityProfile.TENHOU, unused)
            .join(),
        -1f);
    Assert.assertEquals(
        DecisionBranchOutcome.utility(
                settlement, new float[0], 1, EpsilonUtilityProfile.TENHOU, unused)
            .join(),
        2f / 3);
  }

  @Test
  public void continuationAppendsOnlyNextBoundaryToTheOriginalPrefix() {
    int[] scores = {10000, 40000, 30000, 20000};
    float[] prefix = EpsilonGrpFeature.fromProgress(7, 0, 0, scores);
    var settlement = new RoundSettlement(null, scores, new RoundTransition.NextRound(7, 3, 1, 0));
    EpsilonGrpRankPredictor grp =
        (sequence, seat) -> {
          Assert.assertEquals(sequence.length, prefix.length * 2);
          Assert.assertEquals(Arrays.copyOf(sequence, prefix.length), prefix);
          Assert.assertEquals(
              Arrays.copyOfRange(sequence, prefix.length, sequence.length),
              EpsilonGrpFeature.fromProgress(7, 1, 0, scores));
          return CompletableFuture.completedFuture(new float[] {.1f, .2f, .3f, .4f});
        };
    Assert.assertEquals(
        DecisionBranchOutcome.utility(settlement, prefix, 0, EpsilonUtilityProfile.TENHOU, grp)
            .join(),
        -.26666668f,
        1e-6f);
    Assert.assertEquals(prefix.length, EpsilonGrpFeature.FEATURE_SIZE);
  }
}

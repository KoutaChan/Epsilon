package com.epsilon.nano.ai.decision;

import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrainingTargetIdentity;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 探索補正付き価値の固定点と、探索がない場合のGAEへの一致を確認する。 */
public class EpsilonDecisionReturnsTest {

  @Test
  public void twoActionOverlapPolicyValueIsFixedPoint() {
    float[] rollout = {0.9f, 0.1f};
    float[] behavior = {0.5f, 0.5f};
    float[] boundary = {2.0f / 3.0f, -1.0f};
    float[] coefficient = new float[2];
    double overlapMass = 0.0;
    double overlapValue = 0.0;
    for (int action = 0; action < 2; action++) {
      coefficient[action] =
          EpsilonDecisionReturns.selectedTraceCoefficient(rollout[action], behavior[action], 0.1f);
      double overlap = behavior[action] * coefficient[action];
      overlapMass += overlap;
      overlapValue += overlap * boundary[action];
    }
    double betaValue = overlapValue / overlapMass;
    Assert.assertEquals(coefficient[0], 1.0f, 1e-6f);
    Assert.assertEquals(coefficient[1], 0.28f, 1e-6f);
    Assert.assertEquals(betaValue, 0.3020833333333333, 1e-6);

    double expectedTarget = 0.0;
    for (int action = 0; action < 2; action++) {
      float target =
          EpsilonDecisionReturns.scalarVTraceTarget(
              (float) betaValue, boundary[action], boundary[action], 0.95f, coefficient[action]);
      Assert.assertTrue(target >= -1.0f && target <= 2.0f / 3.0f);
      expectedTarget += behavior[action] * target;
    }
    Assert.assertEquals(expectedTarget, betaValue, 1e-6);
  }

  @DataProvider
  public Object[][] lambdas() {
    return new Object[][] {{0.0f}, {0.4f}, {0.95f}, {1.0f}};
  }

  @Test(dataProvider = "lambdas")
  public void onPolicyValueAndActorReturnsMatchGae(float lambda) {
    float[] values = {0.12f, 0.24f, -0.17f};
    float boundary = 2.0f / 3.0f;
    float nextValue = boundary;
    float nextTarget = boundary;
    float nextAdvantage = 0.0f;
    for (int index = values.length - 1; index >= 0; index--) {
      float lookahead = EpsilonDecisionReturns.actorLookaheadTarget(nextValue, nextTarget, lambda);
      float advantage = lookahead - values[index];
      float expectedAdvantage = nextValue - values[index] + lambda * nextAdvantage;
      Assert.assertEquals(advantage, expectedAdvantage, 1e-6f);
      float target =
          EpsilonDecisionReturns.scalarVTraceTarget(
              values[index], nextValue, nextTarget, lambda, 1.0f);
      Assert.assertEquals(target, values[index] + advantage, 1e-6f);
      nextValue = values[index];
      nextTarget = target;
      nextAdvantage = advantage;
    }
  }

  @Test
  public void terminalCorrectionStaysInsideBoundaryAndRejectsOldTargetIdentity() {
    float target = EpsilonDecisionReturns.scalarVTraceTarget(0.5f, -1.0f, -1.0f, 0.95f, 0.28f);
    Assert.assertEquals(target, 0.08f, 1e-6f);
    Assert.assertTrue(target >= -1.0f && target <= 0.5f);
    Assert.assertEquals(
        EpsilonDecisionTrainingTargetIdentity.selectedPg(0.95f, 0.1f).targetProtocolVersion(), 8);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new EpsilonDecisionTrainingTargetIdentity(7, 0.95f, 0.1f));
  }
}

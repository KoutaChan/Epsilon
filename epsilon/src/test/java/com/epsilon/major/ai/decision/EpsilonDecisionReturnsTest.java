package com.epsilon.major.ai.decision;

import com.epsilon.major.ai.decision.data.EpsilonDecisionTrainingTargetIdentity;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 探索補正した Value 教師値の固定点とオンポリシー極限を確認する。 */
public class EpsilonDecisionReturnsTest {
  @Test
  public void clippedOverlapValueIsFixedPointInTwoActionCounterexample() {
    float[] rollout = {0.9f, 0.1f};
    float[] behavior = {0.5f, 0.5f};
    float[] boundary = {2.0f / 3.0f, -1.0f};
    float alpha = 0.1f;
    float lambda = 0.95f;
    float[] coefficient = {
      EpsilonDecisionReturns.selectedTraceCoefficient(rollout[0], behavior[0], alpha),
      EpsilonDecisionReturns.selectedTraceCoefficient(rollout[1], behavior[1], alpha)
    };
    Assert.assertEquals(coefficient[0], 1.0f, 1e-6f);
    Assert.assertEquals(coefficient[1], 0.28f, 1e-6f);
    float overlapMass = behavior[0] * coefficient[0] + behavior[1] * coefficient[1];
    float exactValue =
        (behavior[0] * coefficient[0] * boundary[0] + behavior[1] * coefficient[1] * boundary[1])
            / overlapMass;
    Assert.assertEquals(exactValue, 0.30208333f, 1e-6f);

    float expectedTerminalValue = 0.0f;
    float expectedPreviousValue = 0.0f;
    for (int action = 0; action < 2; action++) {
      float terminalValue =
          EpsilonDecisionReturns.scalarVTraceTarget(
              exactValue, boundary[action], boundary[action], lambda, coefficient[action]);
      expectedTerminalValue += behavior[action] * terminalValue;
      expectedPreviousValue +=
          behavior[action]
              * EpsilonDecisionReturns.scalarVTraceTarget(
                  exactValue, exactValue, terminalValue, lambda, 1.0f);
    }
    Assert.assertEquals(expectedTerminalValue, exactValue, 1e-6f);
    Assert.assertEquals(expectedPreviousValue, exactValue, 1e-6f);
  }

  @Test
  public void onPolicyTraceRecoversGae() {
    float lambda = 0.95f;
    float firstValue = 0.1f;
    float secondValue = 0.2f;
    float thirdValue = 0.3f;
    float boundary = 0.8f;
    float last =
        EpsilonDecisionReturns.scalarVTraceTarget(thirdValue, boundary, boundary, lambda, 1.0f);
    float middle =
        EpsilonDecisionReturns.scalarVTraceTarget(secondValue, thirdValue, last, lambda, 1.0f);
    float first =
        EpsilonDecisionReturns.scalarVTraceTarget(firstValue, secondValue, middle, lambda, 1.0f);
    float expectedAdvantage =
        (secondValue - firstValue)
            + lambda * (thirdValue - secondValue)
            + lambda * lambda * (boundary - thirdValue);
    Assert.assertEquals(first - firstValue, expectedAdvantage, 1e-6f);
    Assert.assertEquals(
        EpsilonDecisionReturns.actorLookaheadTarget(secondValue, middle, lambda) - firstValue,
        expectedAdvantage,
        1e-6f);
  }

  @Test
  public void clippedTargetStaysWithinValueSupport() {
    for (float current : new float[] {-1.0f, 1.0f}) {
      for (float next : new float[] {-1.0f, 1.0f}) {
        for (float target : new float[] {-1.0f, 1.0f}) {
          float result =
              EpsilonDecisionReturns.scalarVTraceTarget(current, next, target, 0.95f, 0.28f);
          Assert.assertTrue(result >= -1.0f && result <= 1.0f);
        }
      }
    }
  }

  @Test
  public void oldTargetProtocolCannotMixWithCorrectedFragments() {
    Assert.assertEquals(
        EpsilonDecisionTrainingTargetIdentity.selectedPg(0.95f, 0.1f).targetProtocolVersion(), 8);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new EpsilonDecisionTrainingTargetIdentity(7, 0.95f, 0.1f));
  }
}

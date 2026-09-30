package com.epsilon.pico.ai.decision;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrainingTargetIdentity;
import org.testng.Assert;
import org.testng.annotations.Test;

public class EpsilonDecisionReturnsTest {

  @Test
  public void overlapPolicyValueIsFixedPointForTwoActionExploration() {
    double[] behavior = {0.5, 0.5};
    float[] rollout = {0.9f, 0.1f};
    float[] outcome = {2.0f / 3.0f, -1.0f};
    float[] coefficient = new float[2];
    double overlapMass = 0.0;
    for (int action = 0; action < 2; action++) {
      coefficient[action] =
          EpsilonDecisionReturns.selectedTraceCoefficient(
              rollout[action], (float) behavior[action], 0.1f);
      overlapMass += behavior[action] * coefficient[action];
    }
    double fixedPoint =
        (behavior[0] * coefficient[0] * outcome[0] + behavior[1] * coefficient[1] * outcome[1])
            / overlapMass;
    Assert.assertEquals(coefficient[0], 1.0f);
    Assert.assertEquals(coefficient[1], 0.28f, 1e-6f);
    Assert.assertEquals(fixedPoint, 0.3020833333333333, 1e-6);

    double expectedTerminalTarget = 0.0;
    double expectedPrecedingTarget = 0.0;
    for (int action = 0; action < 2; action++) {
      float terminalTarget =
          EpsilonDecisionReturns.scalarVTraceTarget(
              (float) fixedPoint, outcome[action], outcome[action], 0.95f, coefficient[action]);
      expectedTerminalTarget += behavior[action] * terminalTarget;
      expectedPrecedingTarget +=
          behavior[action]
              * EpsilonDecisionReturns.scalarVTraceTarget(
                  (float) fixedPoint, (float) fixedPoint, terminalTarget, 0.95f, 1.0f);
      EpsilonDecisionReturns.requireValueTarget(terminalTarget, EpsilonUtilityProfile.TENHOU);
    }
    Assert.assertEquals(expectedTerminalTarget, fixedPoint, 1e-6);
    Assert.assertEquals(expectedPrecedingTarget, fixedPoint, 1e-6);
  }

  @Test
  public void onPolicyTraceMatchesOrdinaryGaeAndStopsAtBoundary() {
    float lambda = 0.8f;
    float boundary = -0.4f;
    float nextValue = 0.3f;
    float currentValue = 0.1f;
    float terminalValue =
        EpsilonDecisionReturns.scalarVTraceTarget(nextValue, boundary, boundary, lambda, 1.0f);
    float valueTarget =
        EpsilonDecisionReturns.scalarVTraceTarget(
            currentValue, nextValue, terminalValue, lambda, 1.0f);
    float advantage =
        EpsilonDecisionReturns.actorLookaheadTarget(nextValue, terminalValue, lambda)
            - currentValue;
    float expectedGae = (nextValue - currentValue) + lambda * (boundary - nextValue);
    Assert.assertEquals(terminalValue, boundary, 1e-6f);
    Assert.assertEquals(valueTarget - currentValue, expectedGae, 1e-6f);
    Assert.assertEquals(advantage, expectedGae, 1e-6f);
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

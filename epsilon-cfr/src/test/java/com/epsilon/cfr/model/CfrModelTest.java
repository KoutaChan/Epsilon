package com.epsilon.cfr.model;

import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrModelTest {
  @Test
  public void signedRegretMatchingRetainsExactZeros() {
    Assert.assertEquals(
        CfrModel.policy(new float[] {-3, 0, 2, 6}, 4, CfrModel.Role.REGRET),
        new double[] {0, 0, .25, .75});
    Assert.assertEquals(
        CfrModel.policy(new float[] {-3, 0, -2}, 3, CfrModel.Role.REGRET),
        new double[] {1d / 3, 1d / 3, 1d / 3});
    Assert.expectThrows(
        IllegalStateException.class,
        () -> CfrModel.policy(new float[] {Float.NaN}, 1, CfrModel.Role.REGRET));
  }

  static DecisionContext context(long seed) {
    var engine = new GameEngine(seed);
    var boundary = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var point = boundary.decisions().getFirst();
    return new DecisionContext(
        seed, point.id(), point.player(), engine.observation(point.player()), point.legalActions());
  }
}

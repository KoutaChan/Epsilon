package com.epsilon.ai.decision;

import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.core.Action;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DecisionBranchComparisonTest {
  private static DecisionBranchBudget budget() {
    return new DecisionBranchBudget(
        new DecisionBranchComparisonSettings(true, false, true, 1, 2, .01));
  }

  @Test
  public void budgetNeverBorrowsFutureInferenceRows() {
    var budget = budget();
    Assert.assertFalse(budget.admit(0));
    budget.recordMainRows(1000);
    Assert.assertTrue(budget.admit(0));
    Assert.assertTrue(budget.admit(0));
    Assert.assertFalse(budget.admit(0));
    Assert.assertFalse(budget.admit(1));
    Assert.assertTrue(budget.takeRows(10));
    Assert.assertFalse(budget.takeRows(1));
    budget.release();
    Assert.assertFalse(budget.admit(0));
    budget.recordMainRows(100);
    Assert.assertTrue(budget.admit(0));
    Assert.assertTrue(budget.takeRows(1));
  }

  @Test
  public void unfinishedKyushuNeverFallsBackToPpoAndCannotCompleteLate() {
    var pair = new DecisionBranchComparison(DecisionBranchGate.KYUSHU, .2f, true, budget());
    var pending = new CompletableFuture<Float>();
    pair.completeMain(CompletableFuture.completedFuture(.5f));
    pair.completeExtra(pending);
    Assert.assertEquals(pair.freeze(), DecisionBranchTarget.KYUSHU_ONLY);
    pending.complete(-.5f);
    Assert.assertEquals(pair.freeze(), DecisionBranchTarget.KYUSHU_ONLY);
  }

  @Test
  public void unfinishedWinRetainsNormalPpo() {
    var pair = new DecisionBranchComparison(DecisionBranchGate.RON, .8f, false, budget());
    Assert.assertEquals(pair.freeze(), DecisionBranchTarget.NONE);
  }

  @Test
  public void completedPairRetainsBothUtilitiesAndOrientation() {
    var pair = new DecisionBranchComparison(DecisionBranchGate.KYUSHU, .2f, false, budget());
    pair.completeMain(CompletableFuture.completedFuture(-.4f));
    pair.completeExtra(CompletableFuture.completedFuture(.3f));
    Assert.assertEquals(
        pair.freeze(), DecisionBranchTarget.completed(DecisionBranchGate.KYUSHU, .2f, .3f, -.4f));
  }

  @Test
  public void explorationAttributionExcludesLeafFloor() {
    // P(decline)=.2*.2+.8*.5=.44; leaf conditional probability=.25.
    // With three leaves and floor .01: leaf probability=.01+.97*.44*.25.
    float selected = .01f + .97f * .44f * .25f;
    double posterior = DecisionBranchSelection.explorationPosterior(.8f, selected, .8f, .01f);
    Assert.assertEquals(posterior * selected, .97 * .4 * .25, 1e-7);
    Assert.assertEquals(DecisionBranchSelection.explorationPosterior(.8f, selected, 0, .01f), 0.0);
  }

  @Test
  public void kyushuConditioningExcludesTsumoAndSamplesOnlyContinuation() {
    var actions =
        List.of(Action.tsumoAgari(), Action.kyushuKyuhai(), Action.dahai(0), Action.dahai(1));
    float[] policy = {.2f, .4f, .1f, .3f};
    Assert.assertEquals(
        DecisionBranchSelection.acceptanceProbability(DecisionBranchGate.KYUSHU, actions, policy),
        .5f,
        1e-6f);
    Assert.assertEquals(
        DecisionBranchSelection.sampleDecline(DecisionBranchGate.KYUSHU, actions, policy, .1), 2);
    Assert.assertEquals(
        DecisionBranchSelection.sampleDecline(DecisionBranchGate.KYUSHU, actions, policy, .9), 3);
  }

  @Test
  public void targetBinaryRoundTrip() throws Exception {
    for (var target :
        List.of(
            DecisionBranchTarget.NONE,
            DecisionBranchTarget.KYUSHU_ONLY,
            DecisionBranchTarget.completed(DecisionBranchGate.RON, .9f, -1, .1f))) {
      var bytes = new ByteArrayOutputStream();
      target.writeTo(new DataOutputStream(bytes));
      Assert.assertEquals(bytes.size(), DecisionBranchTarget.BYTES);
      Assert.assertEquals(
          DecisionBranchTarget.readFrom(
              new DataInputStream(new ByteArrayInputStream(bytes.toByteArray()))),
          target);
    }
  }
}

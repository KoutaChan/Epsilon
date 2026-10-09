package com.epsilon.cfr.collection;

import com.epsilon.ai.decision.DecisionBranchGate;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.core.Action;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrBranchSelectionTest {
  private static final DecisionBranchComparisonSettings ENABLED =
      new DecisionBranchComparisonSettings(true, true, true, 1, 8, .01);

  @Test
  public void ronAlwaysIncludesAcceptanceAndOnlyBranchesAfterDeclining() {
    var plan =
        CfrBranchSelection.plan(
            List.of(Action.ronAgari(), Action.pass(), Action.pon(0)),
            new double[] {.2, .3, .5},
            0,
            ENABLED);
    Assert.assertEquals(plan.gate(), DecisionBranchGate.RON);
    Assert.assertEquals(plan.afterSelected(0, .5), new CfrBranchSelection.Choice(1, -1, 0));
    Assert.assertEquals(plan.afterSelected(1, .5), new CfrBranchSelection.Choice(.3, 0, 1));
    Assert.assertEquals(plan.afterSelected(2, .5), new CfrBranchSelection.Choice(.5, 0, 1));
    assertEnumeratedInclusion(plan);
  }

  @Test
  public void kyushuIncludesBothDirectionsAndExcludesTsumo() {
    var plan =
        CfrBranchSelection.plan(
            List.of(Action.tsumoAgari(), Action.kyushuKyuhai(), Action.dahai(0), Action.dahai(1)),
            new double[] {.2, .4, .1, .3},
            0,
            ENABLED);
    Assert.assertEquals(plan.gate(), DecisionBranchGate.KYUSHU);
    Assert.assertEquals(plan.afterSelected(0, .5), new CfrBranchSelection.Choice(.2, -1, 0));
    Assert.assertEquals(plan.afterSelected(1, .1).alternativeSlot(), 2);
    Assert.assertEquals(plan.afterSelected(1, .9).alternativeSlot(), 3);
    Assert.assertEquals(plan.afterSelected(2, .5).alternativeSlot(), 1);
    Assert.assertEquals(plan.afterSelected(3, .5).alternativeSlot(), 1);
    assertEnumeratedInclusion(plan);
  }

  @Test
  public void zeroTargetAcceptanceIsStillExploredAndProbabilitiesRetainDoublePrecision() {
    double epsilon = 1e-12;
    var plan =
        CfrBranchSelection.plan(
            List.of(Action.kyushuKyuhai(), Action.dahai(0), Action.dahai(1)),
            new double[] {0, .1234567890123, .8765432109877},
            epsilon,
            ENABLED);
    Assert.assertEquals(plan.behavior()[0], epsilon / 3);
    Assert.assertEquals(plan.behavior()[1], (1 - epsilon) * .1234567890123 + epsilon / 3);
    assertEnumeratedInclusion(plan);
  }

  @Test
  public void ordinaryChoicesAndTsumoAreNotBranched() {
    Assert.assertNull(
        CfrBranchSelection.plan(
            List.of(Action.tsumoAgari(), Action.dahai(0)), new double[] {.5, .5}, .1, ENABLED));
    Assert.assertNull(
        CfrBranchSelection.plan(List.of(Action.ronAgari()), new double[] {1}, .1, ENABLED));
    Assert.assertNull(
        CfrBranchSelection.plan(
            List.of(Action.ronAgari(), Action.pass()),
            new double[] {.5, .5},
            .1,
            new DecisionBranchComparisonSettings(false, true, true, 1, 8, .01)));
  }

  @Test
  public void summedBehaviorRoundoffNeverMakesAnInclusionProbabilityExceedOne() {
    for (int count = 2; count <= 34; count++) {
      List<Action> actions = new ArrayList<>();
      actions.add(Action.kyushuKyuhai());
      for (int slot = 1; slot < count; slot++) actions.add(Action.dahai(slot - 1));
      for (int seed = 0; seed < 32; seed++) {
        SplittableRandom random = new SplittableRandom(seed);
        double[] sigma = new double[count];
        double sum = 0;
        for (int slot = 0; slot < count; slot++)
          sum += sigma[slot] = seed == 0 ? 1 : random.nextDouble();
        for (int slot = 0; slot < count; slot++) sigma[slot] /= sum;
        var plan = CfrBranchSelection.plan(actions, sigma, .1, ENABLED);
        for (int selected = 0; selected < count; selected++) {
          var choice = plan.afterSelected(selected, random.nextDouble());
          Assert.assertTrue(choice.mainInclusionProbability() <= 1);
          Assert.assertTrue(choice.mainInclusionProbability() >= plan.behavior()[selected]);
          Assert.assertTrue(choice.alternativeInclusionProbability() > 0);
          Assert.assertTrue(choice.alternativeInclusionProbability() <= 1);
          Assert.assertTrue(
              choice.alternativeInclusionProbability()
                  >= plan.behavior()[choice.alternativeSlot()]);
        }
      }
    }
  }

  /** 本経路の抽選と、それに応じて選ばれるすべての分岐先を列挙し、採用確率を検証する。 */
  private static void assertEnumeratedInclusion(CfrBranchSelection.Plan plan) {
    double[] behavior = plan.behavior();
    double[] observed = new double[behavior.length];
    for (int main = 0; main < behavior.length; main++) {
      observed[main] += behavior[main];
      if (plan.gate() == DecisionBranchGate.KYUSHU && main == plan.acceptanceSlot()) {
        double cumulative = 0;
        for (int alternate = 0; alternate < behavior.length; alternate++) {
          if (alternate == plan.acceptanceSlot() || alternate == plan.excludedSlot()) continue;
          double conditional = behavior[alternate] / plan.declineMass();
          var choice = plan.afterSelected(main, cumulative + conditional / 2);
          Assert.assertEquals(choice.alternativeSlot(), alternate);
          observed[alternate] += behavior[main] * conditional;
          cumulative += conditional;
        }
      } else {
        var choice = plan.afterSelected(main, .5);
        if (choice.alternativeSlot() >= 0) observed[choice.alternativeSlot()] += behavior[main];
      }
    }
    for (int main = 0; main < behavior.length; main++) {
      var choice = plan.afterSelected(main, .5);
      Assert.assertEquals(choice.mainInclusionProbability(), observed[main], 1e-14);
      if (choice.alternativeSlot() >= 0)
        Assert.assertEquals(
            choice.alternativeInclusionProbability(), observed[choice.alternativeSlot()], 1e-14);
    }
  }
}

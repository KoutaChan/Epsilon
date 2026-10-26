package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 規則事前 p0 の決め手の性質（和 1・全て正、まとまりの基準測度、打牌どうしの比）。 */
public class PriorTest {
  private static final PriorSettings SETTINGS = new PriorSettings(4.0, 0.05);

  private static float[] prior(List<Action> legal, float[] shanten, float[] ukeire) {
    int n = legal.size();
    byte[] group = new byte[n];
    ActionGroups.fill(legal, group);
    double[] z = new double[n];
    RulePrior.logits(SETTINGS, n, shanten, ukeire, group, z);
    return Prior.softmax(z, n);
  }

  private static void assertDistribution(float[] p) {
    double sum = 0;
    for (float value : p) {
      Assert.assertTrue(value > 0, "p0 must be positive");
      sum += value;
    }
    Assert.assertEquals(sum, 1.0, 1e-6);
  }

  @Test
  public void discardRatiosFollowShantenAndUkeire() {
    List<Action> legal =
        List.of(Action.dahai(0), Action.dahai(5), Action.riichiDahai(9), Action.tsumoAgari());
    float[] p = prior(legal, new float[] {1, 2, 1, 0}, new float[] {20, 40, 10, 0});
    assertDistribution(p);
    Assert.assertEquals(Math.log(p[0] / (double) p[1]), -4.0 * (1 - 2) + 0.05 * (20 - 40), 1e-4);
    Assert.assertEquals(Math.log(p[0] / (double) p[2]), 0.05 * (20 - 10), 1e-4);
    // 打牌のまとまり全体とツモ和了は半々
    Assert.assertEquals(p[0] + p[1] + p[2], p[3], 1e-6);
  }

  @Test
  public void groupsAreUniformWithoutDiscardChoice() {
    float[] p =
        prior(
            List.of(Action.dahai(3), Action.ankan(3), Action.tsumoAgari()),
            new float[] {0, 0, 0},
            new float[] {30, 0, 0});
    assertDistribution(p);
    for (float value : p) Assert.assertEquals(value, 1f / 3, 1e-6);

    float[] calls =
        prior(
            List.of(Action.chiSequence(0, 2), Action.chiSequence(1, 2), Action.pon(2), Action.pass()),
            new float[4],
            new float[4]);
    assertDistribution(calls);
    Assert.assertEquals(calls[3], 0.5f, 1e-6);
    for (int a = 0; a < 3; a++) Assert.assertEquals(calls[a], 1f / 6, 1e-6);
  }

  @Test
  public void onlyDiscardsKeepTheirRuleShares() {
    float[] p = prior(List.of(Action.dahai(0), Action.dahai(1)), new float[] {0, 1}, new float[] {10, 10});
    assertDistribution(p);
    Assert.assertEquals(p[0] / (double) p[1], Math.exp(4.0), 1e-3 * Math.exp(4.0));
  }
}

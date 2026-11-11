package com.epsilon.prism.policy;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import org.testng.Assert;
import org.testng.annotations.Test;

/** ルールベースの事前分布 p0 の大事な性質をテストする（全ての値が正で和が 1、手のまとまりごとの基準測度、打牌どうしの比）。 */
public class PriorTest {
  private static final PriorSettings SETTINGS = new PriorSettings(4.0, 0.05);

  private static int dahai(int kind) {
    return Action.discard(kind, false, false, false);
  }

  private static float[] prior(int[] legal, float[] shanten, float[] ukeire) {
    return Prior.compute(legal, shanten, ukeire, SETTINGS);
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
    int[] legal = {dahai(0), dahai(5), Action.discard(9, false, false, true), Action.code(Type.TSUMO_AGARI)};
    float[] p = prior(legal, new float[] {1, 2, 1, 0}, new float[] {20, 40, 10, 0});
    assertDistribution(p);
    Assert.assertEquals(Math.log(p[0] / (double) p[1]), -4.0 * (1 - 2) + 0.05 * (20 - 40), 1e-4);
    Assert.assertEquals(Math.log(p[0] / (double) p[2]), 0.05 * (20 - 10), 1e-4);
    // 打牌のまとまり全体とツモ和了で半々に分ける
    Assert.assertEquals(p[0] + p[1] + p[2], p[3], 1e-6);
  }

  @Test
  public void groupsAreUniformWithoutDiscardChoice() {
    float[] p =
        prior(
            new int[] {
              dahai(3), Action.code(Type.ANKAN, 3, TileSelection.NOT_APPLICABLE), Action.code(Type.TSUMO_AGARI)
            },
            new float[] {0, 0, 0},
            new float[] {30, 0, 0});
    assertDistribution(p);
    for (float value : p) Assert.assertEquals(value, 1f / 3, 1e-6);

    float[] calls =
        prior(
            new int[] {
              Action.chi(0, 2, false),
              Action.chi(1, 2, false),
              Action.code(Type.PON, 2, TileSelection.NON_AKA),
              Action.code(Type.PASS)
            },
            new float[4],
            new float[4]);
    assertDistribution(calls);
    Assert.assertEquals(calls[3], 0.5f, 1e-6);
    for (int a = 0; a < 3; a++) Assert.assertEquals(calls[a], 1f / 6, 1e-6);
  }

  @Test
  public void onlyDiscardsKeepTheirRuleShares() {
    float[] p = prior(new int[] {dahai(0), dahai(1)}, new float[] {0, 1}, new float[] {10, 10});
    assertDistribution(p);
    Assert.assertEquals(p[0] / (double) p[1], Math.exp(4.0), 1e-3 * Math.exp(4.0));
  }

  @Test
  public void oneLegalActionIsCertain() {
    Assert.assertEquals(prior(new int[] {dahai(0)}, new float[1], new float[1]), new float[] {1f});
  }
}

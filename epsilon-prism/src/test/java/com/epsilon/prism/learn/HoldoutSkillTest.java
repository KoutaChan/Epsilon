package com.epsilon.prism.learn;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 保留の指標の raw と debiased の対の量（{@link HoldoutSkill.Accumulator}）。網も入力の復号も使わない。 */
public class HoldoutSkillTest {
  /**
   * 差分整合の最適 s = Y + λ·log p0 + 定数では、debiased の g（u = s − λ·log p0 の差）は測った差 m に一致し（傾き 1、説明 1）、raw の g は事前の
   * 傾きの分ずれる。決着した行の一位は s のまま比べる。
   */
  @Test
  public void debiasedGapsRemoveThePriorTilt() {
    float lambda = 0.02f;
    float[] values = {0.10f, 0.05f, Float.NaN, 0.20f};
    float[] prior = {0.5f, 0.3f, 0.1f, 0.1f};
    float[] scores = new float[values.length];
    for (int a = 0; a < values.length; a++)
      scores[a] =
          Float.isNaN(values[a])
              ? -1f
              : (float) (values[a] + lambda * Math.log(prior[a]) + 0.7);
    var accumulator = new HoldoutSkill.Accumulator();
    accumulator.add(scores, values, prior, lambda);
    accumulator.add(scores, values, prior, lambda);
    HoldoutSkill.Group group = accumulator.group();
    Assert.assertEquals(group.pairs().rows(), 2);
    Assert.assertEquals(group.pairs().pairs(), 6);
    Assert.assertEquals(group.debiased().pairs(), 6);
    Assert.assertEquals(group.debiased().slope(), 1.0, 1e-5);
    Assert.assertEquals(group.debiased().explained(), 1.0, 1e-5);
    Assert.assertEquals(group.debiased().signAgreement(), 1.0, 0.0);
    Assert.assertTrue(
        Math.abs(group.pairs().slope() - 1.0) > 0.05, "raw slope " + group.pairs().slope());
    Assert.assertEquals(group.decided().rows(), 2);
    Assert.assertEquals(group.decided().rate(), 1.0, 0.0);
  }
}

package com.epsilon.prism.learn;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 錨の目標: 測った手は (1 − w)·Y + w·s_錨、測らない手は NaN のまま。重み 0 は Y そのもの。 */
public class TeacherTest {
  @Test
  public void anchorMixesMeasuredActionsOnly() {
    float[] values = {0.1f, Float.NaN, 0.3f};
    float[] anchor = {1f, 2f, 3f};
    float[] mixed = Teacher.anchored(values, anchor, 0.5);
    Assert.assertEquals(mixed[0], 0.55f, 1e-6f);
    Assert.assertTrue(Float.isNaN(mixed[1]));
    Assert.assertEquals(mixed[2], 1.65f, 1e-6f);
    Assert.assertEquals(Teacher.anchored(values, anchor, 0)[2], 0.3f, 0f);
    Assert.assertThrows(IllegalArgumentException.class, () -> Teacher.anchored(values, new float[] {1f}, 0.5));
  }
}

package com.epsilon.nano.ai.decision.input;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 保存時に使う行幅が、全容量区分で実際のテンソル配置と一致することを確かめる。 */
public class DecisionInputLayoutTest {
  @Test
  public void serializedElementCountsMatchDenseLayouts() {
    for (int actions : DecisionInputSchema.LEGAL_ACTION_BUCKETS) {
      for (int transitions : DecisionInputSchema.ACTION_TRANSITION_BUCKETS) {
        DecisionBucket bucket = new DecisionBucket(actions, transitions);
        for (int rows : new int[] {1, 3, 64}) {
          DecisionInputLayout layout = new DecisionInputLayout(rows, bucket);
          for (DecisionInputLayout.Slab slab : DecisionInputLayout.Slab.values()) {
            Assert.assertEquals(
                DecisionInputLayout.elementCount(slab, rows, bucket),
                layout.elementCount(slab),
                bucket + ", rows=" + rows + ", slab=" + slab);
          }
        }
      }
    }
  }

  @Test
  public void serializedElementCountsRejectOverflow() {
    Assert.expectThrows(
        ArithmeticException.class,
        () ->
            DecisionInputLayout.elementCount(
                DecisionInputLayout.Slab.CATEGORICAL,
                Integer.MAX_VALUE,
                new DecisionBucket(40, 16)));
  }
}

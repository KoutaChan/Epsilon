package com.epsilon.cfr.data;

import org.testng.Assert;
import org.testng.annotations.Test;

public final class CfrSampleIdTest {
  @Test
  public void retentionPriorityKeepsTheExistingReplayOrdering() {
    // 保存済み世代と新たな行が、同じシードで同じ保持順になることを固定値で確認する。
    Assert.assertEquals(new CfrSampleId(1, 0, 0, 0).retentionPriority(0), 0x0295a3217e9b2d3cL);
    Assert.assertEquals(new CfrSampleId(1, 1, 0, 0).retentionPriority(42), 0xd770553d190ef28eL);
    Assert.assertEquals(
        new CfrSampleId(5, 1234, 987654321, 3).retentionPriority(-1), 0xa57ab2d940b43d0fL);
    Assert.assertEquals(
        new CfrSampleId(Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, 2)
            .retentionPriority(Long.MAX_VALUE),
        0xca17019475ed1587L);
  }
}

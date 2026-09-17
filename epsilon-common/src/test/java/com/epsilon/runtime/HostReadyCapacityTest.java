package com.epsilon.runtime;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 入力バッチの容量解放通知と、終了時の待機解除・新規受け付け停止を検証する。 */
public class HostReadyCapacityTest {
  @Test
  public void releaseWakesTheRecordedCapacityGeneration() {
    var capacity = new DecisionDevicePipeline.HostReadyCapacity(4);
    for (int i = 0; i < 4; i++) Assert.assertTrue(capacity.tryAcquire());
    var blocked = capacity.availability();
    Assert.assertFalse(capacity.tryAcquire());
    var signal = capacity.release(true);
    Assert.assertSame(signal, blocked);
    signal.complete(null);
    Assert.assertTrue(blocked.isDone());
    Assert.assertFalse(capacity.availability().isDone());
    Assert.assertTrue(capacity.tryAcquire());
    for (int i = 0; i < 4; i++) capacity.release(true).complete(null);
    Assert.assertEquals(capacity.inUse(), 0);
  }

  @Test
  public void terminalFailureWakesBlockedAdmissionAndRejectsNewWork() {
    var capacity = new DecisionDevicePipeline.HostReadyCapacity(1);
    Assert.assertTrue(capacity.tryAcquire());
    var blocked = capacity.availability();
    var failure = new IllegalStateException("device failed");
    capacity.fail(failure);
    Assert.assertTrue(blocked.isCompletedExceptionally());
    Assert.expectThrows(IllegalStateException.class, () -> capacity.tryAcquire());
    Assert.assertNull(capacity.release(false));
  }
}

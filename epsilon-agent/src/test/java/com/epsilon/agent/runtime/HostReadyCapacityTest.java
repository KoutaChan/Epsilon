package com.epsilon.agent.runtime;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotSame;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import org.testng.annotations.Test;

/** パイプラインのホスト側のバッファについて、数え方、空いたときの知らせ方、壊れた後は受け付けないことを確かめる。 */
public class HostReadyCapacityTest {

  @Test
  public void aReleaseHandsTheWaitingSideTheCurrentSignal() {
    var capacity = new DecisionDevicePipeline.HostReadyCapacity(2);
    assertTrue(capacity.tryAcquire());
    assertTrue(capacity.tryAcquire());
    assertFalse(capacity.tryAcquire());
    assertEquals(capacity.inUse(), 2);

    CompletableFuture<Void> waiting = capacity.availability();
    CompletableFuture<Void> signal = capacity.release(true);
    assertSame(signal, waiting);
    assertNotSame(capacity.availability(), waiting);
    assertNull(capacity.release(false));
    assertEquals(capacity.inUse(), 0);
    expectThrows(IllegalStateException.class, () -> capacity.release(true));
  }

  @Test
  public void aFailureCompletesTheWaitingSideAndRefusesNewCells() {
    var capacity = new DecisionDevicePipeline.HostReadyCapacity(1);
    assertTrue(capacity.tryAcquire());
    CompletableFuture<Void> waiting = capacity.availability();
    IllegalStateException failure = new IllegalStateException("device lost");
    capacity.fail(failure);
    ExecutionException observed = expectThrows(ExecutionException.class, waiting::get);
    assertSame(observed.getCause(), failure);
    expectThrows(IllegalStateException.class, capacity::tryAcquire);
    assertNull(capacity.release(true));
    expectThrows(IllegalArgumentException.class, () -> new DecisionDevicePipeline.HostReadyCapacity(0));
  }
}

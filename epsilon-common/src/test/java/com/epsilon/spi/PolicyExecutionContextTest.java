package com.epsilon.spi;

import com.epsilon.runtime.MeasurementStatus;
import com.google.gson.Gson;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import org.testng.Assert;
import org.testng.annotations.Test;

/** CPUのみの実行状態と、並列符号化の書き込み回数・失敗時の借用バッファの有効期間を検証する。 */
public class PolicyExecutionContextTest {
  @Test
  public void cpuOnlyContextHasNoApplicableGpuMemoryMeasurement() {
    try (var context = new PolicyExecutionContext(1)) {
      var memory = context.memorySnapshot();
      Assert.assertEquals(memory.status(), MeasurementStatus.NOT_APPLICABLE);
      Assert.assertTrue(memory.devices().isEmpty());
      var json = new Gson().toJsonTree(memory).getAsJsonObject();
      Assert.assertEquals(json.get("status").getAsString(), "not-applicable");
    }
  }

  @Test
  public void writesEveryRowExactlyOnce() {
    AtomicIntegerArray visited = new AtomicIntegerArray(257);
    try (var context = new PolicyExecutionContext(3)) {
      context
          .encodeRows(
              visited.length(),
              (from, to) -> {
                for (int row = from; row < to; row++) visited.incrementAndGet(row);
              })
          .join();
      for (int row = 0; row < visited.length(); row++) Assert.assertEquals(visited.get(row), 1);
    }
  }

  @Test
  public void failedRowDoesNotReleaseBorrowedBatchBeforeOtherWritersFinish() throws Exception {
    CountDownLatch started = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    try (var context = new PolicyExecutionContext(2)) {
      var result =
          context.encodeRows(
              128,
              (from, to) -> {
                if (from == 0) throw new IllegalArgumentException("bad row");
                if (from == 64) {
                  started.countDown();
                  try {
                    release.await();
                  } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                  }
                }
              });
      try {
        Assert.assertTrue(started.await(5, TimeUnit.SECONDS));
        Assert.assertFalse(result.isDone());
      } finally {
        release.countDown();
      }
      Assert.expectThrows(CompletionException.class, result::join);
    }
  }

  @Test
  public void smallAsyncBatchReturnsBeforeEncodingAndCloseDrainsIt() throws Exception {
    var started = new CountDownLatch(1);
    var release = new CountDownLatch(1);
    var closed = new CountDownLatch(1);
    var encodingThread = new AtomicReference<Thread>();
    var context = new PolicyExecutionContext(1);
    Thread closer = null;
    try {
      var result =
          context.encodeRowsAsync(
              1,
              (from, to) -> {
                encodingThread.set(Thread.currentThread());
                Assert.assertEquals(from, 0);
                Assert.assertEquals(to, 1);
                started.countDown();
                try {
                  if (!release.await(5, TimeUnit.SECONDS))
                    throw new AssertionError("Encoding was not released");
                } catch (InterruptedException failure) {
                  Thread.currentThread().interrupt();
                  throw new IllegalStateException(failure);
                }
              });
      Assert.assertTrue(started.await(5, TimeUnit.SECONDS));
      Assert.assertNotSame(encodingThread.get(), Thread.currentThread());
      Assert.assertFalse(result.isDone());
      closer =
          Thread.ofPlatform()
              .start(
                  () -> {
                    context.close();
                    closed.countDown();
                  });
      Assert.assertFalse(closed.await(25, TimeUnit.MILLISECONDS));
      release.countDown();
      result.join();
      Assert.assertTrue(closed.await(5, TimeUnit.SECONDS));
      Assert.expectThrows(
          IllegalStateException.class, () -> context.encodeRowsAsync(1, (from, to) -> {}));
    } finally {
      release.countDown();
      if (closer != null) closer.join(5000);
      context.close();
    }
  }

  @Test
  public void asyncFailureWaitsForEveryRangeAndPreservesExactCoverage() throws Exception {
    var visited = new AtomicIntegerArray(257);
    var lastStarted = new CountDownLatch(1);
    var releaseLast = new CountDownLatch(1);
    try (var context = new PolicyExecutionContext(3)) {
      var result =
          context.encodeRowsAsync(
              visited.length(),
              (from, to) -> {
                for (int row = from; row < to; row++) visited.incrementAndGet(row);
                if (from == 0) throw new IllegalArgumentException("bad first range");
                if (to == visited.length()) {
                  lastStarted.countDown();
                  try {
                    if (!releaseLast.await(5, TimeUnit.SECONDS))
                      throw new AssertionError("Last range was not released");
                  } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(failure);
                  }
                }
              });
      try {
        Assert.assertTrue(lastStarted.await(5, TimeUnit.SECONDS));
        Assert.assertFalse(result.isDone());
      } finally {
        releaseLast.countDown();
      }
      Assert.expectThrows(CompletionException.class, result::join);
      for (int row = 0; row < visited.length(); row++) Assert.assertEquals(visited.get(row), 1);
    }
  }
}

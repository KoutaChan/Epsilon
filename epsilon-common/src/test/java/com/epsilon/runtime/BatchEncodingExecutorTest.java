package com.epsilon.runtime;

import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;
import jdk.jfr.Recording;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 並列符号化で各行を一度だけ書き、失敗やキャンセル時も書き込み中の領域を回収しないことを検証する。 */
public class BatchEncodingExecutorTest {
  @DataProvider
  public Object[][] workerCounts() {
    return new Object[][] {{1}, {4}};
  }

  @Test(dataProvider = "workerCounts", timeOut = 10000)
  public void parallelEncodingWritesEachRowOnceInInputOrder(int workers) {
    var finishes = new AtomicInteger();
    var rows = IntStream.range(0, 513).boxed().toList();
    var writes = new AtomicIntegerArray(rows.size());
    try (var executor =
        new BatchEncodingExecutor<String, int[], int[]>(
            workers,
            (size, key) -> new int[size],
            storage -> {
              finishes.incrementAndGet();
              return storage;
            })) {
      int[] result =
          executor
              .encodeAsync(
                  rows,
                  "shape",
                  (sources, storage, from, to) -> {
                    for (int row = from; row < to; row++) {
                      storage[row] = sources.get(row);
                      writes.incrementAndGet(row);
                    }
                  })
              .join();
      Assert.assertEquals(result, IntStream.range(0, 513).toArray());
      for (int row = 0; row < rows.size(); row++) Assert.assertEquals(writes.get(row), 1);
      Assert.assertEquals(finishes.get(), 1);
    }
  }

  @DataProvider
  public Object[][] profilingModes() {
    return new Object[][] {{false}, {true}};
  }

  @Test(dataProvider = "profilingModes", timeOut = 10000)
  public void failedEncodingWaitsForOtherWritersBeforeReportingCompletion(boolean profiling)
      throws Exception {
    try (var recording = encodingRecording(profiling)) {
      var writerStarted = new CountDownLatch(1);
      var finishWriter = new CountDownLatch(1);
      var finishes = new AtomicInteger();
      var rows = IntStream.range(0, 128).boxed().toList();
      try (var executor =
          new BatchEncodingExecutor<String, int[], int[]>(
              2,
              (size, key) -> new int[size],
              storage -> {
                finishes.incrementAndGet();
                return storage;
              })) {
        var result =
            executor.encodeAsync(
                rows,
                "shape",
                (sources, storage, from, to) -> {
                  if (from == 0) throw new IllegalArgumentException("bad row");
                  if (from == 64) {
                    writerStarted.countDown();
                    try {
                      finishWriter.await();
                    } catch (InterruptedException failure) {
                      Thread.currentThread().interrupt();
                      throw new IllegalStateException(failure);
                    }
                  }
                  for (int row = from; row < to; row++) storage[row] = sources.get(row);
                });
        try {
          Assert.assertTrue(writerStarted.await(5, TimeUnit.SECONDS));
          Assert.assertFalse(result.isDone());
        } finally {
          finishWriter.countDown();
        }
        Assert.expectThrows(CompletionException.class, result::join);
        Assert.assertEquals(finishes.get(), 0);
      }
    }
  }

  @Test
  public void storageAllocationRunsOffCallerAndItsFailureCompletesTheFuture() {
    Thread caller = Thread.currentThread();
    var allocator = new AtomicReference<Thread>();
    var writes = new AtomicInteger();
    var finishes = new AtomicInteger();
    var failure = new IllegalArgumentException("allocation failed");
    try (var executor =
        new BatchEncodingExecutor<String, int[], int[]>(
            1,
            (size, key) -> {
              allocator.set(Thread.currentThread());
              throw failure;
            },
            storage -> {
              finishes.incrementAndGet();
              return storage;
            })) {
      var result =
          executor.encodeAsync(
              java.util.List.of(7),
              "shape",
              (sources, storage, from, to) -> writes.incrementAndGet());
      CompletionException actual = Assert.expectThrows(CompletionException.class, result::join);
      Assert.assertSame(actual.getCause(), failure);
      Assert.assertNotSame(allocator.get(), caller);
      Assert.assertEquals(writes.get(), 0);
      Assert.assertEquals(finishes.get(), 0);
    }
  }

  @Test(dataProvider = "profilingModes", timeOut = 10000)
  public void cancellingReturnedFutureStillLetsCloseDrainTheAcceptedBatch(boolean profiling)
      throws Exception {
    try (var recording = encodingRecording(profiling)) {
      var allocationStarted = new CountDownLatch(1);
      var finishAllocation = new CountDownLatch(1);
      var closeStarted = new CountDownLatch(1);
      var closeFinished = new CountDownLatch(1);
      var writes = new AtomicInteger();
      var finishes = new AtomicInteger();
      var rows = IntStream.range(0, 128).boxed().toList();
      var executor =
          new BatchEncodingExecutor<String, int[], int[]>(
              2,
              (size, key) -> {
                allocationStarted.countDown();
                try {
                  finishAllocation.await();
                } catch (InterruptedException failure) {
                  Thread.currentThread().interrupt();
                  throw new IllegalStateException(failure);
                }
                return new int[size];
              },
              storage -> {
                finishes.incrementAndGet();
                return storage;
              });
      var result =
          executor.encodeAsync(
              rows,
              "shape",
              (sources, storage, from, to) -> {
                for (int row = from; row < to; row++) {
                  storage[row] = sources.get(row);
                  writes.incrementAndGet();
                }
              });
      Thread closer = null;
      try {
        Assert.assertTrue(allocationStarted.await(5, TimeUnit.SECONDS));
        Assert.assertTrue(result.cancel(false));
        closer =
            Thread.ofVirtual()
                .start(
                    () -> {
                      closeStarted.countDown();
                      executor.close();
                      closeFinished.countDown();
                    });
        Assert.assertTrue(closeStarted.await(5, TimeUnit.SECONDS));
        Assert.assertFalse(closeFinished.await(100, TimeUnit.MILLISECONDS));
      } finally {
        finishAllocation.countDown();
        if (closer != null) closer.join(5000);
        executor.close();
      }
      Assert.assertEquals(closeFinished.getCount(), 0);
      Assert.assertEquals(writes.get(), rows.size());
      Assert.assertEquals(finishes.get(), 1);
      Assert.assertTrue(result.isCancelled());
    }
  }

  private static Recording encodingRecording(boolean enabled) {
    var recording = new Recording();
    if (enabled) recording.enable("epsilon.InputEncoding");
    else recording.disable("epsilon.InputEncoding");
    recording.start();
    return recording;
  }
}

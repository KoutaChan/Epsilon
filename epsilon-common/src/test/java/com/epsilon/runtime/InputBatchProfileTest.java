package com.epsilon.runtime;

import com.epsilon.config.settings.InferenceBatchingSettings;
import java.nio.file.Files;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** JFR診断の有無で入力や失敗の意味を変えず、バッチ単位の内訳だけを記録することを検証する。 */
public class InputBatchProfileTest {
  @Test
  public void disabledEventsDoNotFormatBucketsOrAllocateEvents() {
    Object bucket =
        new Object() {
          @Override
          public String toString() {
            throw new AssertionError("Disabled profiling must not format a bucket");
          }
        };
    try (var recording = new Recording()) {
      recording.disable("epsilon.InputSelection");
      recording.disable("epsilon.InputEncoding");
      recording.disable("epsilon.InputStage");
      recording.disable("epsilon.InputCollection");
      recording.start();
      Assert.assertNull(InputBatchProfile.beginSelection());
      Assert.assertNull(InputBatchProfile.beginEncoding(bucket, 1, 1, -1, "TOTAL"));
      Assert.assertNull(InputBatchProfile.beginStage("test", bucket, 1, "copy"));
      Assert.assertNull(InputBatchProfile.beginCollection(1, 0));
      try (var executor =
          new BatchEncodingExecutor<Object, int[], int[]>(
              1, (rows, key) -> new int[rows], storage -> storage)) {
        Assert.assertEquals(
            executor
                .encodeAsync(
                    List.of(7),
                    bucket,
                    (sources, storage, from, to) -> {
                      for (int row = from; row < to; row++) storage[row] = sources.get(row);
                    })
                .join(),
            new int[] {7});
      }
    }
  }

  @Test
  public void selectionRecordsDispatchReasonsAndWaitForEveryRemovedRow() throws Exception {
    var releases = new AtomicInteger();
    Object model = new Object();
    var admission =
        new InferenceAdmission<Object, String, Integer>(
            new InferenceBatchingSettings(1),
            (e, key) -> 2,
            (e, key) ->
                DecisionInferenceIngress.Attempt.acquired(
                    new DecisionInferenceIngress(e, null, releases::incrementAndGet)),
            e -> true);
    try (var recording = recording("epsilon.InputSelection")) {
      admission.add(model, "shape", 1, 0);
      admission.add(model, "shape", 2, 200);
      admission.add(model, "shape", 3, 250);
      var full = admission.startNextEncoding(300, false);
      Assert.assertEquals(full.rows(), List.of(1, 2));
      admission.encodingFailed(full);
      var supply = admission.startNextEncoding(400, false);
      Assert.assertEquals(supply.rows(), List.of(3));
      admission.encodingFailed(supply);
      admission.add(model, "rare", 4, 500);
      var deadline = admission.startNextEncoding(1500, false);
      Assert.assertEquals(deadline.rows(), List.of(4));
      admission.encodingFailed(deadline);
      Assert.assertNull(admission.startNextEncoding(1500, false));
      Assert.assertEquals(releases.get(), 3);
      Assert.assertFalse(admission.hasInFlightBatches());
      Assert.assertFalse(admission.hasQueuedRows());

      List<RecordedEvent> events = readEvents(recording);
      Assert.assertEquals(events.size(), 4);
      RecordedEvent first = events.get(0);
      Assert.assertTrue(first.getBoolean("admitted"));
      Assert.assertEquals(first.getString("dispatchReason"), "FULL");
      Assert.assertEquals(first.getInt("rows"), 2);
      Assert.assertEquals(first.getInt("capacity"), 2);
      Assert.assertEquals(first.getLong("oldestRowWaitNanos"), 300);
      Assert.assertEquals(first.getLong("totalRowWaitNanos"), 400);
      Assert.assertEquals(events.get(1).getString("dispatchReason"), "SUPPLY");
      Assert.assertEquals(events.get(1).getLong("totalRowWaitNanos"), 150);
      Assert.assertEquals(events.get(2).getString("dispatchReason"), "DEADLINE");
      Assert.assertEquals(events.get(2).getLong("totalRowWaitNanos"), 1000);
      Assert.assertFalse(events.get(3).getBoolean("admitted"));
    }
  }

  @DataProvider
  public Object[][] workerCounts() {
    return new Object[][] {{1}, {4}};
  }

  @Test(dataProvider = "workerCounts", timeOut = 10000)
  public void encodingRecordsDisjointTaskRowsWithoutChangingOutput(int workers) throws Exception {
    var rows = IntStream.range(0, 513).boxed().toList();
    var writes = new AtomicInteger();
    try (var recording = recording("epsilon.InputEncoding")) {
      try (var executor =
          new BatchEncodingExecutor<String, int[], int[]>(
              workers, (size, key) -> new int[size], storage -> storage)) {
        int[] encoded =
            executor
                .encodeAsync(
                    rows,
                    "shape",
                    (sources, storage, from, to) -> {
                      for (int row = from; row < to; row++) {
                        storage[row] = sources.get(row);
                        writes.incrementAndGet();
                      }
                    })
                .join();
        Assert.assertEquals(encoded, IntStream.range(0, 513).toArray());
      }
      Assert.assertEquals(writes.get(), rows.size());
      List<RecordedEvent> events = readEvents(recording);
      Assert.assertEquals(events.stream().filter(e -> phase(e, "TOTAL")).count(), 1);
      Assert.assertEquals(events.stream().filter(e -> phase(e, "ALLOCATE")).count(), 1);
      Assert.assertEquals(events.stream().filter(e -> phase(e, "FINISH")).count(), 1);
      Assert.assertEquals(
          events.stream().filter(e -> phase(e, "ROWS")).mapToInt(e -> e.getInt("rows")).sum(),
          rows.size());
      for (RecordedEvent event : events) {
        Assert.assertEquals(event.getString("bucket"), "shape");
        Assert.assertEquals(event.getInt("batchRows"), rows.size());
        Assert.assertTrue(event.getBoolean("success"));
        Assert.assertTrue(event.getLong("queueWaitNanos") >= 0);
        Assert.assertTrue(event.getLong("threadCpuNanos") >= -1);
        if (phase(event, "TOTAL")) Assert.assertEquals(event.getLong("threadCpuNanos"), -1);
      }
    }
  }

  @Test
  public void allocationFailureRecordsFailedTotalWithoutEncodingRows() throws Exception {
    var failure = new IllegalArgumentException("allocation failed");
    try (var recording = recording("epsilon.InputEncoding")) {
      try (var executor =
          new BatchEncodingExecutor<String, int[], int[]>(
              1,
              (size, key) -> {
                throw failure;
              },
              storage -> storage)) {
        var result =
            executor.encodeAsync(
                List.of(7),
                "shape",
                (sources, storage, from, to) -> {
                  for (int row = from; row < to; row++) storage[row] = sources.get(row);
                });
        Assert.assertSame(
            Assert.expectThrows(CompletionException.class, result::join).getCause(), failure);
      }
      List<RecordedEvent> events = readEvents(recording);
      Assert.assertEquals(events.size(), 2);
      Assert.assertEquals(events.stream().filter(e -> phase(e, "TOTAL")).count(), 1);
      Assert.assertEquals(events.stream().filter(e -> phase(e, "ALLOCATE")).count(), 1);
      Assert.assertTrue(events.stream().noneMatch(e -> e.getBoolean("success")));
    }
  }

  private static Recording recording(String name) {
    Recording recording = new Recording();
    recording.enable(name).withThreshold(Duration.ZERO).withoutStackTrace();
    recording.start();
    return recording;
  }

  private static List<RecordedEvent> readEvents(Recording recording) throws Exception {
    recording.stop();
    var file = Files.createTempFile("epsilon-input-profile-", ".jfr");
    try {
      recording.dump(file);
      return RecordingFile.readAllEvents(file);
    } finally {
      Files.delete(file);
    }
  }

  private static boolean phase(RecordedEvent event, String name) {
    return event.getString("phase").equals(name);
  }
}

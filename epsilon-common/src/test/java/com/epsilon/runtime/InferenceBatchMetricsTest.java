package com.epsilon.runtime;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.expectThrows;

import org.testng.annotations.Test;

/** モデルと容量別のバッチ計測、完了待ち後の集計リセット、失敗時の時間集計を検証する。 */
public class InferenceBatchMetricsTest {
  @Test
  public void physicalBatchesKeepTheirCapacityAndModelIdentity() {
    InferenceBatchMetrics metrics = new InferenceBatchMetrics();
    Object first = new Object();
    Object second = new Object();
    metrics.reset(100);
    var partial = new DecisionDevicePipeline.BatchMetadata(first, "wide", "policy", 3, 4);
    var full = new DecisionDevicePipeline.BatchMetadata(first, "narrow", "policy", 2, 2);
    var other = new DecisionDevicePipeline.BatchMetadata(second, "wide", "policy", 1, 4);
    metrics.accepted(110);
    metrics.accepted(111);
    metrics.accepted(112);
    metrics.submitted(partial, 2, 3, 1, 4);
    metrics.submitted(full, 3, 4, 1, 3);
    metrics.submitted(other, 4, 5, 1, 2);
    metrics.finished(partial, true, 20, 130, true, 5, 2, 1, 2);
    metrics.finished(full, true, 21, 132, true, 5, 2, 1, 2);
    metrics.finished(other, true, 22, 134, true, 5, 2, 1, 2);
    var snapshot = metrics.snapshot("gpu:0", 140);
    assertEquals(snapshot.batches().size(), 3);
    assertEquals(
        snapshot.batches().stream().mapToLong(DecisionDevicePipeline.BatchMetrics::rows).sum(), 6);
    assertEquals(
        snapshot.batches().stream().mapToLong(DecisionDevicePipeline.BatchMetrics::capacity).sum(),
        10);
    assertEquals(
        snapshot.batches().stream()
            .mapToLong(DecisionDevicePipeline.BatchMetrics::fullBatches)
            .sum(),
        1);
    assertEquals(snapshot.pipelineIdleNanos(), 16);
    assertEquals(snapshot.batches().getFirst().roundTripNanos(), 20);
    var firstBatch = snapshot.batches().getFirst();
    assertEquals(firstBatch.observedCompletions(), 1);
    assertEquals(
        firstBatch.readyWaitNanos()
            + firstBatch.stageNanos()
            + firstBatch.submitQueueWaitNanos()
            + firstBatch.submitNanos()
            + firstBatch.completionWaitNanos()
            + firstBatch.composeNanos()
            + firstBatch.reclaimWaitNanos()
            + firstBatch.releaseNanos(),
        firstBatch.roundTripNanos());
  }

  @Test
  public void resetRequiresDrainAndKeepsFailedStageOutOfSubmittedBatches() {
    InferenceBatchMetrics metrics = new InferenceBatchMetrics();
    metrics.reset(100);
    metrics.accepted(110);
    expectThrows(IllegalStateException.class, () -> metrics.reset(115));
    var metadata = new DecisionDevicePipeline.BatchMetadata(new Object(), "key", "policy", 4, 4);
    metrics.finished(metadata, false, 10, 120, false, 0, 0, 0, 0);
    assertEquals(metrics.snapshot("gpu:0", 130).batches().size(), 0);
    metrics.reset(130);
    assertEquals(metrics.snapshot("gpu:0", 140).pipelineIdleNanos(), 10);
  }

  @Test
  public void submittedFailureDoesNotDiluteObservedCompletionTimings() {
    InferenceBatchMetrics metrics = new InferenceBatchMetrics();
    var metadata = new DecisionDevicePipeline.BatchMetadata(new Object(), "key", "policy", 4, 4);
    metrics.reset(100);
    metrics.accepted(110);
    metrics.accepted(111);
    metrics.submitted(metadata, 2, 3, 1, 4);
    metrics.submitted(metadata, 3, 4, 2, 5);
    metrics.finished(metadata, true, 20, 130, true, 5, 2, 1, 2);
    metrics.finished(metadata, true, 25, 136, false, 0, 0, 0, 0);
    var batch = metrics.snapshot("gpu:0", 140).batches().getFirst();
    assertEquals(batch.batches(), 2);
    assertEquals(batch.observedCompletions(), 1);
    assertEquals(batch.submitQueueWaitNanos(), 3);
    assertEquals(batch.submitNanos(), 9);
    assertEquals(batch.completionWaitNanos(), 5);
    assertEquals(batch.composeNanos(), 2);
    assertEquals(batch.reclaimWaitNanos(), 1);
    assertEquals(batch.releaseNanos(), 2);
    assertEquals(batch.roundTripNanos(), 45);
    metrics.reset(140);
    assertEquals(metrics.snapshot("gpu:0", 150).batches().size(), 0);
  }
}

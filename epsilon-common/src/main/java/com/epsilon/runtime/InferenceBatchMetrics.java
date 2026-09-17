package com.epsilon.runtime;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

/** 同期やテンソルの読み出しを追加せず、ホストで観測した実行バッチを集計する。 */
final class InferenceBatchMetrics {
  private final Map<Key, Counts> groups = new LinkedHashMap<>();
  private long started = System.nanoTime();
  private long idleStarted = started;
  private long idleNanos;
  private int pending;
  private int maxHostReadyCells;

  synchronized void accepted(long now) {
    if (pending++ == 0) idleNanos += now - idleStarted;
  }

  synchronized void submitted(
      DecisionDevicePipeline.BatchMetadata metadata,
      long readyWaitNanos,
      long stageNanos,
      long submitQueueWaitNanos,
      long submitNanos) {
    if (metadata == null) return;
    Counts counts =
        groups.computeIfAbsent(
            new Key(metadata.model(), metadata.bucket(), metadata.resultKind()),
            ignored -> new Counts());
    counts.batches++;
    counts.rows += metadata.rows();
    counts.capacity += metadata.capacity();
    if (metadata.rows() == metadata.capacity()) counts.fullBatches++;
    counts.readyWaitNanos += readyWaitNanos;
    counts.stageNanos += stageNanos;
    counts.submitQueueWaitNanos += submitQueueWaitNanos;
    counts.submitNanos += submitNanos;
  }

  synchronized void finished(
      DecisionDevicePipeline.BatchMetadata metadata,
      boolean submitted,
      long elapsed,
      long now,
      boolean completionObserved,
      long completionWaitNanos,
      long composeNanos,
      long reclaimWaitNanos,
      long releaseNanos) {
    if (submitted && metadata != null) {
      Counts counts =
          groups.get(new Key(metadata.model(), metadata.bucket(), metadata.resultKind()));
      counts.roundTripNanos += elapsed;
      if (completionObserved) {
        counts.observedCompletions++;
        counts.completionWaitNanos += completionWaitNanos;
        counts.composeNanos += composeNanos;
        counts.reclaimWaitNanos += reclaimWaitNanos;
        counts.releaseNanos += releaseNanos;
      }
    }
    if (--pending == 0) idleStarted = now;
  }

  synchronized void hostReadyCells(int inUse) {
    maxHostReadyCells = Math.max(maxHostReadyCells, inUse);
  }

  synchronized void reset(long now) {
    if (pending != 0)
      throw new IllegalStateException("Inference metrics require a drained pipeline");
    groups.clear();
    started = now;
    idleStarted = now;
    idleNanos = 0;
    maxHostReadyCells = 0;
  }

  synchronized DecisionDevicePipeline.Metrics snapshot(String device, long now) {
    ArrayList<DecisionDevicePipeline.BatchMetrics> batches = new ArrayList<>(groups.size());
    groups.forEach(
        (key, value) ->
            batches.add(
                new DecisionDevicePipeline.BatchMetrics(
                    key.model.getClass().getName()
                        + '@'
                        + Integer.toHexString(System.identityHashCode(key.model)),
                    key.bucket.toString(),
                    key.resultKind.toString(),
                    value.batches,
                    value.rows,
                    value.capacity,
                    value.fullBatches,
                    value.readyWaitNanos,
                    value.stageNanos,
                    value.submitQueueWaitNanos,
                    value.submitNanos,
                    value.observedCompletions,
                    value.completionWaitNanos,
                    value.composeNanos,
                    value.reclaimWaitNanos,
                    value.releaseNanos,
                    value.roundTripNanos)));
    return new DecisionDevicePipeline.Metrics(
        device,
        now - started,
        idleNanos + (pending == 0 ? now - idleStarted : 0),
        maxHostReadyCells,
        batches);
  }

  private record Key(Object model, Object bucket, Object resultKind) {}

  private static final class Counts {
    long batches, rows, capacity, fullBatches, readyWaitNanos, stageNanos, roundTripNanos;
    long submitQueueWaitNanos, submitNanos, observedCompletions;
    long completionWaitNanos, composeNanos, reclaimWaitNanos, releaseNanos;
  }
}

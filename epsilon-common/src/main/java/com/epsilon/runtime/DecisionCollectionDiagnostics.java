package com.epsilon.runtime;

import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 収集区間の既存ホスト計測を、系列に依存せず記録する。 */
public final class DecisionCollectionDiagnostics {

  private static final Logger log = LoggerFactory.getLogger(DecisionCollectionDiagnostics.class);

  private DecisionCollectionDiagnostics() {}

  /** 収集の推論を回収した後、評価器の利用権を閉じる前に呼ぶ。 */
  public static void log(DecisionExecutionContext context, InferenceAdmission.Metrics admission) {
    long admittedBatches =
        admission.fullBatches() + admission.supplyBatches() + admission.deadlineBatches();
    InferenceDispatcher.Metrics dispatch = context.inferenceDispatchMetrics();
    log.info(
        "Decision collection admission: fullBatches={} fullRows={} supplyBatches={} supplyRows={}"
            + " deadlineBatches={} deadlineRows={} meanOldestRowWaitMs={} maxOldestRowWaitMs={}"
            + " hostCapacity={} peakHostReservations={}",
        admission.fullBatches(),
        admission.fullRows(),
        admission.supplyBatches(),
        admission.supplyRows(),
        admission.deadlineBatches(),
        admission.deadlineRows(),
        format(admission.totalOldestRowWaitNanos() * 1e-6 / admittedBatches),
        format(admission.maxOldestRowWaitNanos() * 1e-6),
        dispatch.hostCapacity(),
        dispatch.peakHostReservations());
    for (DecisionDevicePipeline.Metrics device : context.inferenceMetrics()) {
      long batches = 0;
      long rows = 0;
      long capacity = 0;
      long fullBatches = 0;
      long readyWaitNanos = 0;
      long stageNanos = 0;
      long submitQueueWaitNanos = 0;
      long submitNanos = 0;
      long observedCompletions = 0;
      long completionWaitNanos = 0;
      long composeNanos = 0;
      long reclaimWaitNanos = 0;
      long releaseNanos = 0;
      long roundTripNanos = 0;
      for (DecisionDevicePipeline.BatchMetrics batch : device.batches()) {
        batches += batch.batches();
        rows += batch.rows();
        capacity += batch.capacity();
        fullBatches += batch.fullBatches();
        readyWaitNanos += batch.readyWaitNanos();
        stageNanos += batch.stageNanos();
        submitQueueWaitNanos += batch.submitQueueWaitNanos();
        submitNanos += batch.submitNanos();
        observedCompletions += batch.observedCompletions();
        completionWaitNanos += batch.completionWaitNanos();
        composeNanos += batch.composeNanos();
        reclaimWaitNanos += batch.reclaimWaitNanos();
        releaseNanos += batch.releaseNanos();
        roundTripNanos += batch.roundTripNanos();
      }
      log.info(
          "Decision collection device: device={} elapsedMs={} pipelineIdleMs={}"
              + " pipelineIdleFraction={} batches={} rows={} avgBatch={} fillRatio={}"
              + " fullBatches={} meanReadyWaitMs={} meanStageMs={} meanSubmitQueueWaitMs={}"
              + " meanHostSubmitMs={} observedCompletions={} meanCompletionObservedWaitMs={}"
              + " meanComposeMs={} meanReclaimWaitMs={} meanReleaseMs={} meanRoundTripMs={}"
              + " maxHostReadyCells={}",
          device.device(),
          format(device.elapsedNanos() * 1e-6),
          format(device.pipelineIdleNanos() * 1e-6),
          format((double) device.pipelineIdleNanos() / device.elapsedNanos()),
          batches,
          rows,
          format((double) rows / batches),
          format((double) rows / capacity),
          fullBatches,
          format(readyWaitNanos * 1e-6 / batches),
          format(stageNanos * 1e-6 / batches),
          format(submitQueueWaitNanos * 1e-6 / batches),
          format(submitNanos * 1e-6 / batches),
          observedCompletions,
          format(completionWaitNanos * 1e-6 / observedCompletions),
          format(composeNanos * 1e-6 / observedCompletions),
          format(reclaimWaitNanos * 1e-6 / observedCompletions),
          format(releaseNanos * 1e-6 / observedCompletions),
          format(roundTripNanos * 1e-6 / batches),
          device.maxHostReadyCells());
      log.debug(
          "Decision collection device batches: device={} batches={}",
          device.device(),
          device.batches());
    }
  }

  private static String format(double value) {
    return String.format(Locale.ROOT, "%.3f", value);
  }
}

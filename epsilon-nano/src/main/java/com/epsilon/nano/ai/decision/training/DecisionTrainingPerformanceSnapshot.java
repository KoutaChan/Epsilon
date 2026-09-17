package com.epsilon.nano.ai.decision.training;

import java.util.Locale;

/** 1回の学習データ走査で観測した処理量と経過時間。 */
public record DecisionTrainingPerformanceSnapshot(
    double elapsedMillis,
    double fusedTotalElapsedMillis,
    int optimizerSteps,
    int microBatches,
    long samples,
    int fragments,
    long descriptorSamples,
    int configuredSamples,
    int prefetchDepth,
    int prefetchWorkers) {

  public String summary() {
    return String.format(
        Locale.ROOT,
        "DecisionUpdatePerf{elapsedMs=%.1f,fusedTotalElapsedMs=%.1f,optimizerSteps=%d,"
            + "microBatches=%d,samples=%d,fragments=%d,descriptors=%d,configuredSamples=%d,"
            + "prefetchDepth=%d,prefetchWorkers=%d}",
        elapsedMillis,
        fusedTotalElapsedMillis,
        optimizerSteps,
        microBatches,
        samples,
        fragments,
        descriptorSamples,
        configuredSamples,
        prefetchDepth,
        prefetchWorkers);
  }

  @Override
  public String toString() {
    return summary();
  }
}

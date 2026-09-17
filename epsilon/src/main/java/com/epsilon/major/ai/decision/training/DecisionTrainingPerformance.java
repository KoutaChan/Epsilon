package com.epsilon.major.ai.decision.training;

/** ホスト側のDecision 更新処理量と経過時間を記録します。 */
final class DecisionTrainingPerformance {

  private final long startedAt = System.nanoTime();
  private final int configuredSamples;
  private final int prefetchDepth;
  private final int prefetchWorkers;
  private int optimizerSteps;
  private int microBatches;
  private long samples;
  private int fragments;
  private long descriptorSamples;

  DecisionTrainingPerformance(int configuredSamples, int prefetchDepth, int prefetchWorkers) {
    this.configuredSamples = configuredSamples;
    this.prefetchDepth = prefetchDepth;
    this.prefetchWorkers = prefetchWorkers;
  }

  int prefetchDepth() {
    return prefetchDepth;
  }

  int prefetchWorkers() {
    return prefetchWorkers;
  }

  void microBatch(int count) {
    microBatches++;
    samples += count;
  }

  void optimizerStep() {
    optimizerSteps++;
  }

  void fragmentDescriptorsRead(int count) {
    fragments++;
    descriptorSamples += count;
  }

  DecisionTrainingPerformanceSnapshot snapshot() {
    double elapsedMillis = (System.nanoTime() - startedAt) / 1_000_000.0;
    return snapshot(elapsedMillis, elapsedMillis);
  }

  DecisionTrainingPerformanceSnapshot snapshot(
      double elapsedMillis, double fusedTotalElapsedMillis) {
    return new DecisionTrainingPerformanceSnapshot(
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
}

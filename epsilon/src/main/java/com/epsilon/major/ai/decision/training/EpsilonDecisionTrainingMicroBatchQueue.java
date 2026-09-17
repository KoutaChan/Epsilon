package com.epsilon.major.ai.decision.training;

import com.epsilon.major.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 自己対局の学習サンプルを形状ごとに蓄積し、同じ容量区分の小バッチへまとめる。
 *
 * <p>一つのデバイス側バッチへ異なる行動/遷移容量区分を混在させると、それぞれの最大値を掛けた不要な密な テンソルが生じます。このキューは容量区分を混在させず、全GPUを合わせた行上限と、1
 * GPUで許容する行動-遷移 格納枠上限の両方を満たす時点で小バッチを確定します。
 */
final class EpsilonDecisionTrainingMicroBatchQueue {

  private final int maximumRowsPerDevice;
  private final int maximumDeviceTransitionCells;
  private final Map<DecisionBucket, ArrayList<EpsilonDecisionTrainingSampleDescriptor>>
      samplesByBucket = new LinkedHashMap<>();
  private int size;

  EpsilonDecisionTrainingMicroBatchQueue(
      int globalMicroBatchRows, int deviceCount, int maximumDeviceTransitionCells) {
    if (globalMicroBatchRows < deviceCount) {
      throw new IllegalArgumentException(
          "global microbatch rows must be at least the device count: rows="
              + globalMicroBatchRows
              + " devices="
              + deviceCount);
    }
    if (maximumDeviceTransitionCells <= 0) {
      throw new IllegalArgumentException("maximum device transition cells must be positive");
    }
    this.maximumRowsPerDevice = globalMicroBatchRows / deviceCount;
    this.maximumDeviceTransitionCells = maximumDeviceTransitionCells;
  }

  Optional<ArrayList<EpsilonDecisionTrainingSampleDescriptor>> add(
      EpsilonDecisionTrainingSampleDescriptor sample) {
    DecisionBucket bucket = sample.bucket();
    ArrayList<EpsilonDecisionTrainingSampleDescriptor> bucketSamples =
        samplesByBucket.computeIfAbsent(bucket, ignored -> new ArrayList<>(rowLimit(bucket)));
    bucketSamples.add(sample);
    size++;
    if (bucketSamples.size() < rowLimit(bucket)) {
      return Optional.empty();
    }
    samplesByBucket.remove(bucket);
    size -= bucketSamples.size();
    return Optional.of(bucketSamples);
  }

  List<ArrayList<EpsilonDecisionTrainingSampleDescriptor>> drain() {
    ArrayList<ArrayList<EpsilonDecisionTrainingSampleDescriptor>> batches =
        new ArrayList<>(samplesByBucket.size());
    for (ArrayList<EpsilonDecisionTrainingSampleDescriptor> samples : samplesByBucket.values()) {
      batches.add(samples);
    }
    samplesByBucket.clear();
    size = 0;
    return batches;
  }

  int size() {
    return size;
  }

  int rowLimit(DecisionBucket bucket) {
    int transitionCellsPerRow =
        Math.multiplyExact(bucket.legalActionCapacity(), bucket.actionTransitionCapacity());
    return Math.min(
        maximumRowsPerDevice, Math.max(1, maximumDeviceTransitionCells / transitionCellsPerRow));
  }
}

package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 自己対局の学習サンプルを、同じ行動数・遷移数の容量区分ごとに蓄積する。
 *
 * <p>異なる区分を混在させて不要に大きなテンソルを作ることを避ける。全 GPU の合計行数と、1台の GPU が保持できる行動数と遷移数の積の両方に上限を設け、小バッチを確定する。
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

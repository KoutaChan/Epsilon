package com.epsilon.major.ai.decision.training;

import com.epsilon.major.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/** 1行サンプルを同じ容量区分バッチへ束ね、推論結果を元の行順へ戻す。 */
public final class DecisionSampleBatcher {

  private DecisionSampleBatcher() {}

  static List<BatchSlice> partitionByBucket(
      List<EpsilonDecisionSample> samples, EpsilonDecisionEvaluator evaluator) {
    LinkedHashMap<DecisionBucket, ArrayList<IndexedSample>> samplesByBucket = new LinkedHashMap<>();
    for (int sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
      EpsilonDecisionSample sample = samples.get(sampleIndex);
      samplesByBucket
          .computeIfAbsent(sample.input().bucket(), ignored -> new ArrayList<>())
          .add(new IndexedSample(sampleIndex, sample));
    }

    ArrayList<BatchSlice> batchSlices = new ArrayList<>();
    for (Map.Entry<DecisionBucket, ArrayList<IndexedSample>> entry : samplesByBucket.entrySet()) {
      int maximumBatchSize = evaluator.preferredStreamingBatchSize(entry.getKey());
      if (maximumBatchSize < 1) {
        throw new IllegalStateException("Decision evaluator returned a non-positive batch size");
      }
      ArrayList<IndexedSample> bucketSamples = entry.getValue();
      for (int start = 0; start < bucketSamples.size(); start += maximumBatchSize) {
        int size = Math.min(maximumBatchSize, bucketSamples.size() - start);
        int[] sampleIndexes = new int[size];
        ArrayList<DecisionHostBatch> sampleRows = new ArrayList<>(size);
        for (int row = 0; row < size; row++) {
          IndexedSample indexed = bucketSamples.get(start + row);
          sampleIndexes[row] = indexed.sampleIndex();
          sampleRows.add(indexed.sample().input());
        }
        batchSlices.add(new BatchSlice(sampleIndexes, DecisionHostBatch.concatenate(sampleRows)));
      }
    }
    return batchSlices;
  }

  public static List<EpsilonDecisionInferenceServer.Prediction> evaluateInBucketedBatches(
      EpsilonDecisionEvaluator evaluator, List<EpsilonDecisionSample> samples) {
    if (samples.isEmpty()) {
      return List.of();
    }
    EpsilonDecisionInferenceServer.Prediction[] predictions =
        new EpsilonDecisionInferenceServer.Prediction[samples.size()];
    List<BatchSlice> batches = partitionByBucket(samples, evaluator);
    evaluateAsynchronously(evaluator, batches, predictions);
    for (EpsilonDecisionInferenceServer.Prediction prediction : predictions) {
      if (prediction == null) {
        throw new IllegalStateException("Decision evaluator did not return every sample row");
      }
    }
    return List.of(predictions);
  }

  private static void evaluateAsynchronously(
      EpsilonDecisionEvaluator evaluator,
      List<BatchSlice> batches,
      EpsilonDecisionInferenceServer.Prediction[] predictions) {
    ArrayDeque<PendingBatch> pending = new ArrayDeque<>();
    try {
      for (BatchSlice batch : batches) {
        DecisionInferenceIngress.Attempt attempt;
        while (true) {
          attempt = evaluator.tryAcquireInferenceIngress();
          if (attempt.acquired()) {
            break;
          }
          if (pending.isEmpty()) {
            attempt.available().join();
            continue;
          }
          complete(pending.removeFirst(), predictions);
        }
        pending.addLast(
            new PendingBatch(batch, evaluator.submitBatch(batch.hostBatch(), attempt.ingress())));
      }
      while (!pending.isEmpty()) {
        complete(pending.removeFirst(), predictions);
      }
    } catch (RuntimeException | Error failure) {
      drainPending(pending, predictions, failure);
      throw failure;
    }
  }

  private static void drainPending(
      ArrayDeque<PendingBatch> pending,
      EpsilonDecisionInferenceServer.Prediction[] predictions,
      Throwable primaryFailure) {
    while (!pending.isEmpty()) {
      try {
        complete(pending.removeFirst(), predictions);
      } catch (RuntimeException | Error cleanupFailure) {
        primaryFailure.addSuppressed(cleanupFailure);
      }
    }
  }

  private static void complete(
      PendingBatch pending, EpsilonDecisionInferenceServer.Prediction[] predictions) {
    scatter(pending.batch(), pending.predictions().join(), predictions);
  }

  private static void scatter(
      BatchSlice batch,
      List<EpsilonDecisionInferenceServer.Prediction> batchPredictions,
      EpsilonDecisionInferenceServer.Prediction[] destination) {
    if (batchPredictions.size() != batch.size()) {
      throw new IllegalStateException("Decision evaluator result count mismatch");
    }
    for (int row = 0; row < batch.size(); row++) {
      int sampleIndex = batch.sampleIndexes()[row];
      if (destination[sampleIndex] != null) {
        throw new IllegalStateException("Decision evaluator returned a sample row twice");
      }
      destination[sampleIndex] = batchPredictions.get(row);
    }
  }

  private record IndexedSample(int sampleIndex, EpsilonDecisionSample sample) {}

  record BatchSlice(int[] sampleIndexes, DecisionHostBatch hostBatch) {
    int size() {
      return sampleIndexes.length;
    }
  }

  private record PendingBatch(
      BatchSlice batch,
      CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> predictions) {}
}

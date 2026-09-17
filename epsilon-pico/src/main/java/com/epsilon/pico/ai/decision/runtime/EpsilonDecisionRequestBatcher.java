package com.epsilon.pico.ai.decision.runtime;

import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 合法手数と鳴き直後の打牌候補数で要求を分類し、同じ形状の推論バッチへまとめる。 */
public final class EpsilonDecisionRequestBatcher {

  private EpsilonDecisionRequestBatcher() {}

  public static <T> BatchMetrics evaluateRequests(
      EpsilonDecisionEvaluator evaluator,
      List<T> requests,
      RequestEncoder<? super T> requestEncoder,
      PredictionReceiver<? super T> predictionReceiver) {
    if (requests.isEmpty()) {
      return new BatchMetrics(0, 0L, 0);
    }

    Map<DecisionBucket, List<T>> requestsByBucket = new LinkedHashMap<>();
    for (T request : requests) {
      requestsByBucket
          .computeIfAbsent(requestEncoder.selectBucket(request), ignoredBucket -> new ArrayList<>())
          .add(request);
    }

    int inferenceCallCount = 0;
    long inferenceRequestCount = 0L;
    int maximumBatchSize = 0;
    for (Map.Entry<DecisionBucket, List<T>> bucketEntry : requestsByBucket.entrySet()) {
      int preferredBatchSize = evaluator.preferredBatchSize(bucketEntry.getKey());
      List<T> bucketRequests = bucketEntry.getValue();
      for (int requestStart = 0;
          requestStart < bucketRequests.size();
          requestStart += preferredBatchSize) {
        int requestCount = Math.min(preferredBatchSize, bucketRequests.size() - requestStart);
        DecisionBatchBuilder batchBuilder =
            DecisionBatchBuilder.inference(requestCount, bucketEntry.getKey());
        for (int rowIndex = 0; rowIndex < requestCount; rowIndex++) {
          requestEncoder.encodeRow(bucketRequests.get(requestStart + rowIndex), batchBuilder);
        }
        DecisionHostBatch hostBatch = batchBuilder.build();

        List<EpsilonDecisionInferenceServer.Prediction> predictions =
            evaluator.evaluateBatch(hostBatch);
        if (predictions.size() != requestCount) {
          throw new IllegalStateException(
              "Prediction count mismatch: requested="
                  + requestCount
                  + " got="
                  + predictions.size());
        }
        for (int rowIndex = 0; rowIndex < requestCount; rowIndex++) {
          predictionReceiver.accept(
              bucketRequests.get(requestStart + rowIndex),
              predictions.get(rowIndex),
              hostBatch,
              rowIndex);
        }
        inferenceCallCount++;
        inferenceRequestCount += requestCount;
        maximumBatchSize = Math.max(maximumBatchSize, requestCount);
      }
    }
    return new BatchMetrics(inferenceCallCount, inferenceRequestCount, maximumBatchSize);
  }

  /** 実際に発行した型付きバッチの件数、総行数、最大行数。 */
  public record BatchMetrics(int batchCount, long requestCount, int maximumBatchSize) {}

  public interface RequestEncoder<T> {
    DecisionBucket selectBucket(T request);

    void encodeRow(T request, DecisionBatchBuilder batchBuilder);
  }

  @FunctionalInterface
  public interface PredictionReceiver<T> {
    void accept(
        T request,
        EpsilonDecisionInferenceServer.Prediction prediction,
        DecisionHostBatch hostBatch,
        int rowIndex);
  }
}

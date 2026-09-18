package com.epsilon.nano.ai.decision.duel;

import com.epsilon.nano.ai.decision.arena.DecisionBatchEncoder;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionGreedyEvaluator;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

/** セッションが所有する評価器とエンコーダーを借用し、公開観測から最大確率の合法手位置を返す。 */
final class DuelPolicy implements BatchedPolicy {
  private final EpsilonDecisionGreedyEvaluator evaluator;
  private final DecisionBatchEncoder encoder;

  DuelPolicy(EpsilonDecisionGreedyEvaluator evaluator, DecisionBatchEncoder encoder) {
    this.evaluator = evaluator;
    this.encoder = encoder;
  }

  @Override
  public Object batchKey(DecisionRequest request) {
    return DecisionBatchBuilder.selectInferenceBucket(request.legalActions());
  }

  @Override
  public int maxBatchSize(Object key) {
    return evaluator.preferredStreamingBatchSize((DecisionBucket) key);
  }

  @Override
  public boolean needsInferenceWork() {
    return evaluator.needsInferenceWork();
  }

  @Override
  public Ingress tryAcquire(Object key, Runnable capacityAvailable) {
    var attempt = evaluator.tryAcquireInferenceIngress();
    if (!attempt.acquired()) {
      attempt.available().whenComplete((ignored, failure) -> capacityAvailable.run());
      return null;
    }
    return new Reservation((DecisionBucket) key, attempt.ingress());
  }

  private final class Reservation implements Ingress {
    private final DecisionBucket bucket;
    private final DecisionInferenceIngress ingress;
    private boolean consumed;

    Reservation(DecisionBucket bucket, DecisionInferenceIngress ingress) {
      this.bucket = bucket;
      this.ingress = ingress;
    }

    @Override
    public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
      synchronized (this) {
        if (consumed) throw new IllegalStateException("Duel reservation was already consumed");
        consumed = true;
      }
      CompletableFuture<DecisionHostBatch> encoded;
      try {
        encoded =
            encoder.encodeAsync(
                requests,
                bucket,
                (rows, builder, from, to) -> {
                  try (var session = builder.openEncoding()) {
                    for (int row = from; row < to; row++) {
                      DecisionRequest request = rows.get(row);
                      session.encodeObservedInferenceRow(
                          row, request.observation(), request.legalActions());
                    }
                  }
                });
      } catch (RuntimeException | Error failure) {
        ingress.abort();
        throw failure;
      }
      return encoded
          .handle(
              (batch, failure) -> {
                if (failure != null) {
                  ingress.abort();
                  throw new CompletionException(failure);
                }
                return evaluator.submitGreedyActionSlots(batch, ingress);
              })
          .thenCompose(future -> future);
    }

    @Override
    public synchronized void close() {
      if (!consumed) {
        consumed = true;
        ingress.abort();
      }
    }
  }

  /** 借用した資源はDuelSessionが閉じる。 */
  @Override
  public void close() {}
}

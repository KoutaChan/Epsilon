package com.epsilon.cfr.evaluation;

import com.epsilon.cfr.collection.CfrPolicy;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/** 共通の対戦評価に盤面入力と最大確率の合法手を渡す。借用した推論器の寿命は呼び出し側が管理する。 */
final class CfrDuelPolicy implements BatchedPolicy {
  private final CfrPolicy policy;
  private final int maximumBatch;

  CfrDuelPolicy(CfrPolicy policy, int maximumBatch) {
    this.policy = Objects.requireNonNull(policy);
    if (maximumBatch < 1) throw new IllegalArgumentException("Invalid duel inference batch");
    this.maximumBatch = maximumBatch;
  }

  @Override
  public Object batchKey(DecisionRequest request) {
    return DecisionBatchBuilder.selectInferenceBucket(request.legalActions());
  }

  @Override
  public int maxBatchSize(Object key) {
    return maximumBatch;
  }

  @Override
  public boolean needsInferenceWork() {
    return policy.needsInferenceWork();
  }

  @Override
  public Ingress tryAcquire(Object key, Runnable available) {
    var attempt = policy.tryAcquireInferenceIngress();
    if (!attempt.acquired()) {
      attempt.available().whenComplete((ignored, failure) -> available.run());
      return null;
    }
    return new Reservation(attempt.ingress());
  }

  private DecisionContext context(DecisionRequest request) {
    return new DecisionContext(
        request.episodeId(),
        request.id(),
        request.player(),
        request.observation(),
        request.legalActions());
  }

  private final class Reservation implements Ingress {
    private final DecisionInferenceIngress ingress;
    private boolean consumed;

    Reservation(DecisionInferenceIngress ingress) {
      this.ingress = ingress;
    }

    @Override
    public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
      synchronized (this) {
        if (consumed) throw new IllegalStateException("CFR duel ingress already used");
        consumed = true;
      }
      List<DecisionContext> contexts;
      try {
        if (requests.isEmpty() || requests.size() > maximumBatch)
          throw new IllegalArgumentException("CFR duel batch exceeds its admitted capacity");
        contexts = requests.stream().map(CfrDuelPolicy.this::context).toList();
      } catch (RuntimeException | Error failure) {
        ingress.abort();
        throw failure;
      }
      return policy
          .submitBatch(contexts, ingress)
          .thenApply(
              predictions -> {
                if (predictions.size() != contexts.size())
                  throw new IllegalStateException("CFR duel prediction count mismatch");
                int[] selected = new int[contexts.size()];
                for (int row = 0; row < contexts.size(); row++) {
                  double[] probabilities = predictions.get(row).targetPolicy();
                  if (probabilities.length != contexts.get(row).legalActions().size())
                    throw new IllegalStateException("CFR duel legal action count mismatch");
                  double best = -1;
                  for (int slot = 0; slot < probabilities.length; slot++) {
                    double value = probabilities[slot];
                    if (!Double.isFinite(value) || value < 0)
                      throw new IllegalStateException("Invalid CFR duel action probability");
                    if (value > best) {
                      best = value;
                      selected[row] = slot;
                    }
                  }
                  if (best <= 0)
                    throw new IllegalStateException("CFR duel policy has no positive probability");
                }
                return selected;
              });
    }

    @Override
    public synchronized void close() {
      if (!consumed) {
        consumed = true;
        ingress.abort();
      }
    }
  }

  @Override
  public void close() {
    // 推論器は対戦評価セッションが所有し、スケジューラーの処理完了後に解放する。
  }
}

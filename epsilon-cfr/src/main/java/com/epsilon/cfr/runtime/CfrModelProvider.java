package com.epsilon.cfr.runtime;

import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import com.epsilon.spi.ModelProvider;
import com.epsilon.spi.PolicyExecutionContext;
import com.epsilon.util.SeedMixer;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;

/** 配布された平均方策を、各モデル共通の実行コンテキストを通じてArenaへ接続する。 */
public final class CfrModelProvider implements ModelProvider {
  @Override
  public String seriesId() {
    return "epsilon-cfr";
  }

  @Override
  public BatchedPolicy open(
      Path checkpoint, Map<String, String> options, PolicyExecutionContext execution) {
    CfrDeployment deployment = CfrDeployment.open(checkpoint, options);
    try {
      return new Policy(deployment, execution);
    } catch (RuntimeException | Error failure) {
      try {
        deployment.close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  private static final class Policy implements BatchedPolicy {
    private final CfrDeployment deployment;
    private final CfrInferenceCoordinator coordinator;
    private boolean closed;

    Policy(CfrDeployment deployment, PolicyExecutionContext execution) {
      this.deployment = deployment;
      coordinator = new CfrInferenceCoordinator(List.of(deployment.model), execution);
    }

    @Override
    public Object batchKey(DecisionRequest request) {
      return DecisionBatchBuilder.selectInferenceBucket(request.legalActions());
    }

    @Override
    public int maxBatchSize(Object key) {
      return deployment.maxBatch;
    }

    @Override
    public boolean needsInferenceWork() {
      return coordinator.needsInferenceWork();
    }

    @Override
    public synchronized Ingress tryAcquire(Object key, Runnable available) {
      if (closed) throw new IllegalStateException("CFR policy is closed");
      var attempt = coordinator.tryAcquireInferenceIngress();
      if (!attempt.acquired()) {
        attempt.available().whenComplete((ignored, failure) -> available.run());
        return null;
      }
      return new Reservation(attempt.ingress());
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
          if (consumed) throw new IllegalStateException("CFR ingress already used");
          consumed = true;
        }
        List<DecisionRequest> rows;
        List<DecisionContext> contexts;
        try {
          rows = List.copyOf(requests);
          if (rows.isEmpty() || rows.size() > deployment.maxBatch)
            throw new IllegalArgumentException(
                "CFR batch is empty or exceeds its admitted capacity");
          contexts =
              rows.stream()
                  .map(
                      request ->
                          deployment.context(
                              request.episodeId(),
                              request.id(),
                              request.observation(),
                              request.legalActions()))
                  .toList();
        } catch (RuntimeException | Error failure) {
          ingress.abort();
          throw failure;
        }
        return coordinator
            .submit(contexts, ingress)
            .thenApply(
                predictions -> {
                  int[] selected = new int[rows.size()];
                  for (int row = 0; row < selected.length; row++) {
                    var request = rows.get(row);
                    long seed =
                        SeedMixer.mix64(
                            SeedMixer.mix64(deployment.seed ^ request.episodeId()) ^ request.id());
                    double draw = new SplittableRandom(seed).nextDouble(), sum = 0;
                    double[] probabilities = predictions.get(row).targetPolicy();
                    int slot = -1;
                    for (int action = 0; action < probabilities.length; action++) {
                      if (probabilities[action] > 0) slot = action;
                      sum += probabilities[action];
                      if (draw < sum) {
                        slot = action;
                        break;
                      }
                    }
                    if (slot < 0)
                      throw new IllegalStateException("CFR policy has no positive probability");
                    selected[row] = slot;
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
    public synchronized void close() {
      if (closed) return;
      closed = true;
      coordinator.close();
      deployment.close();
    }
  }
}

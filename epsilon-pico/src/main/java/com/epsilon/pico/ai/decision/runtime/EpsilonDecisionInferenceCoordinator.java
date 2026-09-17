package com.epsilon.pico.ai.decision.runtime;

import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.runtime.CpuInferenceDevice;
import com.epsilon.runtime.DecisionDevicePipeline;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.runtime.InferenceDispatcher;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/** 系列の最終ホスト側バッチと結果だけを接続し、容量とデバイス選択はCommonへ渡す。 */
final class EpsilonDecisionInferenceCoordinator implements EpsilonDecisionGreedyEvaluator {
  private final EpsilonDecisionInferenceServer[] servers;
  private final InferenceDispatcher.Registration<Batch, DecisionInferenceResult> registration;
  private boolean closed;
  private Throwable closeFailure;

  EpsilonDecisionInferenceCoordinator(
      EpsilonDecisionInferenceServer[] servers,
      int initialDevice,
      DecisionExecutionContext execution,
      int slots,
      int readyCapacity) {
    this.servers = servers;
    ArrayList<InferenceDispatcher.Backend<Batch, DecisionInferenceResult>> backends =
        new ArrayList<>(servers.length);
    for (int offset = 0; offset < servers.length; offset++) {
      EpsilonDecisionInferenceServer server =
          servers[Math.floorMod(initialDevice + offset, servers.length)];
      if (server.device().isGpu()) {
        server.enableAsyncRing(slots, readyCapacity, execution.streams(server.device()));
        backends.add(new GpuBackend(server));
      } else {
        backends.add(
            new CpuBackend(server, execution.cpuInference(server.device(), readyCapacity)));
      }
    }
    registration = execution.inferenceDispatcher().register(backends);
  }

  @Override
  public boolean needsInferenceWork() {
    return registration.needsWork();
  }

  @Override
  public DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
    var attempt = registration.tryReserve();
    if (!attempt.acquired()) return DecisionInferenceIngress.Attempt.blocked(attempt.available());
    var reservation = attempt.reservation();
    return DecisionInferenceIngress.Attempt.acquired(
        new DecisionInferenceIngress(this, reservation, reservation::close));
  }

  @Override
  public List<EpsilonDecisionInferenceServer.Prediction> evaluateBatch(DecisionHostBatch batch) {
    return submitBatch(batch).join();
  }

  @Override
  public CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> submitBatch(
      DecisionHostBatch batch) {
    requireFilled(batch);
    return predictions(batch.sliceRows(0, batch.size()));
  }

  @Override
  public CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> submitBatch(
      DecisionHostBatch batch, DecisionInferenceIngress ingress) {
    return submitAdmitted(batch, DecisionInferenceResult.Kind.PREDICTIONS, ingress)
        .thenApply(value -> ((DecisionInferenceResult.Predictions) value).values());
  }

  @Override
  public int[] evaluateGreedyActionSlots(DecisionHostBatch batch) {
    return submitGreedyActionSlots(batch).join();
  }

  @Override
  public CompletableFuture<int[]> submitGreedyActionSlots(DecisionHostBatch batch) {
    requireFilled(batch);
    int limit = deviceBatchSize(batch.bucket());
    if (batch.size() <= limit) {
      return submitRows(
              batch.sliceRows(0, batch.size()), DecisionInferenceResult.Kind.GREEDY_ACTION_SLOTS)
          .thenApply(value -> ((DecisionInferenceResult.GreedyActionSlots) value).values());
    }
    ArrayList<CompletableFuture<DecisionInferenceResult>> parts = new ArrayList<>();
    for (int start = 0; start < batch.size(); start += limit) {
      parts.add(
          submitRows(
              batch.sliceRows(start, Math.min(limit, batch.size() - start)),
              DecisionInferenceResult.Kind.GREEDY_ACTION_SLOTS));
    }
    return CompletableFuture.allOf(parts.toArray(CompletableFuture[]::new))
        .thenApply(
            ignored -> {
              int[] result = new int[batch.size()];
              int offset = 0;
              for (var part : parts) {
                int[] values = ((DecisionInferenceResult.GreedyActionSlots) part.join()).values();
                System.arraycopy(values, 0, result, offset, values.length);
                offset += values.length;
              }
              return result;
            });
  }

  @Override
  public CompletableFuture<int[]> submitGreedyActionSlots(
      DecisionHostBatch batch, DecisionInferenceIngress ingress) {
    return submitAdmitted(batch, DecisionInferenceResult.Kind.GREEDY_ACTION_SLOTS, ingress)
        .thenApply(value -> ((DecisionInferenceResult.GreedyActionSlots) value).values());
  }

  private CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> predictions(
      DecisionHostBatch.RowSlice rows) {
    int limit = deviceBatchSize(rows.bucket());
    if (rows.size() <= limit) {
      return submitRows(rows, DecisionInferenceResult.Kind.PREDICTIONS)
          .thenApply(value -> ((DecisionInferenceResult.Predictions) value).values());
    }
    ArrayList<CompletableFuture<DecisionInferenceResult>> parts = new ArrayList<>();
    for (int start = 0; start < rows.size(); start += limit) {
      parts.add(
          submitRows(
              rows.source()
                  .sliceRows(rows.fromInclusive() + start, Math.min(limit, rows.size() - start)),
              DecisionInferenceResult.Kind.PREDICTIONS));
    }
    return CompletableFuture.allOf(parts.toArray(CompletableFuture[]::new))
        .thenApply(
            ignored -> {
              ArrayList<EpsilonDecisionInferenceServer.Prediction> result =
                  new ArrayList<>(rows.size());
              for (var part : parts)
                result.addAll(((DecisionInferenceResult.Predictions) part.join()).values());
              return result;
            });
  }

  private CompletableFuture<DecisionInferenceResult> submitRows(
      DecisionHostBatch.RowSlice rows, DecisionInferenceResult.Kind kind) {
    return registration.submit(new Batch(DecisionHostBatch.RowBatch.of(rows), kind));
  }

  @SuppressWarnings("unchecked")
  private CompletableFuture<DecisionInferenceResult> submitAdmitted(
      DecisionHostBatch batch,
      DecisionInferenceResult.Kind kind,
      DecisionInferenceIngress ingress) {
    DecisionInferenceIngress.Ownership ownership = ingress.handoff(this);
    try {
      requireFilled(batch);
      if (batch.size() > deviceBatchSize(batch.bucket())) {
        throw new IllegalArgumentException("Admitted batch exceeds a device batch");
      }
    } catch (RuntimeException | Error failure) {
      ownership.release();
      throw failure;
    }
    var reservation =
        (InferenceDispatcher.Reservation<Batch, DecisionInferenceResult>) ownership.move();
    return reservation.submit(
        new Batch(DecisionHostBatch.RowBatch.of(batch.sliceRows(0, batch.size())), kind));
  }

  private static void requireFilled(DecisionHostBatch batch) {
    if (batch.size() != batch.capacity())
      throw new IllegalArgumentException("Inference input must be exactly filled");
  }

  private int deviceBatchSize(DecisionBucket bucket) {
    int minimum = Integer.MAX_VALUE;
    for (EpsilonDecisionInferenceServer server : servers)
      minimum = Math.min(minimum, server.preferredBatchSize(bucket));
    return minimum;
  }

  @Override
  public int preferredStreamingBatchSize(DecisionBucket bucket) {
    return deviceBatchSize(bucket);
  }

  @Override
  public int preferredStreamingBatchSize() {
    int minimum = Integer.MAX_VALUE;
    for (EpsilonDecisionInferenceServer server : servers)
      minimum = Math.min(minimum, server.preferredBatchSize());
    return minimum;
  }

  @Override
  public int preferredBatchSize() {
    return preferredStreamingBatchSize() * servers.length;
  }

  @Override
  public int preferredBatchSize(DecisionBucket bucket) {
    return deviceBatchSize(bucket) * servers.length;
  }

  EpsilonDecisionInferenceServer singleServerForDiagnostics() {
    if (servers.length != 1)
      throw new IllegalStateException("Diagnostics require exactly one device");
    registration.close();
    servers[0].awaitDeviceIdle();
    return servers[0];
  }

  /** producer停止後、同じサーバー群を固定入力再生へ借用する。配列の所有権は移さない。 */
  EpsilonDecisionInferenceServer[] serversForReplayBenchmark() {
    registration.awaitIdle();
    for (EpsilonDecisionInferenceServer server : servers) server.awaitDeviceIdle();
    return servers;
  }

  boolean resourcesReleased() {
    for (EpsilonDecisionInferenceServer server : servers)
      if (!server.resourcesReleased()) return false;
    return true;
  }

  @Override
  public synchronized void close() {
    if (closed) {
      rethrow(closeFailure);
      return;
    }
    closed = true;
    registration.close();
    Throwable failure = null;
    for (int index = servers.length - 1; index >= 0; index--) {
      try {
        servers[index].close();
      } catch (RuntimeException | Error problem) {
        if (failure == null) failure = problem;
        else failure.addSuppressed(problem);
      }
    }
    closeFailure = failure;
    rethrow(failure);
  }

  private static void rethrow(Throwable failure) {
    if (failure instanceof RuntimeException runtime) throw runtime;
    if (failure instanceof Error error) throw error;
  }

  private record Batch(DecisionHostBatch.RowBatch rows, DecisionInferenceResult.Kind kind) {
    long estimatedWork() {
      DecisionBucket bucket = rows.bucket();
      return (128L + 2L * bucket.legalActionCapacity() * bucket.actionTransitionCapacity())
          * rows.size();
    }
  }

  private record GpuBackend(EpsilonDecisionInferenceServer server)
      implements InferenceDispatcher.Backend<Batch, DecisionInferenceResult> {
    @Override
    public InferenceDispatcher.Device device() {
      return server.inferenceDevice();
    }

    @Override
    public CompletableFuture<DecisionInferenceResult> trySubmit(Batch batch, Runnable handoff) {
      DecisionDevicePipeline.HostReadyCell cell = server.acquireHostReadyCell();
      if (cell == null) return null;
      cell.onSlotHandoff(handoff);
      return server.submitToRing(batch.rows, batch.kind, cell);
    }
  }

  private record CpuBackend(EpsilonDecisionInferenceServer server, CpuInferenceDevice device)
      implements InferenceDispatcher.Backend<Batch, DecisionInferenceResult> {
    @Override
    public CompletableFuture<DecisionInferenceResult> trySubmit(Batch batch, Runnable handoff) {
      return device.trySubmit(
          batch.estimatedWork(),
          handoff,
          () -> server.evaluateSynchronously(batch.rows, batch.kind));
    }
  }
}

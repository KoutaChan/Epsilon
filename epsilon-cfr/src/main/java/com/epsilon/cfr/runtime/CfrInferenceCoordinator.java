package com.epsilon.cfr.runtime;

import ai.djl.pytorch.engine.PtNDArray;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.model.CfrInferenceWorkspace;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionInferenceStoragePool;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.runtime.CpuInferenceDevice;
import com.epsilon.runtime.DecisionDevicePipeline;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.runtime.InferenceDispatcher;
import com.epsilon.spi.PolicyExecutionContext;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/** CFRの入出力を、Nanoと共通の受付制御・デバイス割り当て・処理完了待ちに接続する。 */
final class CfrInferenceCoordinator implements AutoCloseable {
  private final CfrModel encoder;
  private final PolicyExecutionContext execution;
  private final List<DecisionDevicePipeline.Lease> leases = new ArrayList<>();
  private final List<GpuBackend> gpuBackends = new ArrayList<>();
  private final InferenceDispatcher.Registration<Batch, List<Prediction>> registration;
  private final DecisionInferenceStoragePool inputStorage;
  private final Set<CompletableFuture<Void>> returningInputs = ConcurrentHashMap.newKeySet();
  private boolean closed;

  CfrInferenceCoordinator(List<CfrModel> models, PolicyExecutionContext execution) {
    if (models.isEmpty()) throw new IllegalArgumentException("No CFR inference models");
    encoder = models.getFirst();
    this.execution = execution;
    List<InferenceDispatcher.Backend<Batch, List<Prediction>>> backends = new ArrayList<>();
    try {
      for (CfrModel model : models) {
        if (model.device().isGpu()) {
          var lease =
              DecisionDevicePipeline.acquire(
                  model.device(),
                  execution.slotsPerDevice(),
                  execution.readyBatchesPerDevice(),
                  execution.devices().streams(model.device()));
          leases.add(lease);
          var backend = new GpuBackend(model, lease, execution);
          gpuBackends.add(backend);
          backends.add(backend);
        } else {
          backends.add(
              new CpuBackend(
                  model,
                  execution
                      .devices()
                      .cpuInference(model.device(), execution.readyBatchesPerDevice()),
                  execution));
        }
      }
      inputStorage = inputStorage(backends, encoder.config().inferenceBatchSize());
      registration = execution.devices().inferenceDispatcher().register(backends);
    } catch (RuntimeException | Error failure) {
      for (var lease : leases) {
        try {
          lease.close();
        } catch (RuntimeException | Error closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      throw failure;
    }
  }

  private static DecisionInferenceStoragePool inputStorage(
      List<InferenceDispatcher.Backend<Batch, List<Prediction>>> backends, int batchSize) {
    int admitted =
        backends.stream()
            .map(InferenceDispatcher.Backend::device)
            .distinct()
            .mapToInt(device -> device.hostCapacity() + device.slots())
            .sum();
    var buckets = new HashSet<DecisionBucket>();
    for (int actions : DecisionInputSchema.LEGAL_ACTION_BUCKETS)
      for (int transitions : DecisionInputSchema.ACTION_TRANSITION_BUCKETS)
        buckets.add(DecisionBucket.forInferenceCounts(actions, transitions));
    long rowBytes =
        new DecisionInputLayout(1, new DecisionBucket(DecisionInputSchema.MAX_LEGAL_ACTIONS, 1))
            .inputByteCount();
    // 待機中の再利用配列だけの上限。実行中の入力数は、既存の受付予約とデバイス枠が制限する。
    long retainedBytes = Math.min(256L << 20, rowBytes * batchSize * admitted);
    return new DecisionInferenceStoragePool(
        retainedBytes, Math.multiplyExact(admitted, buckets.size()));
  }

  boolean needsInferenceWork() {
    return registration.needsWork();
  }

  synchronized DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
    if (closed) throw new IllegalStateException("CFR inference is closed");
    var attempt = registration.tryReserve();
    if (!attempt.acquired()) return DecisionInferenceIngress.Attempt.blocked(attempt.available());
    var reservation = attempt.reservation();
    return DecisionInferenceIngress.Attempt.acquired(
        new DecisionInferenceIngress(this, reservation, reservation::close));
  }

  CompletableFuture<List<Prediction>> submit(List<DecisionContext> contexts) {
    while (true) {
      var attempt = tryAcquireInferenceIngress();
      if (attempt.acquired()) return submit(contexts, attempt.ingress());
      attempt.available().join();
    }
  }

  @SuppressWarnings("unchecked")
  CompletableFuture<List<Prediction>> submit(
      List<DecisionContext> contexts, DecisionInferenceIngress ingress) {
    var ownership = ingress.handoff(this);
    var reservation = (InferenceDispatcher.Reservation<Batch, List<Prediction>>) ownership.move();
    DecisionInferenceStoragePool.Lease storage;
    try {
      storage = inputStorage.acquire();
    } catch (RuntimeException | Error failure) {
      reservation.close();
      return CompletableFuture.failedFuture(failure);
    }
    var returned = new CompletableFuture<Void>();
    returningInputs.add(returned);
    CompletableFuture<List<Prediction>> inference;
    try {
      inference =
          encode(contexts, storage)
              .handle(
                  (batch, failure) -> {
                    if (failure != null) {
                      reservation.close();
                      return CompletableFuture.<List<Prediction>>failedFuture(failure);
                    }
                    return reservation.submit(batch);
                  })
              .thenCompose(future -> future);
    } catch (RuntimeException | Error failure) {
      reservation.close();
      inference = CompletableFuture.failedFuture(failure);
    }
    var released =
        inference.whenComplete(
            (ignored, failure) -> {
              try {
                storage.close();
              } finally {
                returned.complete(null);
                returningInputs.remove(returned);
              }
            });
    // 呼び出し側のcancelは内部の回収処理へ伝えず、入力を使用中の処理が終わってから返す。
    return released.copy();
  }

  private CompletableFuture<Batch> encode(
      List<DecisionContext> contexts, DecisionInferenceStoragePool.Lease storage) {
    if (contexts.isEmpty()) throw new IllegalArgumentException("Empty CFR inference batch");
    long work = 0;
    for (DecisionContext context : contexts) work += 128L + 2L * context.legalActions().size();
    long estimatedWork = work;
    var encoded = new CompletableFuture<CfrModel.EncodedBatch>();
    // 容量区分の計画と入力配列の確保もCPUプールで行い、収集側の対局進行を止めない。
    return execution
        .encodeRowsAsync(
            1,
            (from, to) ->
                encoder
                    .encodeAsync(contexts, execution, storage)
                    .whenComplete(
                        (batch, failure) -> {
                          if (failure == null) encoded.complete(batch);
                          else encoded.completeExceptionally(failure);
                        }))
        .thenCompose(ignored -> encoded)
        .thenApply(batch -> new Batch(batch, contexts.size(), estimatedWork));
  }

  @Override
  public synchronized void close() {
    if (closed) return;
    closed = true;
    Throwable failure = null;
    try {
      registration.close();
    } catch (RuntimeException | Error problem) {
      failure = problem;
    }
    // 共通ディスパッチャーの結果通知は別スレッドなので、入力領域の返却まで待ってからプールを閉じる。
    CompletableFuture.allOf(returningInputs.toArray(CompletableFuture[]::new)).join();
    inputStorage.close();
    for (int i = leases.size() - 1; i >= 0; i--) {
      try {
        leases.get(i).close();
      } catch (RuntimeException | Error problem) {
        if (failure == null) failure = problem;
        else failure.addSuppressed(problem);
      }
    }
    for (GpuBackend backend : gpuBackends) {
      try {
        backend.close();
      } catch (RuntimeException | Error problem) {
        if (failure == null) failure = problem;
        else failure.addSuppressed(problem);
      }
    }
    if (failure instanceof RuntimeException problem) throw problem;
    if (failure instanceof Error problem) throw problem;
  }

  DecisionInferenceStoragePool.Snapshot inputStorageSnapshot() {
    return inputStorage.snapshot();
  }

  private record Batch(CfrModel.EncodedBatch encoded, int rows, long work) {}

  private record CpuBackend(
      CfrModel model, CpuInferenceDevice device, PolicyExecutionContext execution)
      implements InferenceDispatcher.Backend<Batch, List<Prediction>> {
    @Override
    public CompletableFuture<List<Prediction>> trySubmit(Batch batch, Runnable handoff) {
      var completion =
          new CfrPredictionCompletion(
              () -> batch.encoded().encodePayloadsAsync(execution), handoff);
      var computed =
          device.trySubmit(
              batch.work(), completion::handoff, () -> model.evaluateEncoded(batch.encoded()));
      if (computed == null) return null;
      computed.whenComplete(
          (policies, failure) -> {
            if (failure == null) completion.complete(policies);
            else completion.fail(failure);
          });
      return completion.result();
    }
  }

  private static final class GpuBackend
      implements InferenceDispatcher.Backend<Batch, List<Prediction>>, AutoCloseable {
    private final CfrModel model;
    private final DecisionDevicePipeline.Lease lease;
    private final PolicyExecutionContext execution;
    private final IdentityHashMap<DecisionDevicePipeline.Slot, CfrInferenceWorkspace> workspaces =
        new IdentityHashMap<>();

    GpuBackend(
        CfrModel model, DecisionDevicePipeline.Lease lease, PolicyExecutionContext execution) {
      this.model = model;
      this.lease = lease;
      this.execution = execution;
    }

    CfrInferenceWorkspace workspace(DecisionDevicePipeline.Slot slot) {
      return workspaces.computeIfAbsent(slot, ignored -> model.newInferenceWorkspace());
    }

    @Override
    public InferenceDispatcher.Device device() {
      return lease.inferenceDevice();
    }

    @Override
    public CompletableFuture<List<Prediction>> trySubmit(Batch batch, Runnable handoff) {
      var cell = lease.acquireHostReadyCell();
      if (cell == null) return null;
      var completion =
          new CfrPredictionCompletion(
              () -> batch.encoded().encodePayloadsAsync(execution), handoff);
      try {
        cell.onSlotHandoff(completion::handoff);
        var work = new InferenceWork(this, batch, completion);
        lease.claim(cell);
        lease.submitClaimed(work, cell);
      } catch (RuntimeException | Error failure) {
        cell.abortIfActive();
        completion.fail(failure);
      }
      return completion.result();
    }

    @Override
    public void close() {
      Throwable failure = null;
      for (var workspace : workspaces.values()) {
        try {
          workspace.close();
        } catch (RuntimeException | Error problem) {
          if (failure == null) failure = problem;
          else failure.addSuppressed(problem);
        }
      }
      if (failure instanceof RuntimeException problem) throw problem;
      if (failure instanceof Error problem) throw problem;
    }
  }

  /** デバイスへの入力転送、演算、出力転送の所有権と完了通知を共通パイプラインへ渡す。 */
  private static final class InferenceWork implements DecisionDevicePipeline.Work {
    private final GpuBackend backend;
    private final Batch batch;
    private final CfrPredictionCompletion completion;
    private CfrModel.Policies policies;
    private CfrInferenceWorkspace.Batch staged;
    private boolean h2dStarted, h2dRecorded, computeStarted, computeRecorded;
    private Throwable failure;

    InferenceWork(GpuBackend backend, Batch batch, CfrPredictionCompletion completion) {
      this.backend = backend;
      this.batch = batch;
      this.completion = completion;
    }

    @Override
    public long estimatedWorkUnits() {
      return batch.work();
    }

    @Override
    public DecisionDevicePipeline.BatchMetadata batchMetadata() {
      return new DecisionDevicePipeline.BatchMetadata(
          "epsilon-cfr", "board", "predictions", batch.rows(), batch.rows());
    }

    @Override
    public void stage(DecisionDevicePipeline.Slot slot) {
      staged = backend.workspace(slot).stage(batch.encoded());
    }

    @Override
    public void submit(DecisionDevicePipeline.Slot slot) {
      try {
        h2dStarted = true;
        try (var ignored = slot.h2dStream().openScope()) {
          staged.upload();
          slot.h2dReady().record();
          h2dRecorded = true;
        }
        computeStarted = true;
        try (var ignored = slot.computeStream().openScope()) {
          slot.h2dReady().waitOnStream();
          var output = staged.forward();
          slot.computeReady().record();
          computeRecorded = true;
          if (output != null) slot.prepareOutput((PtNDArray) output);
        }
      } catch (RuntimeException | Error problem) {
        recordLatestCompletion(slot, problem);
        throw problem;
      }
    }

    private void recordLatestCompletion(DecisionDevicePipeline.Slot slot, Throwable problem) {
      if (h2dStarted && !h2dRecorded) {
        try (var ignored = slot.h2dStream().openScope()) {
          slot.h2dReady().record();
          h2dRecorded = true;
        } catch (Throwable recordingFailure) {
          problem.addSuppressed(recordingFailure);
        }
      }
      if (h2dRecorded && !computeRecorded) {
        try (var ignored = slot.computeStream().openScope()) {
          slot.h2dReady().waitOnStream();
          slot.computeReady().record();
          computeRecorded = true;
        } catch (Throwable recordingFailure) {
          problem.addSuppressed(recordingFailure);
        }
      }
    }

    @Override
    public boolean isComplete(DecisionDevicePipeline.Slot slot) {
      return slot.computeReady().isComplete();
    }

    @Override
    public void complete(DecisionDevicePipeline.Slot slot) {
      policies =
          staged.complete(
              staged.outputElements() == 0 ? FloatBuffer.allocate(0) : slot.outputFloats());
    }

    @Override
    public void release(DecisionDevicePipeline.Slot slot) {
      if (staged != null) {
        staged.close();
        staged = null;
      }
    }

    @Override
    public boolean abort(DecisionDevicePipeline.Slot slot, Throwable problem) {
      fail(problem);
      if ((h2dStarted && !h2dRecorded) || (computeStarted && !computeRecorded)) {
        problem.addSuppressed(
            new IllegalStateException("CFR stream completion could not be recorded"));
        return false;
      }
      try {
        if (computeRecorded) slot.computeReady().synchronize();
        else if (h2dRecorded) slot.h2dReady().synchronize();
        release(slot);
        return true;
      } catch (Throwable synchronizationFailure) {
        problem.addSuppressed(synchronizationFailure);
        return false;
      }
    }

    @Override
    public void fail(Throwable problem) {
      if (failure == null) failure = problem;
      else if (failure != problem) failure.addSuppressed(problem);
    }

    @Override
    public void publish() {
      if (failure == null) completion.complete(policies);
      else completion.fail(failure);
    }
  }
}

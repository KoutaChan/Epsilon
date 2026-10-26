package com.epsilon.prism.policy;

import ai.djl.pytorch.engine.PtNDArray;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionInferenceStoragePool;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.prism.model.EncodedRows;
import com.epsilon.prism.model.InferenceWorkspace;
import com.epsilon.prism.model.PrismModel;
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

/**
 * 1つの版の方策。網の凍結コピーと版の番号（policy.json の {@link PolicyParams}）からなり、σ は貪欲（s の argmax、同点は p0 の大きい手）。s は融合
 * 推論の FP32 の出力をそのまま {@link PolicyRequest#scores} に書く。retain と s の最大が同点の問い合わせに p0（{@link Prior}）を、retain の問い合わせに
 * payload を書く。
 *
 * <p>GPU では Nano と共通の受付・予約・デバイスパイプラインに載せる。複数の GPU には同じ版の複製を置き、共通の振り分けがバッチを空いた GPU へ
 * 送る。CPU のモデルでは同じ符号化と仕上げを呼び出しスレッドで行う（試験用）。payload は retain の行にだけ作り、推論とは合流させない。
 */
public final class NetPolicyService implements PolicyService {
  private final PolicyParams params;
  private final PriorSettings prior;
  private final List<PrismModel> devices;
  private final PolicyExecutionContext execution;
  private final int maxBatch;
  private final int maxMultiTransitionBatch;
  private final RequestEncoder encoder = new RequestEncoder();
  private final List<DecisionDevicePipeline.Lease> leases = new ArrayList<>();
  private final List<GpuBackend> backends = new ArrayList<>();
  private final InferenceDispatcher.Registration<Batch, Void> registration;
  private final DecisionInferenceStoragePool storage;
  private final Set<CompletableFuture<Void>> returning = ConcurrentHashMap.newKeySet();
  private boolean closed;

  /**
   * @param params 版の番号
   * @param prior p0 の係数（retain の行に保存し、s の最大が同点なら貪欲の同点を割る）
   * @param devices デバイスごとの凍結コピー。所有権を受け取る。GPU を並べるか、CPU の1つだけにする
   */
  public NetPolicyService(
      PolicyParams params,
      PriorSettings prior,
      List<PrismModel> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      int maxMultiTransitionBatch) {
    this.params = params;
    this.prior = prior;
    this.devices = devices;
    this.execution = execution;
    this.maxBatch = maxBatch;
    this.maxMultiTransitionBatch = maxMultiTransitionBatch;
    if (!devices.getFirst().device().isGpu()) {
      storage = null;
      registration = null;
      return;
    }
    try {
      for (PrismModel net : devices) {
        var device = net.device();
        var lease =
            DecisionDevicePipeline.acquire(
                device,
                execution.slotsPerDevice(),
                execution.readyBatchesPerDevice(),
                execution.devices().streams(device));
        leases.add(lease);
        backends.add(new GpuBackend(lease, net));
      }
      storage = storage(backends, devices.getFirst().config().inferenceBatchSize());
      registration = execution.devices().inferenceDispatcher().register(new ArrayList<>(backends));
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

  private static DecisionInferenceStoragePool storage(List<GpuBackend> backends, int batchSize) {
    int admitted = 0;
    for (GpuBackend backend : backends)
      admitted += backend.device().hostCapacity() + backend.device().slots();
    var buckets = new HashSet<DecisionBucket>();
    for (int actions : DecisionInputSchema.LEGAL_ACTION_BUCKETS)
      for (int transitions : DecisionInputSchema.ACTION_TRANSITION_BUCKETS)
        buckets.add(DecisionBucket.forInferenceCounts(actions, transitions));
    long rowBytes =
        new DecisionInputLayout(1, new DecisionBucket(DecisionInputSchema.MAX_LEGAL_ACTIONS, 1))
            .inputByteCount();
    long retainedBytes = Math.min(256L << 20, rowBytes * batchSize * admitted);
    return new DecisionInferenceStoragePool(
        retainedBytes, Math.multiplyExact(admitted, buckets.size()));
  }

  @Override
  public int version() {
    return params.version();
  }

  @Override
  public boolean inline() {
    return false;
  }

  @Override
  public int maxBatch(Capacity capacity) {
    return capacity == Capacity.MULTIPLE_TRANSITIONS
        ? Math.min(maxBatch, maxMultiTransitionBatch)
        : maxBatch;
  }

  @Override
  public boolean needsWork() {
    return registration == null || registration.needsWork();
  }

  @Override
  public synchronized DecisionInferenceIngress.Attempt tryAcquireIngress() {
    if (registration == null)
      return DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(this));
    var attempt = registration.tryReserve();
    if (!attempt.acquired()) return DecisionInferenceIngress.Attempt.blocked(attempt.available());
    var reservation = attempt.reservation();
    return DecisionInferenceIngress.Attempt.acquired(
        new DecisionInferenceIngress(this, reservation, reservation::close));
  }

  @Override
  @SuppressWarnings("unchecked")
  public CompletableFuture<Void> submit(
      List<PolicyRequest> rows, DecisionInferenceIngress ingress) {
    var ownership = ingress.handoff(this);
    if (registration == null) {
      ownership.move();
      return submitDirect(rows);
    }
    var reservation = (InferenceDispatcher.Reservation<Batch, Void>) ownership.move();
    DecisionInferenceStoragePool.Lease lease;
    try {
      lease = storage.acquire();
    } catch (RuntimeException | Error failure) {
      reservation.close();
      return CompletableFuture.failedFuture(failure);
    }
    var returned = new CompletableFuture<Void>();
    returning.add(returned);
    long work = 0;
    for (PolicyRequest request : rows) work += 128L + 2L * request.legal().size();
    long estimatedWork = work;
    CompletableFuture<Void> inference;
    try {
      inference =
          encoder
              .encode(rows, execution, lease)
              .handle(
                  (encoded, failure) -> {
                    if (failure != null) {
                      reservation.close();
                      return CompletableFuture.<Void>failedFuture(failure);
                    }
                    return reservation.submit(new Batch(rows, encoded, estimatedWork));
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
                lease.close();
              } finally {
                returned.complete(null);
                returning.remove(returned);
              }
            });
    return released;
  }

  /** CPU のモデル。符号化から σ までを呼び出しスレッドで行う。 */
  private CompletableFuture<Void> submitDirect(List<PolicyRequest> rows) {
    try {
      EncodedRows encoded = encoder.encode(rows, null, null).join();
      RequestEncoder.finish(rows, encoded, devices.getFirst().scores(encoded), prior);
      return CompletableFuture.completedFuture(null);
    } catch (RuntimeException | Error failure) {
      return CompletableFuture.failedFuture(failure);
    }
  }

  @Override
  public synchronized void close() {
    if (closed) return;
    closed = true;
    Throwable failure = null;
    try {
      if (registration != null) registration.close();
    } catch (RuntimeException | Error problem) {
      failure = problem;
    }
    CompletableFuture.allOf(returning.toArray(CompletableFuture[]::new)).join();
    List<AutoCloseable> resources = new ArrayList<>();
    resources.add(storage);
    resources.addAll(leases);
    resources.addAll(backends);
    resources.addAll(devices);
    for (AutoCloseable resource : resources) {
      try {
        if (resource != null) resource.close();
      } catch (Exception problem) {
        if (failure == null) failure = problem;
        else failure.addSuppressed(problem);
      }
    }
    if (failure instanceof RuntimeException problem) throw problem;
    if (failure instanceof Error problem) throw problem;
    if (failure != null) throw new IllegalStateException("Cannot close policy service", failure);
  }

  private record Batch(List<PolicyRequest> rows, EncodedRows encoded, long work) {}

  private final class GpuBackend
      implements InferenceDispatcher.Backend<Batch, Void>, AutoCloseable {
    private final DecisionDevicePipeline.Lease lease;
    private final PrismModel net;
    private final IdentityHashMap<DecisionDevicePipeline.Slot, InferenceWorkspace> workspaces =
        new IdentityHashMap<>();

    GpuBackend(DecisionDevicePipeline.Lease lease, PrismModel net) {
      this.lease = lease;
      this.net = net;
    }

    InferenceWorkspace workspace(DecisionDevicePipeline.Slot slot) {
      return workspaces.computeIfAbsent(slot, ignored -> new InferenceWorkspace(List.of(net)));
    }

    @Override
    public InferenceDispatcher.Device device() {
      return lease.inferenceDevice();
    }

    @Override
    public CompletableFuture<Void> trySubmit(Batch batch, Runnable handoff) {
      var cell = lease.acquireHostReadyCell();
      if (cell == null) return null;
      var result = new CompletableFuture<Void>();
      try {
        cell.onSlotHandoff(handoff);
        var work = new InferenceWork(this, batch, result);
        lease.claim(cell);
        lease.submitClaimed(work, cell);
      } catch (RuntimeException | Error failure) {
        cell.abortIfActive();
        result.completeExceptionally(failure);
      }
      return result;
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
  private final class InferenceWork implements DecisionDevicePipeline.Work {
    private final GpuBackend backend;
    private final Batch batch;
    private final CompletableFuture<Void> result;
    private InferenceWorkspace.Batch staged;
    private boolean h2dStarted, h2dRecorded, computeStarted, computeRecorded;
    private Throwable failure;

    InferenceWork(GpuBackend backend, Batch batch, CompletableFuture<Void> result) {
      this.backend = backend;
      this.batch = batch;
      this.result = result;
    }

    @Override
    public long estimatedWorkUnits() {
      return batch.work();
    }

    @Override
    public DecisionDevicePipeline.BatchMetadata batchMetadata() {
      int rows = batch.rows().size();
      return new DecisionDevicePipeline.BatchMetadata("epsilon-prism", "board", "scores", rows, rows);
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
      var scores =
          staged.complete(
              staged.outputElements() == 0 ? FloatBuffer.allocate(0) : slot.outputFloats());
      RequestEncoder.finish(batch.rows(), batch.encoded(), scores.nets()[0], prior);
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
            new IllegalStateException("Epsilon-Prism stream completion could not be recorded"));
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
      if (failure == null) result.complete(null);
      else result.completeExceptionally(failure);
    }
  }
}

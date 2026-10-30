package com.epsilon.prism.policy;

import ai.djl.Device;
import com.epsilon.network.NetworkRuntime;
import com.epsilon.prism.Limits;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.Tokenizer;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.runtime.DecisionDevicePipeline;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.runtime.InferenceDispatcher;
import com.epsilon.spi.PolicyExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 1 つの版の方策。問い合わせが運ぶ記録 O（出す側が捕捉する。{@link #readsRecord}）から
 * {@link Tokenizer} で入力の塊を作り（{@link BatchEncoder}）、網の s を推論の道（{@link InferencePath}: eager か
 * HIP、{@link DeviceInference}）で出す。σ は s の貪欲（同点は p0）、p0 は retain と s の最大が同点の問い合わせにだけ書く（{@link GreedyAnswer}。
 * p0 の特徴は Tokenizer が作る）。受付の区分は 1 つで、鳴きの行も同じ受け付けに入る（{@link #admissionKey}）。
 *
 * <p>GPU では Nano と共通の受付・予約・デバイスパイプラインに載せ、複数の GPU には同じ版の写しを置く。符号化は符号化のプールで進み、順伝播は
 * 実行枠ごとの計算のストリームで出すので、次の受け付けの符号化と 2 つの実行枠の順伝播が重なる。CPU の網では同じ道を呼び出しスレッドで通す。
 *
 * <p>行は収集が O から作る。入力は色の正準形（萬子・筒子・索子の入れ替えで変わらない）。仕事の量（{@link Throughput}）は閉じるときに記録に書く。
 */
public final class NetworkPolicy implements PolicyService {

  private static final Logger LOG = LoggerFactory.getLogger(NetworkPolicy.class);

  /** 計測の始まりがまだないこと。 */
  private static final long UNSET = Long.MIN_VALUE;

  /**
   * 方策の仕事の量。
   *
   * @param decisions 答えた問い合わせの数
   * @param batches 受け付けた塊の数
   * @param chunks 順伝播の数（1 つの受け付けは詰めたトークンの上限と塊の容量で分ける）
   * @param seconds 最初の受け付けから最後の答えまでの秒
   * @param encodeSeconds 付け・計画・符号化にかけたスレッドの秒の和
   */
  public record Throughput(
      long decisions, long batches, long chunks, double seconds, double encodeSeconds) {

    /** 1 秒あたりの答えた問い合わせの数。 */
    public double decisionsPerSecond() {
      return seconds > 0 ? decisions / seconds : 0;
    }
  }

  /** デバイスごとに推論の道を開く口。 */
  @FunctionalInterface
  private interface DeviceOpener {
    DeviceInference open(Device device) throws IOException;
  }

  private final PolicyParams params;
  private final PriorSettings prior;
  private final InferencePath inference;
  private final List<DeviceInference> devices;
  private final PolicyExecutionContext execution;
  private final int maxBatch;
  private final BatchEncoder encoder;
  private final List<DecisionDevicePipeline.Lease> leases = new ArrayList<>();
  private final InferenceDispatcher.Registration<Batch, Void> registration;
  private final Set<CompletableFuture<Void>> returning = ConcurrentHashMap.newKeySet();
  private final LongAdder decisions = new LongAdder();
  private final LongAdder batches = new LongAdder();
  private final LongAdder chunks = new LongAdder();
  private final AtomicLong first = new AtomicLong(UNSET);
  private final AtomicLong last = new AtomicLong(UNSET);
  private boolean closed;

  private NetworkPolicy(
      PolicyParams params,
      PriorSettings prior,
      InferencePath inference,
      List<DeviceInference> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      BatchEncoder encoder) {
    this.params = params;
    this.prior = prior;
    this.inference = inference;
    this.devices = devices;
    this.execution = execution;
    this.maxBatch = maxBatch;
    this.encoder = encoder;
    if (!devices.getFirst().device().isGpu()) {
      registration = null;
      return;
    }
    try {
      List<GpuBackend> backends = new ArrayList<>(devices.size());
      for (DeviceInference device : devices) {
        var lease =
            DecisionDevicePipeline.acquire(
                device.device(),
                execution.slotsPerDevice(),
                execution.readyBatchesPerDevice(),
                execution.devices().streams(device.device()));
        leases.add(lease);
        backends.add(new GpuBackend(lease, device));
      }
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

  /**
   * 保存した網を eager で開く（{@link #open(PolicyParams, Path, List, PolicyExecutionContext, int, PriorSettings, InferencePath)} に
   * {@link InferencePath#EAGER} を渡すのと同じ）。
   *
   * @param params 版の番号
   * @param model 網のディレクトリ（weights.bin と model.json。architecture は epsilon-prism-v2）
   * @param devices GPU の列か、CPU 1 つ
   * @param execution 符号化のプールと GPU の実行の資源
   * @param maxBatch 1 回の受け付けの行の上限
   * @param prior p0 の係数
   * @return 方策（網の所有権を持つ）
   * @throws IOException 網が読めないとき
   */
  public static NetworkPolicy open(
      PolicyParams params,
      Path model,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      PriorSettings prior)
      throws IOException {
    return open(params, model, devices, execution, maxBatch, prior, InferencePath.EAGER);
  }

  /**
   * 保存した網を開く。推論の道は網の形とデバイスで決める（{@link InferencePath#resolve}。HIP を選んで HIP の持たない形なら eager にして記録に
   * 書く）。HIP では先に libprism_hip.so を読み、読めなければ英語の例外で止まる。
   *
   * @param params 版の番号（HIP の重みの束の版の番号にもなる）
   * @param model 網のディレクトリ（weights.bin と model.json。architecture は epsilon-prism-v2）
   * @param devices GPU の列か、CPU 1 つ
   * @param execution 符号化のプールと GPU の実行の資源
   * @param maxBatch 1 回の受け付けの行の上限
   * @param prior p0 の係数
   * @param inference 設定の推論の道
   * @return 方策（網の所有権を持つ）
   * @throws IOException 網が読めないとき
   * @throws IllegalStateException HIP の道で libprism_hip.so が読めないか、GPU と合わないとき
   */
  public static NetworkPolicy open(
      PolicyParams params,
      Path model,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      PriorSettings prior,
      InferencePath inference)
      throws IOException {
    Objects.requireNonNull(inference, "inference");
    // エンジンを初めて取る前に、全モデル共通の実行既定値（PyTorch の CPU スレッド数）を決める
    NetworkRuntime.initialize();
    PrismShape shape = PrismNetwork.readShape(model);
    InferencePath path = inference.resolve(devices.getFirst(), shape);
    if (path != inference) {
      LOG.info(
          "PRISM policy v{}: {} inference does not run {} on {}; using {}",
          params.version(),
          inference,
          shape,
          devices.getFirst(),
          path);
    }
    return switch (path) {
      case EAGER ->
          eager(params, model, devices, execution, maxBatch, prior, BatchEncoder.TOKEN_BUDGET);
      case HIP ->
          hip(
              params,
              model,
              devices,
              execution,
              maxBatch,
              prior,
              BatchEncoder.UNBOUNDED,
              HipInference.gpu(BatchPlan.Capacity.DEFAULT));
    };
  }

  /**
   * eager の道で開く（試験は詰めたトークンの上限を小さくして塊を分ける）。
   *
   * @param tokenBudget 1 回の順伝播の詰めたトークンの上限
   */
  static NetworkPolicy eager(
      PolicyParams params,
      Path model,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      PriorSettings prior,
      int tokenBudget)
      throws IOException {
    boolean frontier = PrismNetwork.readShape(model).frontier();
    return assemble(
        params,
        prior,
        InferencePath.EAGER,
        devices,
        execution,
        maxBatch,
        new BatchEncoder(tokenBudget, frontier, BatchEncoder.Memory.DIRECT),
        device -> EagerInference.load(model, device));
  }

  /**
   * HIP の道で開く。重みの束は CPU に読んだ FP32 の網から 1 回だけ作り、全てのデバイスの道具に写す。道具は GPU なら実行枠の数、呼び出しスレッド
   * の口（試験の CPU の参照）なら 1 つ。
   *
   * @param tokenBudget 1 回の順伝播の詰めたトークンの上限（本番は {@link BatchEncoder#UNBOUNDED}）
   * @param lanes 実行枠の道具を作る口（本番は {@link HipInference#gpu}）
   */
  static NetworkPolicy hip(
      PolicyParams params,
      Path model,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      PriorSettings prior,
      int tokenBudget,
      HipInference.Lanes lanes)
      throws IOException {
    WeightPack pack = WeightPack.load(model, params.version());
    int scores = Math.multiplyExact(maxBatch, Limits.MAX_LEGAL_ACTIONS);
    return assemble(
        params,
        prior,
        InferencePath.HIP,
        devices,
        execution,
        maxBatch,
        new BatchEncoder(tokenBudget, false, BatchEncoder.Memory.PINNED),
        device ->
            HipInference.open(
                device, pack, device.isGpu() ? execution.slotsPerDevice() : 1, scores, lanes));
  }

  /** デバイスごとに推論の道を開いて方策を組む（失敗したら、開いた道と符号化の池を閉じる）。 */
  private static NetworkPolicy assemble(
      PolicyParams params,
      PriorSettings prior,
      InferencePath inference,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      BatchEncoder encoder,
      DeviceOpener opener)
      throws IOException {
    List<DeviceInference> opened = new ArrayList<>(devices.size());
    try {
      for (Device device : devices) {
        opened.add(opener.open(device));
      }
      return new NetworkPolicy(params, prior, inference, opened, execution, maxBatch, encoder);
    } catch (IOException | RuntimeException | Error failure) {
      for (DeviceInference device : opened) {
        try {
          device.close();
        } catch (RuntimeException | Error problem) {
          failure.addSuppressed(problem);
        }
      }
      try {
        encoder.close();
      } catch (RuntimeException | Error problem) {
        failure.addSuppressed(problem);
      }
      throw failure;
    }
  }

  @Override
  public int version() {
    return params.version();
  }

  /** 使っている推論の道（設定の道を網の形とデバイスで決めたもの）。 */
  public InferencePath inference() {
    return inference;
  }

  @Override
  public boolean inline() {
    return false;
  }

  /** 入力は問い合わせの記録 O だけ（符号化のスレッドは局面に触れない）。 */
  @Override
  public boolean readsRecord() {
    return true;
  }

  /** 行の上限は受け付けの区分に依らない（遷移の多い行は詰めたトークンの上限と塊の容量で順伝播を分ける）。 */
  @Override
  public int maxBatch(Capacity capacity) {
    return maxBatch;
  }

  /**
   * 全ての行を 1 つの区分で受け付ける。塊は区分に依らず {@link BatchPlan.Capacity#DEFAULT} で取り、行の出来事・遷移・手の実際の数で分けるので、
   * 鳴きの行を混ぜてよい（鳴きの行だけの小さな受け付けと、その待ちを作らない）。
   */
  @Override
  public Capacity admissionKey(PolicyRequest request) {
    return Capacity.SINGLE_TRANSITION;
  }

  @Override
  public boolean needsWork() {
    return registration == null || registration.needsWork();
  }

  @Override
  public synchronized DecisionInferenceIngress.Attempt tryAcquireIngress() {
    if (registration == null) {
      return DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(this));
    }
    var attempt = registration.tryReserve();
    if (!attempt.acquired()) {
      return DecisionInferenceIngress.Attempt.blocked(attempt.available());
    }
    var reservation = attempt.reservation();
    return DecisionInferenceIngress.Attempt.acquired(
        new DecisionInferenceIngress(this, reservation, reservation::close));
  }

  @Override
  @SuppressWarnings("unchecked")
  public CompletableFuture<Void> submit(List<PolicyRequest> rows, DecisionInferenceIngress ingress) {
    var ownership = ingress.handoff(this);
    first.compareAndSet(UNSET, System.nanoTime());
    if (registration == null) {
      ownership.move();
      return submitDirect(rows);
    }
    var reservation = (InferenceDispatcher.Reservation<Batch, Void>) ownership.move();
    Encoding encoding;
    try {
      encoding = encoder.acquire(rows);
    } catch (RuntimeException | Error failure) {
      reservation.close();
      return CompletableFuture.failedFuture(failure);
    }
    var returned = new CompletableFuture<Void>();
    returning.add(returned);
    CompletableFuture<Void> scored;
    try {
      scored =
          encoder
              .encode(encoding, execution)
              .handle(
                  (ignored, failure) -> {
                    if (failure != null) {
                      reservation.close();
                      return CompletableFuture.<Void>failedFuture(failure);
                    }
                    return reservation.submit(new Batch(encoding));
                  })
              .thenCompose(future -> future);
    } catch (RuntimeException | Error failure) {
      reservation.close();
      scored = CompletableFuture.failedFuture(failure);
    }
    // 結果を公開するのは実行枠とデバイスの資源を返した後なので、ここで作業域と塊を池に返してよい
    return scored.whenComplete(
        (ignored, failure) -> {
          try {
            encoder.release(encoding);
          } finally {
            returned.complete(null);
            returning.remove(returned);
          }
        });
  }

  /** CPU の網。符号化から σ までを呼び出しスレッドで行う。 */
  private CompletableFuture<Void> submitDirect(List<PolicyRequest> rows) {
    Encoding encoding = encoder.acquire(rows);
    try {
      encoder.encode(encoding, null).join();
      encoding.answer(devices.getFirst().score(encoding), prior);
      answered(encoding);
      return CompletableFuture.completedFuture(null);
    } catch (RuntimeException | Error failure) {
      return CompletableFuture.failedFuture(failure);
    } finally {
      encoder.release(encoding);
    }
  }

  /** 答えを書き終えた受け付けを数える。 */
  private void answered(Encoding encoding) {
    decisions.add(encoding.rows().size());
    batches.increment();
    chunks.add(encoding.chunks().size());
    last.accumulateAndGet(System.nanoTime(), Math::max);
  }

  /**
   * ここまでの仕事の量。
   *
   * @return 仕事の量
   */
  public Throughput throughput() {
    long start = first.get();
    long end = last.get();
    double seconds = start == UNSET || end == UNSET ? 0 : Math.max(0, end - start) / 1e9;
    return new Throughput(
        decisions.sum(), batches.sum(), chunks.sum(), seconds, encoder.seconds());
  }

  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    closed = true;
    Throwable failure = null;
    try {
      if (registration != null) {
        registration.close();
      }
    } catch (RuntimeException | Error problem) {
      failure = problem;
    }
    CompletableFuture.allOf(returning.toArray(CompletableFuture[]::new)).join();
    Throughput done = throughput();
    if (done.decisions() > 0) {
      LOG.info(
          "PRISM policy v{} ({}): decisions={} batches={} chunks={} rows/batch={} seconds={}"
              + " decisions/s={} encodeSeconds={}",
          params.version(),
          inference,
          done.decisions(),
          done.batches(),
          done.chunks(),
          String.format("%.1f", (double) done.decisions() / done.batches()),
          String.format("%.1f", done.seconds()),
          String.format("%.0f", done.decisionsPerSecond()),
          String.format("%.1f", done.encodeSeconds()));
    }
    // 実行枠を全て返し（リース）、デバイスの道（計画はストリームを待ってから返す・ストリーム・網）を閉じてから、入力の blob を返す
    List<AutoCloseable> resources = new ArrayList<>(leases);
    resources.addAll(devices);
    resources.add(encoder);
    for (AutoCloseable resource : resources) {
      try {
        resource.close();
      } catch (Exception problem) {
        if (failure == null) {
          failure = problem;
        } else {
          failure.addSuppressed(problem);
        }
      }
    }
    if (failure instanceof RuntimeException problem) {
      throw problem;
    }
    if (failure instanceof Error problem) {
      throw problem;
    }
    if (failure != null) {
      throw new IllegalStateException("Cannot close the PRISM policy", failure);
    }
  }

  /** 符号化を終えた 1 回の受け付け。 */
  private record Batch(Encoding encoding) {}

  /** 1 つの GPU の受け口。 */
  private final class GpuBackend implements InferenceDispatcher.Backend<Batch, Void> {
    private final DecisionDevicePipeline.Lease lease;
    private final DeviceInference device;

    GpuBackend(DecisionDevicePipeline.Lease lease, DeviceInference device) {
      this.lease = lease;
      this.device = device;
    }

    @Override
    public InferenceDispatcher.Device device() {
      return lease.inferenceDevice();
    }

    @Override
    public CompletableFuture<Void> trySubmit(Batch batch, Runnable handoff) {
      var cell = lease.acquireHostReadyCell();
      if (cell == null) {
        return null;
      }
      var result = new CompletableFuture<Void>();
      try {
        cell.onSlotHandoff(handoff);
        var work = new InferenceWork(device, batch, result);
        lease.claim(cell);
        lease.submitClaimed(work, cell);
      } catch (RuntimeException | Error failure) {
        cell.abortIfActive();
        result.completeExceptionally(failure);
      }
      return result;
    }
  }

  /**
   * 入力の転送・順伝播・出力の所有権と完了の通知を、共通のパイプラインへ渡す（デバイスの仕事は推論の道の {@link DeviceInference.SlotWork}）。
   */
  private final class InferenceWork implements DecisionDevicePipeline.Work {
    private final DeviceInference device;
    private final Batch batch;
    private final CompletableFuture<Void> result;
    private DeviceInference.SlotWork work;
    private Throwable failure;

    InferenceWork(DeviceInference device, Batch batch, CompletableFuture<Void> result) {
      this.device = device;
      this.batch = batch;
      this.result = result;
    }

    @Override
    public long estimatedWorkUnits() {
      return batch.encoding().paddedTokens();
    }

    @Override
    public DecisionDevicePipeline.BatchMetadata batchMetadata() {
      int rows = batch.encoding().rows().size();
      return new DecisionDevicePipeline.BatchMetadata(
          "epsilon-prism-v2", inference.name(), "scores", rows, rows);
    }

    /** 入力は提出のときに推論の道が転送するので、ここでは何もしない。 */
    @Override
    public void stage(DecisionDevicePipeline.Slot slot) {}

    @Override
    public void submit(DecisionDevicePipeline.Slot slot) {
      work = device.work(batch.encoding());
      work.submit(slot);
    }

    @Override
    public boolean isComplete(DecisionDevicePipeline.Slot slot) {
      return slot.computeReady().isComplete();
    }

    @Override
    public void complete(DecisionDevicePipeline.Slot slot) {
      batch.encoding().answer(work.scores(slot), prior);
      answered(batch.encoding());
    }

    @Override
    public void release(DecisionDevicePipeline.Slot slot) {
      if (work != null) {
        work.close();
        work = null;
      }
    }

    @Override
    public boolean abort(DecisionDevicePipeline.Slot slot, Throwable problem) {
      fail(problem);
      if (work != null && !work.settle(slot, problem)) {
        return false;
      }
      try {
        release(slot);
        return true;
      } catch (Throwable releaseFailure) {
        problem.addSuppressed(releaseFailure);
        return false;
      }
    }

    @Override
    public void fail(Throwable problem) {
      if (failure == null) {
        failure = problem;
      } else if (failure != problem) {
        failure.addSuppressed(problem);
      }
    }

    @Override
    public void publish() {
      if (failure == null) {
        result.complete(null);
      } else {
        result.completeExceptionally(failure);
      }
    }
  }
}

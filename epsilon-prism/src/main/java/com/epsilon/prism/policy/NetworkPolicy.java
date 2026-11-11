package com.epsilon.prism.policy;

import ai.djl.Device;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.DecisionDevicePipeline;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.agent.runtime.InferenceDispatcher;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.Limits;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
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
 * 1 つのバージョンの方策。要求が運ぶ記録 O（要求を出す側が取り込む）から、入力の形式の 1 行のライターで入力のバッチを作る
 * （{@link BatchEncoder}）。ネットワークの s は推論のバックエンド（{@link InferencePath}: eager か HIP、{@link DeviceInference}）で出す。
 * σ は s が最大の手に確率 1 を置く（同点は p0 で決める）。p0 は、{@code retain} の要求と、s の最大が同点の要求にだけ書く
 * （{@link GreedyAnswer}。p0 の特徴はライターが作る）。受付の区分は 1 つで、鳴きの行も同じ受付に入る（{@link #admission}）。
 *
 * <p>GPU では、Nano と共通の受付・予約・デバイスパイプラインに載せる。GPU が複数あれば、それぞれに同じバージョンのネットワークを置く。
 * 符号化は符号化のプールで進み、順伝播はスロットごとの計算ストリームで出す。そのため、次の受付の符号化と 2 つのスロットの順伝播が重なる。
 * CPU のネットワークでは、同じ処理を呼び出しスレッドで行う。
 *
 * <p>学習の行は収集が O から作る。入力は色の正準形（萬子・筒子・索子を入れ替えても変わらない）。処理量（{@link Throughput}）は、閉じる
 * ときにログに書く。
 */
public final class NetworkPolicy implements PolicyService {

  private static final Logger LOG = LoggerFactory.getLogger(NetworkPolicy.class);

  /** 計測がまだ始まっていない印。 */
  private static final long UNSET = Long.MIN_VALUE;

  /**
   * 方策の処理量。
   *
   * @param decisions 答えた要求の数
   * @param batches 受付の数
   * @param chunks 順伝播の回数（1 つの受付は、パディング込みのトークン数の上限とバッチの容量で分ける）
   * @param seconds 最初の受付から最後の答えまでの時間（秒）
   * @param encodeSeconds リーダーを付ける処理・計画・符号化に、各スレッドがかけた時間の合計（秒）
   */
  public record Throughput(
      long decisions, long batches, long chunks, double seconds, double encodeSeconds) {

    /** 1 秒あたりに答えた要求の数。 */
    public double decisionsPerSecond() {
      return seconds > 0 ? decisions / seconds : 0;
    }
  }

  /** デバイスごとに推論のバックエンドを開く関数。 */
  @FunctionalInterface
  private interface DeviceOpener {
    DeviceInference open(Device device) throws IOException;
  }

  private final InputFormat format;
  private final PolicyParams params;
  private final PriorSettings prior;
  private final InferencePath inference;
  private final List<DeviceInference> devices;
  private final NetworkRuntime runtime;
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
      InputFormat format,
      PolicyParams params,
      PriorSettings prior,
      InferencePath inference,
      List<DeviceInference> devices,
      NetworkRuntime runtime,
      int maxBatch,
      BatchEncoder encoder) {
    this.format = format;
    this.params = params;
    this.prior = prior;
    this.inference = inference;
    this.devices = devices;
    this.runtime = runtime;
    this.maxBatch = maxBatch;
    this.encoder = encoder;
    if (!devices.getFirst().device().isGpu()) {
      registration = null;
      return;
    }
    try {
      List<GpuBackend> backends = new ArrayList<>(devices.size());
      for (DeviceInference device : devices) {
        var lease = runtime.lease(device.device());
        leases.add(lease);
        backends.add(new GpuBackend(lease, device));
      }
      registration = runtime.dispatcher().register(backends);
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
   * 保存したネットワークを開く。p0 の係数は {@link PriorSettings#STANDARD}。推論のバックエンドは、ネットワークの形とデバイスで決める
   * （{@link InferencePath#resolve}）。HIP を選んでも HIP が扱えない形なら、eager にしてログに書く。HIP では先に libprism_hip.so を読み、
   * 読めなければ例外で止まる。
   *
   * @param params バージョン番号（HIP に渡す重みのバージョン番号にもなる）
   * @param model ネットワークのディレクトリ（weights.bin と model.json。architecture は epsilon-prism-v2）
   * @param devices GPU の列か、CPU 1 つ
   * @param runtime 符号化のプールと GPU の実行のリソース
   * @param maxBatch 1 回の受付で受ける行の上限
   * @param precision eager で計算するときの精度（HIP は BFLOAT16 だけ）
   * @param inference 設定で選んだ推論のバックエンド
   * @return 方策（ネットワークの所有権を持つ）
   * @throws IOException ネットワークが読めないとき
   * @throws IllegalArgumentException HIP に BFLOAT16 でない精度を渡したとき
   * @throws IllegalStateException HIP で libprism_hip.so が読めないか、GPU と合わないとき
   */
  public static NetworkPolicy open(
      PolicyParams params,
      Path model,
      List<Device> devices,
      NetworkRuntime runtime,
      int maxBatch,
      Precision precision,
      InferencePath inference)
      throws IOException {
    Objects.requireNonNull(precision, "precision");
    Objects.requireNonNull(inference, "inference");
    // エンジンを初めて取る前に、全てのモデルに共通する実行の既定値（PyTorch の CPU スレッド数）を決める
    NetworkRuntime.initialize();
    PrismShape shape = PrismNetwork.readShape(model);
    InputFormat format = PrismNetwork.readFormat(model);
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
          eager(
              params,
              model,
              devices,
              runtime,
              maxBatch,
              precision,
              PriorSettings.STANDARD,
              BatchEncoder.eagerBudget(devices.getFirst(), runtime));
      case HIP -> {
        if (precision != Precision.BFLOAT16) {
          throw new IllegalArgumentException("PRISM's native kernels compute in BFLOAT16, not " + precision);
        }
        yield hip(
            params,
            model,
            devices,
            runtime,
            maxBatch,
            PriorSettings.STANDARD,
            BatchEncoder.UNBOUNDED,
            HipInference.gpu(BatchPlan.Capacity.standard(format)));
      }
    };
  }

  /**
   * eager で開く。テストでは、パディング込みのトークン数の上限を小さくしてバッチを分ける。
   *
   * @param precision 順伝播の計算の精度
   * @param prior p0 の係数
   * @param tokenBudget 1 回の順伝播で扱うトークン数（パディング込み）の上限
   */
  static NetworkPolicy eager(
      PolicyParams params,
      Path model,
      List<Device> devices,
      NetworkRuntime runtime,
      int maxBatch,
      Precision precision,
      PriorSettings prior,
      int tokenBudget)
      throws IOException {
    InputFormat format = PrismNetwork.readFormat(model);
    return assemble(
        format,
        params,
        prior,
        InferencePath.EAGER,
        devices,
        runtime,
        maxBatch,
        new BatchEncoder(format, tokenBudget, BatchEncoder.Memory.DIRECT),
        device -> EagerInference.load(model, device, precision));
  }

  /**
   * HIP で開く。重みは CPU に読んだ FP32 のネットワークから 1 回だけ作り、全てのデバイスのレーンにコピーする。レーンの数は、GPU なら
   * スロットの数、CPU（テストの参照実装を呼び出しスレッドから呼ぶ）なら 1。
   *
   * @param tokenBudget 1 回の順伝播で扱うトークン数（パディング込み）の上限（本番は {@link BatchEncoder#UNBOUNDED}）
   * @param lanes レーンを作るファクトリ（本番は {@link HipInference#gpu}）
   */
  static NetworkPolicy hip(
      PolicyParams params,
      Path model,
      List<Device> devices,
      NetworkRuntime runtime,
      int maxBatch,
      PriorSettings prior,
      int tokenBudget,
      HipInference.Lanes lanes)
      throws IOException {
    WeightPack pack = WeightPack.load(model, params.version());
    InputFormat format = PrismNetwork.readFormat(model);
    int scores = Math.multiplyExact(maxBatch, Limits.MAX_LEGAL_ACTIONS);
    return assemble(
        format,
        params,
        prior,
        InferencePath.HIP,
        devices,
        runtime,
        maxBatch,
        new BatchEncoder(format, tokenBudget, BatchEncoder.Memory.PINNED),
        device ->
            HipInference.open(
                device, pack, device.isGpu() ? runtime.slots() : 1, scores, lanes));
  }

  /** デバイスごとに推論のバックエンドを開いて、方策を組み立てる。失敗したら、開いたバックエンドと作業領域・バッチのプールを閉じる。 */
  private static NetworkPolicy assemble(
      InputFormat format,
      PolicyParams params,
      PriorSettings prior,
      InferencePath inference,
      List<Device> devices,
      NetworkRuntime runtime,
      int maxBatch,
      BatchEncoder encoder,
      DeviceOpener opener)
      throws IOException {
    List<DeviceInference> opened = new ArrayList<>(devices.size());
    try {
      for (Device device : devices) {
        opened.add(opener.open(device));
      }
      return new NetworkPolicy(format, params, prior, inference, opened, runtime, maxBatch, encoder);
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

  /** ネットワークが読む入力の形式。 */
  public InputFormat format() {
    return format;
  }

  /** 使っている推論のバックエンド。設定で選んだものを、ネットワークの形とデバイスで決め直したもの。 */
  public InferencePath inference() {
    return inference;
  }

  @Override
  public boolean inline() {
    return false;
  }

  /** 行の上限は受付の区分に依らない。遷移の多い行は、パディング込みのトークン数の上限とバッチの容量で順伝播を分ける。 */
  @Override
  public int maxBatch(int admission) {
    return maxBatch;
  }

  /**
   * 全ての行を 1 つの区分で受け付ける。バッチは区分に依らず {@link BatchPlan.Capacity#standard} の容量で取り、行のイベント・遷移・手の
   * 実際の数で分ける。そのため鳴きの行を混ぜてよい（鳴きの行だけの小さな受付と、その待ちを作らない）。
   */
  @Override
  public int admission(Observation observation) {
    return 0;
  }

  @Override
  public boolean needsWork() {
    return registration == null || registration.needsWork();
  }

  @Override
  public synchronized DecisionInferenceIngress.Attempt tryAcquireIngress(int admission) {
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
              .encode(encoding, runtime)
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
    // 結果を公開するのはスロットとデバイスのリソースを返した後なので、ここで作業領域とバッチをプールに返してよい
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

  /** CPU のネットワーク。符号化から σ までを呼び出しスレッドで行う。 */
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

  /** 答えを書き終えた受付を数える。バッチをプールに返す前に呼ぶ。 */
  private void answered(Encoding encoding) {
    decisions.add(encoding.rows().size());
    batches.increment();
    chunks.add(encoding.chunks().size());
    last.accumulateAndGet(System.nanoTime(), Math::max);
  }

  /**
   * ここまでの処理量を返す。
   *
   * @return 処理量
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
    // スロットのリースを全て返す。次にデバイスごとのバックエンドを閉じ（実行計画はストリームを待ってから解放し、
    // ストリームとネットワークも閉じる）、最後に入力の blob を解放する
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

  /** 符号化を終えた 1 回の受付。 */
  private record Batch(Encoding encoding) {}

  /** 1 つの GPU に受付を渡すバックエンド。 */
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
   * 1 回の受付を共通のパイプラインに載せるジョブ。入力の転送・順伝播・出力の所有権と、完了の通知をパイプラインに渡す。デバイスの処理は
   * 推論のバックエンドの {@link DeviceInference.SlotWork} が行う。
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
      return new DecisionDevicePipeline.BatchMetadata(rows, rows);
    }

    /** 入力は提出のときに推論のバックエンドが転送するので、ここでは何もしない。 */
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

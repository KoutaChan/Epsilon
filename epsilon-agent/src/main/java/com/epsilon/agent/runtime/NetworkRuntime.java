package com.epsilon.agent.runtime;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtMemoryStats;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 1 つの処理（収集・学習・評価）がネットワークを動かすためのランタイム。持つのは、推論の受付（{@link InferenceDispatcher}）、GPU ごとの
 * 3 本のストリーム（{@link Streams}）とその上で動く推論のパイプライン、入力を符号化するスレッドプール。GPU のリソースは使うときに
 * 作り、CPU だけの処理では作らない。
 *
 * <p>PyTorch の作業領域のキャッシュはストリームごとにあるので、学習器は反復をまたいで同じランタイムからストリームを借りる。借り直すと、
 * 前の作業領域が使われないまま残る。学習だけに使うランタイム（{@link #forLearning}）は、ストリームを貸して処理の終わりを待つだけで、
 * 推論は実行しない。
 */
public final class NetworkRuntime implements AutoCloseable {

  private static final Logger LOG = LoggerFactory.getLogger(NetworkRuntime.class);

  /** 行数がこれより少なければ、符号化を分けずに 1 つのタスクで済ませる。 */
  private static final int SERIAL_ROWS = 128;

  /** 符号化のタスク 1 つが受け持つ行数の目安。 */
  private static final int ROWS_PER_TASK = 64;

  /** GPU ごとに同時に実行するバッチの数の上限。 */
  private static final int MAX_SLOTS = 4;

  static {
    // 小さなネットワークの CPU の推論を多くのスレッドから呼ぶので、PyTorch の CPU のスレッドは 1 つにする。DJL のシステムプロパティか OpenMP・MKL の環境変数で指定してあれば、そちらに従う
    String omp = System.getenv("OMP_NUM_THREADS");
    String mkl = System.getenv("MKL_NUM_THREADS");
    if (System.getProperty("ai.djl.pytorch.num_threads") == null
        && (omp == null || omp.isBlank())
        && (mkl == null || mkl.isBlank())) {
      System.setProperty("ai.djl.pytorch.num_threads", "1");
    }
  }

  /** 符号化のスレッド数。学習だけに使うランタイムでは 0。 */
  private final int encodingWorkers;

  /** GPU ごとに同時に実行するバッチの数。学習だけに使うランタイムでは 0。 */
  private final int slots;

  private final int readyBatches;
  private final InferenceDispatcher dispatcher = new InferenceDispatcher();
  private final Map<Device, Streams> devices = new LinkedHashMap<>();
  private ExecutorService encoder;
  private boolean closed;
  private Throwable drainFailure;

  /**
   * @param encodingWorkers 入力を符号化するスレッド数
   * @param slots GPU ごとに同時に実行するバッチの数（1〜4）
   * @param readyBatches GPU ごとに、符号化を終えて実行を待てるバッチの数（同時に実行するバッチの数以上）
   */
  public NetworkRuntime(int encodingWorkers, int slots, int readyBatches) {
    if (encodingWorkers < 1) {
      throw new IllegalArgumentException("Encoding workers must be positive: " + encodingWorkers);
    }
    if (slots < 1 || slots > MAX_SLOTS || readyBatches < slots) {
      throw new IllegalArgumentException(
          "GPU slots must be in [1, " + MAX_SLOTS + "] and ready batches must cover them: " + slots + '/' + readyBatches);
    }
    this.encodingWorkers = encodingWorkers;
    this.slots = slots;
    this.readyBatches = readyBatches;
  }

  private NetworkRuntime() {
    encodingWorkers = 0;
    slots = 0;
    readyBatches = 0;
  }

  /**
   * 学習だけに使うランタイムを作る。できるのは、GPU のストリームを貸すこと（{@link #streams}）と、全てのストリームの処理を待つこと
   * （{@link #awaitIdle}）だけ。入力の符号化と推論のパイプラインは持たないので、{@link #encodeRows}・{@link #lease}・{@link #slots}・
   * {@link #readyBatches} を呼ぶと IllegalStateException を投げる。
   */
  public static NetworkRuntime forLearning() {
    return new NetworkRuntime();
  }

  /** PyTorch のエンジンを初めて取る前に呼ぶ。中身は空で、呼ぶとこのクラスが初期化され、PyTorch が CPU で使うスレッド数の既定値が一度だけ決まる。 */
  public static void initialize() {}

  public InferenceDispatcher dispatcher() {
    return dispatcher;
  }

  /**
   * GPU ごとに同時に実行するバッチの数。
   *
   * @throws IllegalStateException 学習だけに使うランタイムのとき
   */
  public int slots() {
    requireInference();
    return slots;
  }

  /**
   * GPU ごとに、符号化を終えて実行を待てるバッチの数。
   *
   * @throws IllegalStateException 学習だけに使うランタイムのとき
   */
  public int readyBatches() {
    requireInference();
    return readyBatches;
  }

  /** GPU の 3 本のストリームを借りる。ストリームを閉じるのはこのランタイムで、借りた側は閉じない。 */
  public synchronized Streams streams(Device device) {
    if (closed) {
      throw new IllegalStateException("The network runtime is closed");
    }
    if (drainFailure != null) {
      throw new IllegalStateException("The network runtime failed to drain", drainFailure);
    }
    if (!device.isGpu()) {
      throw new IllegalArgumentException("Streams need a GPU device: " + device);
    }
    return devices.computeIfAbsent(device, Streams::new);
  }

  /**
   * GPU の推論のパイプラインを借りる。同じ GPU のモデルは 1 つのパイプラインを共有する。
   *
   * @throws IllegalStateException 学習だけに使うランタイムのとき
   */
  public DecisionDevicePipeline.Lease lease(Device device) {
    requireInference();
    return DecisionDevicePipeline.acquire(streams(device), slots, readyBatches);
  }

  /**
   * どの GPU の推論のパイプラインも、失敗でリソースを隔離していないことを確かめる。隔離は解けないので、隔離したパイプラインのある GPU では、
   * このランタイムはもう推論できない。
   *
   * @throws IllegalStateException 隔離したパイプラインがあるとき
   */
  public synchronized void requireUsable() {
    for (Streams streams : devices.values()) {
      DecisionDevicePipeline.requireNotQuarantined(streams);
    }
  }

  /**
   * 行を互いに重ならない範囲に分けて、符号化のスレッドプールで書く。全ての範囲を書き終えたときに完了する。
   *
   * @param rows 先頭から書く行数
   * @param encoder 1 つの範囲の行を書く処理。戻るまでに、借りた作業領域を返す
   * @throws IllegalStateException 学習だけに使うランタイムのとき
   */
  public CompletableFuture<Void> encodeRows(int rows, RangeEncoder encoder) {
    requireInference();
    int tasks = rows < SERIAL_ROWS ? 1 : Math.min(encodingWorkers, rows / ROWS_PER_TASK);
    return scheduleRows(rows, tasks, encoder);
  }

  /**
   * 全ての GPU で、全てのストリームの処理が終わるのを待つ。バッチを出すのを止めて、ほかの処理に切り替えるときに呼ぶ。推論のパイプラインが
   * 返されていなければ IllegalStateException を投げる。
   */
  public synchronized void awaitIdle() {
    if (closed) {
      throw new IllegalStateException("The network runtime is closed");
    }
    if (drainFailure != null) {
      throw new IllegalStateException("The network runtime failed to drain", drainFailure);
    }
    try {
      for (Streams streams : devices.values()) {
        DecisionDevicePipeline.requireReleased(streams);
        streams.awaitIdle();
      }
    } catch (RuntimeException | Error failure) {
      drainFailure = failure;
      throw failure;
    }
  }

  /** 出した推論が全て終わるまで待つ。モデルが借りたパイプラインは、借りたままにしておく。 */
  public synchronized void awaitInferenceIdle() {
    dispatcher.awaitIdle();
    for (Streams streams : devices.values()) {
      synchronized (streams) {
        if (streams.pipeline != null) {
          streams.pipeline.awaitIdle();
        }
      }
    }
  }

  /** 次の計測の区間を始める。バッチを出すのを止め、推論が終わった後に呼ぶ。GPU のメモリ使用量の最大も、ここから測り直す。 */
  public synchronized void resetInferenceMetrics() {
    dispatcher.resetMetrics();
    for (Map.Entry<Device, Streams> entry : devices.entrySet()) {
      Streams streams = entry.getValue();
      synchronized (streams) {
        if (streams.pipeline != null) {
          streams.pipeline.resetMetrics();
        }
      }
      engine().resetPeakMemoryStats(entry.getKey());
    }
  }

  /**
   * 計測の区間について、受付の集計と、GPU ごとのパイプラインの集計をログに書く。推論が終わった後、モデルがパイプラインを返す前に呼ぶ。
   * GPU ごとに、PyTorch がキャッシュしているメモリの量（今の値と区間の最大）、区間で使ったメモリの最大、キャッシュを空けて確保を
   * やり直した回数（プロセスの累計）も書く。
   */
  public void logInference(InferenceAdmission.Metrics admission) {
    InferenceDispatcher.Metrics dispatch = dispatcher.metrics();
    LOG.info(
        "Inference admission: admitted={} meanOldestRowWaitMs={} maxOldestRowWaitMs={} hostCapacity={}"
            + " peakReservations={}",
        admission.admitted(),
        milliseconds(admission.totalOldestWaitNanos(), admission.batches()),
        milliseconds(admission.maxOldestWaitNanos(), 1),
        dispatch.hostCapacity(),
        dispatch.peakReservations());
    for (DecisionDevicePipeline.Metrics device : pipelineMetrics()) {
      DecisionDevicePipeline.BatchMetrics total = device.total();
      LOG.info(
          "Inference device {}: elapsedMs={} pipelineIdleMs={} batches={} rows={} fillRatio={} fullBatches={}"
              + " meanReadyWaitMs={} meanStageMs={} meanSubmitQueueWaitMs={} meanSubmitMs={}"
              + " meanCompletionWaitMs={} meanComposeMs={} meanReclaimWaitMs={} meanReleaseMs={}"
              + " meanRoundTripMs={} maxHostReadyCells={}",
          device.device(),
          milliseconds(device.elapsedNanos(), 1),
          milliseconds(device.pipelineIdleNanos(), 1),
          total.batches(),
          total.rows(),
          format((double) total.rows() / total.capacity()),
          total.fullBatches(),
          milliseconds(total.readyWaitNanos(), total.batches()),
          milliseconds(total.stageNanos(), total.batches()),
          milliseconds(total.submitQueueWaitNanos(), total.batches()),
          milliseconds(total.submitNanos(), total.batches()),
          milliseconds(total.completionWaitNanos(), total.observedCompletions()),
          milliseconds(total.composeNanos(), total.observedCompletions()),
          milliseconds(total.reclaimWaitNanos(), total.observedCompletions()),
          milliseconds(total.releaseNanos(), total.observedCompletions()),
          milliseconds(total.roundTripNanos(), total.batches()),
          device.maxHostReadyCells());
    }
    for (Device device : gpus()) {
      PtMemoryStats memory = engine().getMemoryStats(device);
      LOG.info(
          "Inference device {} memory: reservedGb={} peakReservedGb={} peakAllocatedGb={} allocRetries={}",
          device,
          gigabytes(memory.getReservedBytes()),
          gigabytes(memory.getPeakReservedBytes()),
          gigabytes(memory.getPeakAllocatedBytes()),
          memory.getAllocationRetries());
    }
  }

  /** 処理の最後に呼ぶ。呼ぶ前に、全ての借り手と終わっていない処理を閉じておく。 */
  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    try {
      dispatcher.close();
      if (encoder != null) {
        encoder.close();
      }
      awaitIdle();
    } finally {
      closed = true;
    }
    Throwable failure = null;
    for (Streams streams : devices.values()) {
      failure = closeAfter(streams.d2h, failure);
      failure = closeAfter(streams.h2d, failure);
      failure = closeAfter(streams.compute, failure);
    }
    devices.clear();
    if (failure instanceof RuntimeException runtime) {
      throw runtime;
    }
    if (failure instanceof Error error) {
      throw error;
    }
  }

  /** 1 つの範囲（始まりを含み、終わりを含まない）の行を書く処理。戻るまでに、借りた作業領域を返す。 */
  @FunctionalInterface
  public interface RangeEncoder {
    void encode(int fromInclusive, int toExclusive);
  }

  private void requireInference() {
    if (slots == 0) {
      throw new IllegalStateException("A runtime for learning has no inference slots or encoders");
    }
  }

  private synchronized CompletableFuture<Void> scheduleRows(int rows, int tasks, RangeEncoder encodeRange) {
    if (closed) {
      throw new IllegalStateException("The network runtime is closed");
    }
    if (encoder == null) {
      encoder =
          Executors.newFixedThreadPool(encodingWorkers, Thread.ofPlatform().daemon().name("epsilon-encode-", 0).factory());
    }
    CompletableFuture<?>[] parts = new CompletableFuture<?>[tasks];
    for (int task = 0; task < tasks; task++) {
      int from = task * rows / tasks;
      int to = (task + 1) * rows / tasks;
      parts[task] = CompletableFuture.runAsync(() -> encodeRange.encode(from, to), encoder);
    }
    return CompletableFuture.allOf(parts);
  }

  private synchronized List<DecisionDevicePipeline.Metrics> pipelineMetrics() {
    List<DecisionDevicePipeline.Metrics> result = new ArrayList<>(devices.size());
    for (Streams streams : devices.values()) {
      synchronized (streams) {
        if (streams.pipeline != null) {
          result.add(streams.pipeline.metrics());
        }
      }
    }
    return result;
  }

  private synchronized List<Device> gpus() {
    return new ArrayList<>(devices.keySet());
  }

  private static PtEngine engine() {
    return (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
  }

  private static String milliseconds(long nanos, long count) {
    return format(nanos * 1e-6 / count);
  }

  private static String gigabytes(long bytes) {
    return format(bytes / 1e9);
  }

  private static String format(double value) {
    return String.format(Locale.ROOT, "%.3f", value);
  }

  private static Throwable closeAfter(PtStream stream, Throwable failure) {
    try {
      stream.close();
    } catch (RuntimeException | Error closeFailure) {
      if (failure == null) {
        return closeFailure;
      }
      failure.addSuppressed(closeFailure);
    }
    return failure;
  }

  /** 1 つの GPU の 3 本のストリーム（入力の転送・計算・出力の転送）。モデルもテンソルも持たない。借りる側は閉じない。 */
  public static final class Streams {

    private final Device device;
    private final PtStream compute;
    private final PtStream h2d;
    private final PtStream d2h;

    /** この GPU の推論のパイプライン。借り手が無ければ null。このオブジェクトのロックを取って読み書きする。 */
    DecisionDevicePipeline pipeline;

    private Streams(Device device) {
      this.device = device;
      PtEngine engine = (PtEngine) Engine.getEngine("PyTorch");
      compute = engine.newStream(device);
      PtStream createdH2d = null;
      try {
        createdH2d = engine.newStream(device);
        d2h = engine.newStream(device);
        h2d = createdH2d;
      } catch (RuntimeException | Error failure) {
        if (createdH2d != null) {
          closeAfter(createdH2d, failure);
        }
        closeAfter(compute, failure);
        throw failure;
      }
    }

    public Device device() {
      return device;
    }

    public PtStream compute() {
      return compute;
    }

    public PtStream h2d() {
      return h2d;
    }

    public PtStream d2h() {
      return d2h;
    }

    private void awaitIdle() {
      for (PtStream stream : List.of(h2d, compute, d2h)) {
        try (PtEvent completion = stream.newEvent()) {
          try (PtStreamScope ignored = stream.openScope()) {
            completion.record();
          }
          completion.synchronize();
        }
      }
    }
  }
}

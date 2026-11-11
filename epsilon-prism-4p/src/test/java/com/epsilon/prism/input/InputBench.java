package com.epsilon.prism.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.prism.Limits;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism4p.input.InputSchema;
import com.epsilon.prism4p.input.InputWriter;
import com.epsilon.prism4p.input.Tokenizer;
import com.sun.management.ThreadMXBean;
import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordedFrame;
import jdk.jfr.consumer.RecordingFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 観測の取り込みと Tokenizer にかかる CPU のコスト（µs/判断）を測るベンチマーク。新しいエンジンで GREEDY に打つ半荘の、全ての判断で
 * 測る。
 *
 * <p>本番の流れと同じく、取り込んだ O を {@value #BATCH_ROWS} 行ためてから計画（{@link BatchPlan}）を作り、行ごとにライターを付け替えて
 * {@link Tokenizer} でバッチに書く。測るのは取り込みと、計画と符号化（計画の前置和、付け替え、Tokenizer、ライターの確かめと写し）で、
 * 半荘の進行と打ち方の解析は測らない。1 スレッドの回（ウォームアップの後に同じ半荘を {@value #RUNS} 回測り、中央値も出す）と、
 * {@value #THREADS} スレッドで別々の半荘を並べて回す回を測り、ログ（{@code INPUT-BENCH} の行）に書く。1 スレッドの最後の回では、
 * 取り込みと符号化の中で割り当てたバイト数も数える。
 *
 * <p>{@code -Dprism.bench=true} のときだけ動く。{@code -Dprism.bench.jfr=<file>} を付けると、ウォームアップの後に 1 スレッドの回を JFR の
 * サンプルで記録し、メソッドごとに自身の時間と呼び出し先を含む時間の割合を書く（記録はそのテストの JVM の中で読む）。
 */
public class InputBench {
  private static final Logger log = LoggerFactory.getLogger(InputBench.class);

  private static final long SEED = 500;
  private static final int GAMES = 24;
  private static final int BATCH_ROWS = 64;
  private static final int THREADS = 15;
  private static final int RUNS = 5;
  private static final BatchPlan.Capacity CAPACITY =
      new BatchPlan.Capacity(
          BATCH_ROWS,
          BATCH_ROWS * Limits.MAX_EVENT_TOKENS,
          BATCH_ROWS * Limits.MAX_TRANSITIONS,
          BATCH_ROWS * Limits.MAX_LEGAL_ACTIONS);

  @Test(timeOut = 1_800_000)
  public void measureCaptureAndTokenizer() throws Exception {
    if (!Boolean.getBoolean("prism.bench")) {
      throw new SkipException("input benchmark runs only with -Dprism.bench=true");
    }
    log.info(environment());
    // JIT が Tokenizer の大きいメソッドを C2 でコンパイルし終えるまで（約 20 万判断）ウォームアップする。
    log.info(walk(SEED + 5_000, GAMES * 8, false).line("threads=1 warm-up"));
    log.info(walk(SEED + 7_000, GAMES * 4, false).line("threads=1 warm-up"));
    String profile = System.getProperty("prism.bench.jfr");
    if (profile != null) {
      profile(Path.of(profile));
    }
    // ほかの処理による揺れを避けるため、同じ半荘を 5 回測って中央値を出す。
    double[] totals = new double[RUNS];
    for (int run = 0; run < RUNS; run++) {
      Meter meter = walk(SEED, GAMES, false);
      totals[run] = meter.microsPerDecision();
      log.info(meter.line("threads=1 run=" + (run + 1)));
    }
    Arrays.sort(totals);
    log.info("INPUT-BENCH threads=1 median {} µs/decision", String.format(Locale.ROOT, "%.2f", totals[RUNS / 2]));
    log.info(walk(SEED, GAMES, true).line("threads=1 (allocation counted)"));

    ExecutorService pool = Executors.newFixedThreadPool(THREADS);
    try {
      parallel(pool, SEED + 1_000);
      List<Meter> meters = parallel(pool, SEED + 2_000);
      Meter total = new Meter(false);
      double slowest = 0;
      double fastest = Double.MAX_VALUE;
      for (Meter meter : meters) {
        total.add(meter);
        slowest = Math.max(slowest, meter.microsPerDecision());
        fastest = Math.min(fastest, meter.microsPerDecision());
      }
      log.info(
          total.line(
              String.format(
                  Locale.ROOT,
                  "threads=%d (per thread %.2f..%.2f µs)",
                  THREADS,
                  fastest,
                  slowest)));
    } finally {
      pool.shutdownNow();
    }
    log.info(environment());
  }

  private static List<Meter> parallel(ExecutorService pool, long seed) throws Exception {
    List<Future<Meter>> futures = new ArrayList<>();
    for (int thread = 0; thread < THREADS; thread++) {
      long first = seed + (long) thread * GAMES;
      futures.add(pool.submit(() -> walk(first, GAMES, false)));
    }
    List<Meter> meters = new ArrayList<>();
    for (Future<Meter> future : futures) {
      meters.add(future.get());
    }
    return meters;
  }

  private static Meter walk(long seed, int games, boolean allocation) {
    Meter meter = new Meter(allocation);
    new TableWalk(TENHOU_4P, TableWalk.Policy.GREEDY).run(seed, games, meter);
    meter.flush();
    return meter;
  }

  /** 取り込みと符号化の時間を測る。 */
  private static final class Meter implements TableWalk.Visitor {
    private final ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    private final ObservationBatch observations = new ObservationBatch(TENHOU_4P, 1 << 14);
    private final Observation read = new Observation(TENHOU_4P);
    private final Tokenizer tokenizer = new Tokenizer();
    private final InputWriter input = new InputWriter();
    private final HostBatch batch =
        new HostBatch(InputSchema.FORMAT, CAPACITY, ByteBuffer.allocateDirect(BatchPlan.maxBytes(InputSchema.FORMAT, CAPACITY)));
    private final boolean allocation;
    private final ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    private int pending;
    private long decisions;
    private long captureNanos;
    private long encodeNanos;
    private long bytes;
    private long allocated;

    Meter(boolean allocation) {
      this.allocation = allocation;
    }

    @Override
    public void decision(Engine engine, int slot) {
      Decisions boundary = engine.decisions();
      long before = allocation ? threads.getCurrentThreadAllocatedBytes() : 0L;
      long started = System.nanoTime();
      writer.capture(engine.view(boundary.seat(slot)), boundary, slot, observations);
      pending++;
      captureNanos += System.nanoTime() - started;
      if (allocation) {
        allocated += threads.getCurrentThreadAllocatedBytes() - before;
      }
      decisions++;
      if (pending == BATCH_ROWS) {
        flush();
      }
    }

    /** ためた O の計画を作り、全ての行を符号化する。 */
    void flush() {
      if (pending == 0) {
        return;
      }
      long before = allocation ? threads.getCurrentThreadAllocatedBytes() : 0L;
      long started = System.nanoTime();
      BatchPlan plan = batch.plan();
      plan.clear();
      for (int row = 0; row < pending; row++) {
        if (plan.add(read.bind(observations, row)) != row) {
          throw new IllegalStateException("benchmark batch is too small");
        }
      }
      bytes += plan.seal();
      for (int row = 0; row < pending; row++) {
        tokenizer.write(read.bind(observations, row), input.bind(batch, row));
      }
      encodeNanos += System.nanoTime() - started;
      if (allocation) {
        allocated += threads.getCurrentThreadAllocatedBytes() - before;
      }
      observations.clear();
      pending = 0;
    }

    void add(Meter other) {
      decisions += other.decisions;
      captureNanos += other.captureNanos;
      encodeNanos += other.encodeNanos;
      bytes += other.bytes;
    }

    double microsPerDecision() {
      return (captureNanos + encodeNanos) / 1000.0 / decisions;
    }

    String line(String label) {
      return String.format(
          Locale.ROOT,
          "INPUT-BENCH %s: decisions %d capture %.0f ns encode %.0f ns total %.2f µs/decision"
              + " transfer %.0f B/decision%s",
          label,
          decisions,
          (double) captureNanos / decisions,
          (double) encodeNanos / decisions,
          microsPerDecision(),
          (double) bytes / decisions,
          allocation
              ? String.format(Locale.ROOT, " allocated %.1f B/decision", (double) allocated / decisions)
              : "");
    }
  }

  /** 1 スレッドの回を JFR のサンプルで記録し、メソッドごとの時間の割合を書く。 */
  private static void profile(Path file) throws IOException {
    try (Recording recording = new Recording()) {
      recording.enable("jdk.ExecutionSample").withPeriod(Duration.ofMillis(1));
      recording.start();
      log.info(walk(SEED + 9_000, GAMES * 8, false).line("threads=1 profiled"));
      recording.stop();
      recording.dump(file);
    }
    Map<String, Integer> self = new HashMap<>();
    Map<String, Integer> inclusive = new HashMap<>();
    int samples = 0;
    for (RecordedEvent event : RecordingFile.readAllEvents(file)) {
      if (!event.getEventType().getName().equals("jdk.ExecutionSample")
          || event.getStackTrace() == null
          || event.getStackTrace().getFrames().isEmpty()) {
        continue;
      }
      List<RecordedFrame> frames = event.getStackTrace().getFrames();
      samples++;
      self.merge(name(frames.get(0)), 1, Integer::sum);
      Set<String> seen = new HashSet<>();
      for (RecordedFrame frame : frames) {
        String name = name(frame);
        if (name.startsWith("com.epsilon.") && seen.add(name)) {
          inclusive.merge(name, 1, Integer::sum);
        }
      }
    }
    log.info("INPUT-BENCH profile: samples {}", samples);
    print("self", self, samples, 40);
    print("incl", inclusive, samples, 60);
  }

  private static String name(RecordedFrame frame) {
    return frame.getMethod().getType().getName() + "." + frame.getMethod().getName();
  }

  private static void print(String label, Map<String, Integer> counts, int samples, int top) {
    counts.entrySet().stream()
        .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
        .limit(top)
        .forEach(
            entry ->
                log.info(
                    "INPUT-BENCH profile {} {}% {}",
                    label,
                    String.format(Locale.ROOT, "%5.1f", 100.0 * entry.getValue() / samples),
                    entry.getKey().replace("com.epsilon.", "")));
  }

  private static String environment() {
    OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
    StringBuilder gcs = new StringBuilder();
    for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
      gcs.append(gc.getName()).append('=').append(gc.getCollectionCount()).append('/');
      gcs.append(gc.getCollectionTime()).append("ms ");
    }
    return String.format(
        Locale.ROOT,
        "INPUT-BENCH env: java %s cpus %d loadavg %.1f gc %s",
        System.getProperty("java.version"),
        os.getAvailableProcessors(),
        os.getSystemLoadAverage(),
        gcs.toString().trim());
  }
}

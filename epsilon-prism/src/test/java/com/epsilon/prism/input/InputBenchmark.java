package com.epsilon.prism.input;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.Limits;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
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
 * 捕捉と Tokenizer の CPU の費用（µs/判断）を、GREEDY の打ち方の半荘の全ての網に問う判断で測る。
 *
 * <p>本番の流れと同じく、捕捉した O を {@value #BATCH_ROWS} 行ためてから計画を作り（{@link BatchPlan}）、行ごとに書き手を付け替えて
 * {@link Tokenizer} で塊に書く。測るのは捕捉と、計画・符号化（計画の前置和・付け替え・Tokenizer・書き手の確かめと写し）で、半荘の進行と打ち方の
 * 解析は測らない。1 スレッド（温めの後に同じ半荘を {@value #RUNS} 回。中央の値も出す）と、{@value #THREADS} スレッドで別々の半荘を並べた
 * 回を測り、ログ（{@code INPUT-BENCH} の行）に書く。1 スレッドの最後の回は、捕捉と符号化の中の割り当てのバイトも数える。
 *
 * <p>{@code -Dprism.bench=true} のときだけ動く。{@code -Dprism.bench.jfr=<file>} を付けると、温めの後に 1 スレッドの回を JFR の標本で
 * 記録し、メソッドごとの自分の時間と含む時間の割合を書く（記録はその試験の JVM の中で読む）。
 */
public class InputBenchmark {
  private static final Logger log = LoggerFactory.getLogger(InputBenchmark.class);

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
    // JIT が Tokenizer の大きいメソッドを C2 で仕上げるまで（約 20 万判断）温める。
    log.info(walk(SEED + 5_000, GAMES * 8, false).line("threads=1 warm-up"));
    log.info(walk(SEED + 7_000, GAMES * 4, false).line("threads=1 warm-up"));
    String profile = System.getProperty("prism.bench.jfr");
    if (profile != null) {
      profile(Path.of(profile));
    }
    // サーバのほかの仕事の揺れを避けるため、同じ半荘を 5 回測って中央の値を出す。
    double[] totals = new double[RUNS];
    for (int run = 0; run < RUNS; run++) {
      Meter meter = walk(SEED, GAMES, false);
      totals[run] = meter.microsPerDecision();
      log.info(meter.line("threads=1 run=" + (run + 1)));
    }
    Arrays.sort(totals);
    log.info("INPUT-BENCH threads=1 median {} µs/decision", String.format(Locale.ROOT, "%.2f", totals[RUNS / 2]));
    log.info(walk(SEED, GAMES, true).line("threads=1 (allocation counted)"));
    // 前線の腕（FrontierFeatures）は基準の費用と分けて、温めの後に同じ半荘で測る。
    log.info(walk(SEED + 3_000, GAMES * 2, false, true).line("threads=1 frontier warm-up"));
    log.info(walk(SEED, GAMES, true, true).line("threads=1 frontier"));

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
    return walk(seed, games, allocation, false);
  }

  private static Meter walk(long seed, int games, boolean allocation, boolean frontier) {
    Meter meter = new Meter(allocation, frontier);
    new DecisionWalk(DecisionWalk.Policy.GREEDY).run(seed, games, meter);
    meter.flush();
    return meter;
  }

  /** 捕捉と符号化を測る（前線の腕を測る回は、前線を書く時間も別に測る）。 */
  private static final class Meter implements DecisionWalk.Visitor {
    private final ObservationWriter writer = new ObservationWriter();
    private final ObservationArena arena = new ObservationArena(1 << 14);
    private final Observation read = new Observation();
    private final Tokenizer tokenizer = new Tokenizer();
    private final InputWriter input = new InputWriter();
    private final HostBatch batch =
        new HostBatch(CAPACITY, ByteBuffer.allocateDirect(BatchPlan.maxBytes(CAPACITY)));
    private final FrontierFeatures frontier = new FrontierFeatures();
    private final FrontierBatch frontierBatch =
        new FrontierBatch(batch, ByteBuffer.allocateDirect(FrontierBatch.maxBytes(CAPACITY)));
    private final int[] bases = new int[BATCH_ROWS];
    private final boolean allocation;
    private final boolean measureFrontier;
    private final ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    private int pending;
    private long decisions;
    private long captureNanos;
    private long encodeNanos;
    private long frontierNanos;
    private long bytes;
    private long allocated;

    Meter(boolean allocation) {
      this(allocation, false);
    }

    Meter(boolean allocation, boolean measureFrontier) {
      this.allocation = allocation;
      this.measureFrontier = measureFrontier;
    }

    @Override
    public void visit(GameEngine engine, EngineDecisionPoint point, int round) {
      PublicObservation o = engine.observation(point.player());
      List<Action> legal = point.legalActions();
      long before = allocation ? threads.getCurrentThreadAllocatedBytes() : 0L;
      long started = System.nanoTime();
      bases[pending++] = writer.capture(o, point.kind(), legal, arena);
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
      long[] words = arena.words();
      for (int row = 0; row < pending; row++) {
        if (plan.add(read.bind(words, bases[row])) != row) {
          throw new IllegalStateException("benchmark batch is too small");
        }
      }
      bytes += plan.seal();
      for (int row = 0; row < pending; row++) {
        tokenizer.write(read.bind(words, bases[row]), input.bind(batch, row));
      }
      encodeNanos += System.nanoTime() - started;
      if (measureFrontier) {
        long frontierStarted = System.nanoTime();
        for (int row = 0; row < pending; row++) {
          frontier.write(read.bind(words, bases[row]), frontierBatch, row);
        }
        frontierNanos += System.nanoTime() - frontierStarted;
      }
      if (allocation) {
        allocated += threads.getCurrentThreadAllocatedBytes() - before;
      }
      arena.clear();
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
              + " transfer %.0f B/decision%s%s",
          label,
          decisions,
          (double) captureNanos / decisions,
          (double) encodeNanos / decisions,
          microsPerDecision(),
          (double) bytes / decisions,
          measureFrontier
              ? String.format(Locale.ROOT, " frontier %.0f ns/decision", (double) frontierNanos / decisions)
              : "",
          allocation
              ? String.format(Locale.ROOT, " allocated %.1f B/decision", (double) allocated / decisions)
              : "");
    }
  }

  /** 1 スレッドの回を JFR の標本で記録し、メソッドごとの割合を書く。 */
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

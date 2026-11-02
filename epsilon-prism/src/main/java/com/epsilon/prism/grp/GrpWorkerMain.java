package com.epsilon.prism.grp;

import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 裏の作業の主クラス（学習の JVM と別の主クラスなので、走らせ役が学習の JVM を待つ照合に掛からない）。走らせ役が各反復の後に、GPU を隠し
 * nice 19・CPU 4 本の別の JVM で起動する: {@code GrpWorkerMain --run DIR --hold-dir DIR --own-hold NAME}。
 *
 * <ol>
 *   <li>採点の錠（{@link WorkerLock.Kind#SCORE}）が取れれば、ほかの札が消えるまで待ち（{@link ForeignHolds}）、MemAvailable が {@value
 *       #SCORE_MEMORY_GIB} GiB 以上なら未採点のコホートを採点する（{@link PhiScorer}。phiMonitor = SCORE のときだけ）。警報（{@link
 *       DriftAlarm}）は phiDrift が OFF でなければ書いた行ごとに進める。錠が取れなければ採点しない（待つ作業は 1 つだけ）。
 *   <li>phiSource = SELF_GRP で学習の錠（{@link WorkerLock.Kind#REFIT}）が取れれば、札を待ってから、AUTO なら効く時期の来た候補を入れ
 *       （{@link AutoSwitch}）、巻き戻しを点検し（{@link RollbackCheck}）、手動の要求か予定表（{@link RefitScheduler}）で決まれば 1 回学習する
 *       （{@link GrpRefitter}）。READY なら AUTO の切り替えをもう一度確かめる。学習は MemAvailable が {@value #REFIT_MEMORY_GIB} GiB 以上のとき
 *       だけ決める（足りなければ手動の要求を次の作業に残す）。
 * </ol>
 *
 * <p>心拍（{@link WorkerHeartbeat}）は錠を持つあいだ錠の種類ごとのファイルに書く。起動の初めに oom_score_adj を 1000 にする（メモリが尽きた
 * とき学習の JVM より先に殺される）。失敗は全て 1 回だけ記録して終わる（終了の状態は 0。系譜は止めない）。
 */
public final class GrpWorkerMain {
  private static final Logger LOG = LoggerFactory.getLogger(GrpWorkerMain.class);

  /** 学習を決めるのに要る MemAvailable（GiB）。 */
  static final long REFIT_MEMORY_GIB = 10;

  /** 採点を始めるのに要る MemAvailable（GiB）。 */
  static final long SCORE_MEMORY_GIB = 4;

  /** 札を確かめる間隔。 */
  static final Duration HOLD_POLL = Duration.ofSeconds(30);

  static final String USAGE =
      "GrpWorkerMain --run DIR --hold-dir DIR --own-hold NAME | GrpWorkerMain --run DIR --copy-jar";

  private GrpWorkerMain() {}

  /**
   * 引数。
   *
   * @param run run のディレクトリ
   * @param holdDirectory 札の置き場（--copy-jar なら null）
   * @param ownHold 走らせ役自身の札の名前（--copy-jar なら null）
   * @param copyJar jar を写してパスを出すだけなら true
   */
  record Arguments(Path run, Path holdDirectory, String ownHold, boolean copyJar) {

    /** 引数を読む（誤りは IllegalArgumentException）。 */
    static Arguments parse(String[] arguments) {
      Path run = null, holds = null;
      String own = null;
      boolean copy = false;
      for (int i = 0; i < arguments.length; i++) {
        String name = arguments[i];
        if (name.equals("--copy-jar")) {
          copy = true;
          continue;
        }
        if (i + 1 >= arguments.length) throw new IllegalArgumentException("Missing value for " + name);
        String value = arguments[++i];
        switch (name) {
          case "--run" -> run = Path.of(value);
          case "--hold-dir" -> holds = Path.of(value);
          case "--own-hold" -> own = value;
          default -> throw new IllegalArgumentException("Unknown argument " + name);
        }
      }
      if (run == null) throw new IllegalArgumentException("--run is required");
      if (copy && (holds != null || own != null))
        throw new IllegalArgumentException("--copy-jar takes only --run");
      if (!copy && (holds == null || own == null))
        throw new IllegalArgumentException("--hold-dir and --own-hold are required");
      return new Arguments(run, holds, own, copy);
    }
  }

  /**
   * 引数を読んで作業を 1 回行う。
   *
   * @return 終了の状態（引数の誤りだけ 0 でない）
   */
  public static int run(String[] arguments) {
    Arguments parsed;
    try {
      parsed = Arguments.parse(arguments);
    } catch (IllegalArgumentException wrong) {
      System.err.println(wrong.getMessage());
      System.err.println("usage: " + USAGE);
      return 2;
    }
    var files = new GrpFiles(parsed.run());
    if (parsed.copyJar()) {
      System.out.println(copyJar(files));
      return 0;
    }
    try {
      work(parsed, files);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      LOG.warn("the GRP worker was interrupted");
    } catch (IOException | RuntimeException failure) {
      LOG.warn("the GRP worker stopped: {}", failure.toString());
    }
    return 0;
  }

  public static void main(String[] arguments) {
    int status = run(arguments);
    if (status != 0) System.exit(status);
  }

  /** jar を grp/jars へ写したパス（写せなければ今の jar のまま。走らせ役はどちらでも起動できる）。 */
  static Path copyJar(GrpFiles files) {
    Path running = null;
    try {
      running = WorkerJar.running();
      return WorkerJar.copy(files, running);
    } catch (IOException | RuntimeException failure) {
      LOG.warn("the worker jar was not copied: {}", failure.toString());
      return running == null ? Path.of("") : running;
    }
  }

  private static void work(Arguments arguments, GrpFiles files) throws IOException, InterruptedException {
    PrismConfig config =
        PrismConfig.read(Files.readString(files.run().resolve("config.json"), StandardCharsets.UTF_8));
    config.validate();
    yieldToOomKiller();
    String jar = jarSha256(files);
    if (config.phiMonitor().scores()) score(arguments, files, config, jar);
    if (config.phiSource() == PhiSource.SELF_GRP) refit(arguments, files, config, jar);
  }

  /** メモリが尽きたとき、学習の JVM より先にこの作業が選ばれるようにする（特権なしで上げられる。書けなければ 1 回書き残すだけ）。 */
  private static void yieldToOomKiller() {
    try {
      Files.writeString(Path.of("/proc/self/oom_score_adj"), "1000\n", StandardCharsets.US_ASCII);
    } catch (IOException | RuntimeException unsupported) {
      LOG.info("oom_score_adj was not raised: {}", unsupported.toString());
    }
  }

  /** 採点の錠を持って、未採点のコホートを採点する。 */
  private static void score(Arguments arguments, GrpFiles files, PrismConfig config, String jar)
      throws IOException, InterruptedException {
    Optional<WorkerLock> lock = WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE);
    if (lock.isEmpty()) {
      LOG.info("another GRP worker holds the score lock");
      return;
    }
    try (WorkerLock held = lock.get();
        var heartbeat = WorkerHeartbeat.start(files, WorkerLock.Kind.SCORE, "score: waiting for holds")) {
      ForeignHolds.awaitClear(arguments.holdDirectory(), arguments.ownHold(), HOLD_POLL);
      long available = memAvailableBytes();
      if (available >= 0 && available < SCORE_MEMORY_GIB << 30) {
        LOG.info("no scoring: MemAvailable {} MiB", available >> 20);
        return;
      }
      List<Integer> pending = PhiScorer.pending(files);
      for (int cohort : pending) {
        heartbeat.state(String.format("scoring c%05d", cohort));
        Optional<PhiMonitorLine> line;
        try {
          line = PhiScorer.score(files, config, cohort, jar);
        } catch (IOException | RuntimeException failure) {
          LOG.warn("c{} was not scored: {}", cohort, failure.toString());
          continue;
        }
        if (line.isEmpty()) continue;
        LOG.info("{}", line.get().summary());
        if (config.phiDrift() == PhiDrift.OFF) continue;
        try {
          DriftAlarm.State state = DriftAlarm.update(files, config, line.get());
          LOG.info("drift[{}] on={} streak={} ewma={}", cohort, state.on(), state.streak(), state.dSpgEwma());
        } catch (IOException | RuntimeException failure) {
          LOG.warn("c{} did not advance the drift alarm: {}", cohort, failure.toString());
        }
      }
    }
  }

  /** 学習の錠を持って、自動の切り替え・巻き戻しの点検・学習を行う。 */
  private static void refit(Arguments arguments, GrpFiles files, PrismConfig config, String jar)
      throws IOException, InterruptedException {
    Optional<WorkerLock> lock = WorkerLock.tryAcquire(files, WorkerLock.Kind.REFIT);
    if (lock.isEmpty()) {
      LOG.info("another GRP worker holds the refit lock");
      return;
    }
    try (WorkerLock held = lock.get();
        var heartbeat = WorkerHeartbeat.start(files, WorkerLock.Kind.REFIT, "refit: waiting for holds")) {
      ForeignHolds.awaitClear(arguments.holdDirectory(), arguments.ownHold(), HOLD_POLL);
      autoSwitch(files, config, heartbeat);
      heartbeat.state("rollback check");
      try {
        RollbackCheck.check(files, config).ifPresent(verdict -> LOG.info("rollback check {}", verdict));
      } catch (IOException | RuntimeException failure) {
        LOG.warn("the rollback check failed: {}", failure.toString());
      }
      long available = memAvailableBytes();
      if (available >= 0 && available < REFIT_MEMORY_GIB << 30) {
        LOG.info("no refit: MemAvailable {} MiB", available >> 20);
        return;
      }
      try {
        Optional<RefitScheduler.Decision> decision = RefitScheduler.decide(files, config, lastSealed(files));
        if (decision.isEmpty()) return;
        heartbeat.state(String.format("refit h=%d %s", decision.get().holdout(), decision.get().cause()));
        RefitRecord record =
            GrpRefitter.refit(files, config, decision.get().holdout(), decision.get().cause(), jar);
        LOG.info("refit {} {} {}", record.id(), record.status(), record.detail());
        if (record.status() == RefitStatus.READY) autoSwitch(files, config, heartbeat);
      } catch (IOException | RuntimeException failure) {
        LOG.warn("the refit failed: {}", failure.toString());
      }
    }
  }

  /**
   * AUTO なら効く時期の来た候補を入れる。学習の直後にも呼ぶ: 取り置き h の候補は反復 h + grpSwitchLag で効くので、学習の JVM が反復 h+1 を
   * 回しているうちに書かないと、次の作業の起動（反復 h+1 の後）では h+2 が既に始まっていて SKIPPED_NOT_READY になる。
   */
  private static void autoSwitch(GrpFiles files, PrismConfig config, WorkerHeartbeat heartbeat) {
    if (config.grpSwitch() != GrpSwitch.AUTO) return;
    heartbeat.state("auto switch");
    try {
      AutoSwitch.apply(files, config).ifPresent(entry -> LOG.info("auto switch to {}", entry));
    } catch (IOException | RuntimeException failure) {
      LOG.warn("the automatic switch failed: {}", failure.toString());
    }
  }

  /** 動いている jar の sha（分からなければ空）。写しから動いていなければ書き残す。 */
  private static String jarSha256(GrpFiles files) {
    try {
      Path running = WorkerJar.running();
      if (!running.toAbsolutePath().normalize().startsWith(files.jars().toAbsolutePath().normalize()))
        LOG.warn("the GRP worker runs from {}, not from a copy in {}", running, files.jars());
      return WorkerJar.runningSha256();
    } catch (IOException | RuntimeException failure) {
      LOG.warn("the worker jar sha is unknown: {}", failure.toString());
      return "";
    }
  }

  /** 対局の記録がある最新の封印済みのコホート（なければ 0）。 */
  static int lastSealed(GrpFiles files) throws IOException {
    for (int cohort : PhiScorer.recorded(files).descendingSet())
      if (Files.exists(files.cohort(cohort).resolve(CohortWriter.MANIFEST))) return cohort;
    return 0;
  }

  /** /proc/meminfo の MemAvailable（バイト。読めなければ −1）。 */
  static long memAvailableBytes() {
    try {
      for (String line : Files.readAllLines(Path.of("/proc/meminfo"), StandardCharsets.US_ASCII))
        if (line.startsWith("MemAvailable:"))
          return Long.parseLong(line.replaceAll("[^0-9]", "")) << 10;
    } catch (IOException | RuntimeException unknown) {
      // Linux でなければ分からない
    }
    return -1;
  }
}

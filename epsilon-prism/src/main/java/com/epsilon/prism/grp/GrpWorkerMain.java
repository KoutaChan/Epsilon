package com.epsilon.prism.grp;

import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GRP の裏の作業の主クラス（学習の JVM と別の主クラスなので、走らせ役が学習の JVM を待つ照合に掛からない）。走らせ役が反復の後に、GPU を隠し
 * nice 19・CPU 4 本の別の JVM で起動する: {@code GrpWorkerMain --run DIR}。学習の JVM は採点だけを自分のスレッドで行う（{@link PhiScoring}）。
 *
 * <ol>
 *   <li>phiMonitor = SCORE で採点の錠（{@link WorkerLock.Kind#SCORE}）が取れれば、未採点のコホートを採点する（{@link PhiScoring#scorePending}）。
 *       錠が取れなければ採点しない（学習の JVM が採点している）。
 *   <li>phiSource = SELF_GRP で学習の錠（{@link WorkerLock.Kind#REFIT}）が取れれば、AUTO なら効く時期の来た候補を入れ（{@link AutoSwitch}）、
 *       巻き戻しを点検し（{@link RollbackCheck}）、手動の要求か予定表（{@link RefitScheduler}）で決まれば 1 回学習する（{@link GrpRefitter}）。
 *       READY なら AUTO の切り替えをもう一度確かめる。学習は MemAvailable が {@value #REFIT_MEMORY_GIB} GiB 以上のときだけ決める（足りなければ
 *       手動の要求を次の作業に残す）。
 * </ol>
 *
 * <p>心拍（{@link WorkerHeartbeat}）は錠を持つあいだ錠の種類ごとのファイルに書く。起動の初めに oom_score_adj を 1000 にする（メモリが尽きた
 * とき学習の JVM より先に殺される）。失敗は全て 1 回だけ記録して終わる（終了の状態は 0。系譜は止めない）。
 */
public final class GrpWorkerMain {
  private static final Logger LOG = LoggerFactory.getLogger(GrpWorkerMain.class);

  /** 学習を決めるのに要る MemAvailable（GiB）。 */
  static final long REFIT_MEMORY_GIB = 10;

  static final String USAGE = "GrpWorkerMain --run DIR";

  private GrpWorkerMain() {}

  /**
   * 引数（{@code --run DIR}）を読む。
   *
   * @return run のディレクトリ
   * @throws IllegalArgumentException 引数が合わないとき
   */
  static Path parse(String[] arguments) {
    if (arguments.length != 2 || !arguments[0].equals("--run"))
      throw new IllegalArgumentException("Expected --run DIR, got " + String.join(" ", arguments));
    return Path.of(arguments[1]);
  }

  /**
   * 引数を読んで作業を 1 回行う。
   *
   * @return 終了の状態（引数の誤りだけ 0 でない）
   */
  public static int run(String[] arguments) {
    Path run;
    try {
      run = parse(arguments);
    } catch (IllegalArgumentException wrong) {
      System.err.println(wrong.getMessage());
      System.err.println("usage: " + USAGE);
      return 2;
    }
    var files = new GrpFiles(run);
    try {
      work(files);
    } catch (IOException | RuntimeException failure) {
      LOG.warn("the GRP worker stopped: {}", failure.toString());
    }
    return 0;
  }

  public static void main(String[] arguments) {
    int status = run(arguments);
    if (status != 0) System.exit(status);
  }

  private static void work(GrpFiles files) throws IOException {
    PrismConfig config =
        PrismConfig.read(Files.readString(files.run().resolve("config.json"), StandardCharsets.UTF_8));
    config.validate();
    yieldToOomKiller();
    if (config.phiMonitor().scores()) score(files, config);
    if (config.phiSource() == PhiSource.SELF_GRP) refit(files, config);
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
  private static void score(GrpFiles files, PrismConfig config) throws IOException {
    Optional<WorkerLock> lock = WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE);
    if (lock.isEmpty()) {
      LOG.info("another GRP worker holds the score lock");
      return;
    }
    try (WorkerLock held = lock.get();
        var heartbeat = WorkerHeartbeat.start(files, WorkerLock.Kind.SCORE, "score: starting")) {
      PhiScoring.scorePending(files, config, heartbeat::state);
    }
  }

  /** 学習の錠を持って、自動の切り替え・巻き戻しの点検・学習を行う。 */
  private static void refit(GrpFiles files, PrismConfig config) throws IOException {
    Optional<WorkerLock> lock = WorkerLock.tryAcquire(files, WorkerLock.Kind.REFIT);
    if (lock.isEmpty()) {
      LOG.info("another GRP worker holds the refit lock");
      return;
    }
    try (WorkerLock held = lock.get();
        var heartbeat = WorkerHeartbeat.start(files, WorkerLock.Kind.REFIT, "refit: starting")) {
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
            GrpRefitter.refit(files, config, decision.get().holdout(), decision.get().cause());
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

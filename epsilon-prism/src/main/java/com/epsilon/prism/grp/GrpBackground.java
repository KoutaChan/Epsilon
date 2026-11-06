package com.epsilon.prism.grp;

import ai.djl.engine.Engine;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習の JVM の GRP の裏の作業。デーモンのスレッド 1 本で、頼まれるたびに 1 回: 未採点のコホートの採点（phiMonitor = SCORE。{@link PhiScorer}。
 * phiDrift が OFF でなければ書いた行ごとに警報 {@link DriftAlarm} を進める）と、SELF_GRP なら巻き戻しの点検（{@link RollbackCheck}）と
 * 学習（{@link RefitScheduler} が決めれば {@link GrpRefitter}）。待ちは 1 つまで（最中に頼まれた分は次の 1 回にまとめる。未採点のコホートと学習の
 * 時期は毎回ファイルから決めるので取りこぼさない）。失敗は全て警告を書くだけで、系譜は止めない。合格した候補を入れるのは学習の JVM の反復の
 * 始め（{@link AutoSwitch}）。
 *
 * <p>閉じるときは止める合図を立てる: 採点はコホートの途中で止めず（phimon.jsonl と警報の状態を書きかけにしない）、次のコホートと学習を
 * 始めない。学習は次の歩の前に止まり、試行を捨てる（{@link GrpRefitter}）。
 */
public final class GrpBackground implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(GrpBackground.class);

  /** 閉じるときに今の作業を待つ間、1 行書く間隔（秒）。 */
  private static final long CLOSE_NOTE_SECONDS = 60;

  private final GrpFiles files;
  private final PrismConfig config;

  /** 作業のスレッド（採点も学習もしなければ null）。 */
  private final ThreadPoolExecutor executor;

  private volatile boolean stopping;

  private GrpBackground(GrpFiles files, PrismConfig config, ThreadPoolExecutor executor) {
    this.files = files;
    this.config = config;
    this.executor = executor;
  }

  /** 作業のスレッドを始める（phiMonitor = SCORE でも SELF_GRP でもなければ何もしない）。 */
  public static GrpBackground start(GrpFiles files, PrismConfig config) {
    ThreadPoolExecutor executor = null;
    if (config.phiMonitor().scores() || config.phiSource() == PhiSource.SELF_GRP)
      executor =
          new ThreadPoolExecutor(
              1,
              1,
              0,
              TimeUnit.SECONDS,
              new ArrayBlockingQueue<>(1),
              task -> {
                Thread thread = new Thread(task, "grp-background");
                thread.setDaemon(true);
                return thread;
              },
              new ThreadPoolExecutor.DiscardPolicy());
    return new GrpBackground(files, config, executor);
  }

  /** 1 回の作業を頼む（待たない。既に 1 回待っていれば何もしない）。 */
  public void request() {
    if (executor != null) executor.execute(this::pass);
  }

  /** 1 回分: 採点し、SELF_GRP なら巻き戻しを点検して、時期なら学習する。 */
  void pass() {
    try {
      if (config.phiMonitor().scores()) scorePending();
      if (config.phiSource() == PhiSource.SELF_GRP && !stopping) {
        rollbackCheck();
        refit();
      }
    } catch (Throwable failure) {
      LOG.warn("the GRP background stopped: {}", failure.toString());
    }
  }

  /** 未採点のコホート（{@link PhiScorer#pending}）を古い順に採点する。コホートごとの失敗は警告を書いて次に進む。 */
  private void scorePending() throws IOException {
    List<Integer> pending = PhiScorer.pending(files);
    for (int cohort : pending) {
      if (stopping) return;
      Optional<PhiMonitorLine> line;
      try {
        line = PhiScorer.score(files, config, cohort);
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

  private void rollbackCheck() {
    try {
      RollbackCheck.check(files, config).ifPresent(verdict -> LOG.info("rollback check {}", verdict));
    } catch (IOException | RuntimeException failure) {
      LOG.warn("the rollback check failed: {}", failure.toString());
    }
  }

  /** 時期なら 1 回学習する。torch の演算スレッドは学習の間だけ grpThreads にする。 */
  private void refit() {
    try {
      Optional<RefitScheduler.Decision> decision = RefitScheduler.decide(files, config, lastSealed(files));
      if (decision.isEmpty() || stopping) return;
      LOG.info("GRP refit h={} {} started on {} torch threads", decision.get().holdout(), decision.get().cause(), config.grpThreads());
      // JniUtils は torch の native を読み込まない。読み込むのはエンジンを開いたとき
      Engine.getEngine("PyTorch");
      int threads = JniUtils.getNumThreads();
      JniUtils.setNumThreads(config.grpThreads());
      try {
        GrpRefitter.refit(files, config, decision.get().holdout(), decision.get().cause(), () -> stopping)
            .ifPresent(record -> LOG.info("GRP refit {} {} {}", record.id(), record.status(), record.detail()));
      } finally {
        JniUtils.setNumThreads(threads);
      }
    } catch (IOException | RuntimeException failure) {
      LOG.warn("the GRP refit failed: {}", failure.toString());
    }
  }

  /** 対局の記録がある最新の封印済みのコホート（なければ 0）。 */
  static int lastSealed(GrpFiles files) throws IOException {
    for (int cohort : PhiScorer.recorded(files).descendingSet())
      if (Files.exists(files.cohort(cohort).resolve(CohortWriter.MANIFEST))) return cohort;
    return 0;
  }

  /** 止める合図を立て、待ちを捨て、今の作業が終わるのを待つ。 */
  @Override
  public void close() {
    if (executor == null) return;
    stopping = true;
    executor.getQueue().clear();
    executor.shutdown();
    boolean interrupted = false;
    while (true) {
      try {
        if (executor.awaitTermination(CLOSE_NOTE_SECONDS, TimeUnit.SECONDS)) break;
        LOG.info("waiting for the GRP background to finish");
      } catch (InterruptedException ignored) {
        interrupted = true;
      }
    }
    if (interrupted) Thread.currentThread().interrupt();
  }
}

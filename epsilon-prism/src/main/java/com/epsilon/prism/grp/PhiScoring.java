package com.epsilon.prism.grp;

import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 封印したコホートの対局の記録の採点（{@link PhiScorer}。phiDrift が OFF でなければ書いた行ごとに警報 {@link DriftAlarm} を進める）。学習の
 * JVM ではデーモンのスレッド 1 本で行い（{@link #start}。nice と CPU は JVM と同じ）、待ちは 1 つまで（採点の最中に頼まれた分は次の 1 回に
 * まとめる。未採点のコホートは毎回 {@link PhiScorer#pending} で選ぶので取りこぼさない）。採点の錠（{@link WorkerLock.Kind#SCORE}）が取れなければ
 * 採点しない。失敗は全て警告を書くだけで、系譜は止めない。
 */
public final class PhiScoring implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(PhiScoring.class);

  /** 閉じるときに今の採点を待つ間、1 行書く間隔（秒）。 */
  private static final long CLOSE_NOTE_SECONDS = 60;

  private final GrpFiles files;
  private final PrismConfig config;

  /** 採点のスレッド（phiMonitor が SCORE でなければ null）。 */
  private final ThreadPoolExecutor executor;

  private PhiScoring(GrpFiles files, PrismConfig config, ThreadPoolExecutor executor) {
    this.files = files;
    this.config = config;
    this.executor = executor;
  }

  /** 採点のスレッドを始める（phiMonitor が SCORE でなければ何もしない）。 */
  public static PhiScoring start(GrpFiles files, PrismConfig config) {
    ThreadPoolExecutor executor = null;
    if (config.phiMonitor().scores())
      executor =
          new ThreadPoolExecutor(
              1,
              1,
              0,
              TimeUnit.SECONDS,
              new ArrayBlockingQueue<>(1),
              task -> {
                Thread thread = new Thread(task, "phi-scoring");
                thread.setDaemon(true);
                return thread;
              },
              new ThreadPoolExecutor.DiscardPolicy());
    return new PhiScoring(files, config, executor);
  }

  /** 未採点のコホートの採点を頼む（待たない。既に 1 回待っていれば何もしない）。 */
  public void request() {
    if (executor != null) executor.execute(this::pass);
  }

  /** 1 回分: 錠を取って、未採点のコホートを全て採点する。 */
  private void pass() {
    try {
      Optional<WorkerLock> lock = WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE);
      if (lock.isEmpty()) {
        LOG.info("another GRP worker holds the score lock");
        return;
      }
      try (WorkerLock held = lock.get()) {
        scorePending(files, config, state -> {});
      }
    } catch (Throwable failure) {
      LOG.warn("the Φ scoring stopped: {}", failure.toString());
    }
  }

  /**
   * 未採点のコホート（{@link PhiScorer#pending}）を古い順に採点する。錠は呼び手が取る。コホートごとの失敗は警告を書いて次に進む。
   *
   * @param progress 今の状態（例 scoring c00312）を受け取る口
   * @throws IOException 未採点のコホートが選べないとき
   */
  public static void scorePending(GrpFiles files, PrismConfig config, Consumer<String> progress) throws IOException {
    List<Integer> pending = PhiScorer.pending(files);
    for (int cohort : pending) {
      progress.accept(String.format("scoring c%05d", cohort));
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

  /** 待ちを捨て、今の採点が終わるのを待つ（割り込まない。phimon.jsonl と警報の状態を書きかけにしないため）。 */
  @Override
  public void close() {
    if (executor == null) return;
    executor.getQueue().clear();
    executor.shutdown();
    boolean interrupted = false;
    while (true) {
      try {
        if (executor.awaitTermination(CLOSE_NOTE_SECONDS, TimeUnit.SECONDS)) break;
        LOG.info("waiting for the Φ scoring to finish");
      } catch (InterruptedException ignored) {
        interrupted = true;
      }
    }
    if (interrupted) Thread.currentThread().interrupt();
  }
}

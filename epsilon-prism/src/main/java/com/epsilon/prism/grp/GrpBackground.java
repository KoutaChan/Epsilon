package com.epsilon.prism.grp;

import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import java.io.IOException;
import java.time.Duration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習の JVM で、GRP の作業を裏で行う。動くのは自分の GRP の run だけ。デーモンのスレッド 1 本で、要求されるたびに次の作業を順に
 * 1 回行う。まず、反復の区切りの処理のうちまだ済んでいないものを古い順に行う（{@link GrpLifecycle}）。次に、採点する設定なら未採点の
 * コホートを採点する（{@link PhiScorer}）。最後に、run の control にある手動の学び直しの要求に番号の順で従い、最新の
 * 確定済みのコホート n で M(n) を学ぶ（{@link GrpRefitter}）。
 *
 * <p>学び直しの要求には 1 回だけ従い、従った後で control/done に移す。時点のコホートがまだ確定していない要求は、確定する
 * まで残す。閉じるために学び直しを止めた要求も残し、次の起動で従う。監視が学び直しを止めている間（HALT）も、従わずに残す。
 * 取り消しの要求が指す要求には従わず、学習の途中で指されたら学習を止めて試行を捨てる。どちらの要求も、消すのはセーフポイントの確かめに
 * 任せる。
 *
 * <p>待たせておける要求は 1 つまでで、実行中に来た要求は次の 1 回にまとめる。何を済ませるかは毎回ファイルから決めるので、取りこぼしは
 * ない。失敗しても警告を書くだけで、系譜は止めない。
 *
 * <p>反復 t の収集は、区切り t − 1 の処理が済んでから始める（{@link #awaitBoundary}）。区切り t − 1 の処理は M(t − 2) を学んだ後に
 * 行うので、学習が反復 1 つの間に終わらなければ系譜が待つ。待った時間は呼び出し元が記録する。
 *
 * <p>処理を済ませた区切りでも、失敗した処理を済ませ直す間と、手動で学習し直した候補を関門に通す間は、次の反復の Φ の段を待たせる。
 * Φ の段がもう通っていれば待たせず、その反復から効く行を phi.jsonl に足さない（決まった Φ を書き換えない）。
 *
 * <p>run が一時停止している間は、区切り・採点するコホート・学び直しのそれぞれの前で待つ（{@link Gate}）。
 *
 * <p>閉じるときは停止のフラグを立てる。区切りと採点はコホートの途中では止めないが、次のコホートと学習は始めない。学習は次のステップの
 * 前に止まり、その試行を捨てる。
 */
public final class GrpBackground implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(GrpBackground.class);

  /** 閉じるときに実行中の作業を待つ間、ログを 1 行書く間隔（秒）。 */
  private static final long CLOSE_NOTE_SECONDS = 60;

  /** 学び直しの途中で、頼みが取り消されたかを control から読む間隔（ナノ秒）。 */
  private static final long CANCEL_CHECK_NANOS = TimeUnit.SECONDS.toNanos(1);

  /** 作業の合間に通るゲート。run が一時停止している間は戻らない。 */
  @FunctionalInterface
  public interface Gate {
    /** 一時停止している間は待つ。停止が決まったら待たずに戻る。 */
    void awaitRunning() throws InterruptedException;
  }

  private final GrpFiles files;
  private final PhiSetup setup;
  private final GrpLifecycle lifecycle;
  private final Gate gate;
  private final ControlRequests requests;

  /** 作業のスレッド。採点も学び直しもしない run では null。 */
  private final ThreadPoolExecutor executor;

  /** 処理を済ませた最後の区切り。処理に失敗した区切りも済ませたとみなす。 */
  private int processed;

  /** Φ の段を通した最後の区切り。区切り h を通した後は、反復 h ＋ 1 の Φ が決まったものとして扱う。 */
  private int passed;

  /** 処理を済ませた後も、Φ の段を待たせている区切り（無ければ 0）。 */
  private int held;

  private volatile boolean stopping;

  private GrpBackground(GrpFiles files, PhiSetup setup, Gate gate, ThreadPoolExecutor executor) {
    this.files = files;
    this.setup = setup;
    this.gate = gate;
    this.executor = executor;
    lifecycle = setup.selfPlay() ? new GrpLifecycle(files, setup) : null;
    requests = new ControlRequests(new RunDirectory(files.run()));
  }

  /**
   * 作業のスレッドを始める。自分の GRP の run でなければスレッドを作らない。
   *
   * @param gate 作業の合間に通るゲート
   */
  public static GrpBackground start(GrpFiles files, PhiSetup setup, Gate gate) {
    ThreadPoolExecutor executor = null;
    if (setup.selfPlay())
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
    return new GrpBackground(files, setup, gate, executor);
  }

  /** 作業を 1 回要求する。終わるのは待たない。既に 1 回分が待っていれば何もしない。 */
  public void request() {
    if (executor != null) executor.execute(this::pass);
  }

  /**
   * 区切りの処理が済むまで待つ。済んだ区切りでも、作業のスレッドが待たせている間は待つ。自分の GRP の run でないか、コホートが 1 未満
   * なら待たない。閉じ始めたら待つのをやめる。
   *
   * @param cohort 待つ区切りのコホート
   * @param timeout 待つ時間の上限
   * @return 区切りの処理が済んだか、待つ必要が無いか、閉じ始めたなら true。時間切れなら false
   */
  public boolean awaitBoundary(int cohort, Duration timeout) throws InterruptedException {
    if (lifecycle == null || cohort < 1) return true;
    long deadline = System.nanoTime() + timeout.toNanos();
    synchronized (this) {
      while ((processed < cohort || held == cohort) && !stopping) {
        long left = deadline - System.nanoTime();
        if (left <= 0) return false;
        TimeUnit.NANOSECONDS.timedWait(this, left);
      }
      if (cohort > passed) passed = cohort;
    }
    return true;
  }

  /**
   * 区切り h の次の反復の Φ の段を、{@link #release} まで待たせる。待たせている間だけ、その反復から効く行を phi.jsonl に足せる。
   *
   * @return 待たせたなら true。Φ の段がもう区切り h を通っていれば false
   */
  private synchronized boolean hold(int h) {
    if (passed >= h) return false;
    held = h;
    return true;
  }

  /** {@link #hold} で待たせた Φ の段を通す。 */
  private synchronized void release() {
    held = 0;
    notifyAll();
  }

  /** 要求 1 回分の作業。 */
  void pass() {
    int sealed = 0;
    try {
      sealed = lastSealed();
      if (lifecycle != null) boundaries(sealed);
      if (setup.scores()) scorePending();
      if (lifecycle != null && !stopping) {
        gate.awaitRunning();
        refits(sealed);
      }
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      LOG.warn("the GRP background was interrupted");
    } catch (Throwable failure) {
      LOG.warn("the GRP background stopped: {}", failure.toString(), failure);
    } finally {
      markProcessed(sealed);
    }
  }

  /**
   * まだ済んでいない区切りの処理を古い順に行う。最後の区切りが監視の行を書いた後、候補の関門の途中で止まっていれば、先にその関門を
   * 済ませる。再起動で学習が途中で止まって M(b − 1) が無ければ、区切り b の処理の前に学ぶ。先に学ぶのは、
   * 予定の学び直しの設定で、学び直しが止まっておらず、その区切りの処理が遅れていないときだけ。
   */
  private void boundaries(int sealed) throws IOException, InterruptedException {
    int next = PhiHistory.nextIteration(files);
    int last = lifecycle.lastBoundary();
    if (!stopping && lifecycle.gateLeft(last)) {
      gate.awaitRunning();
      boolean waits = hold(last);
      try {
        lifecycle.lateGate(last - 1, waits);
      } catch (IOException | RuntimeException failure) {
        LOG.warn("c{} gate failed: {}", last, failure.toString(), failure);
      } finally {
        release();
      }
    }
    for (int b = last + 1; b <= sealed && !stopping; b++) {
      gate.awaitRunning();
      try {
        if (b + 1 >= next && setup.selfGrp().grp().refit() == GrpRefit.SCHEDULE && !lifecycle.halted())
          GrpRefitter.refit(files, setup, b - 1, RefitCause.SCHEDULE, () -> stopping);
        if (stopping) return;
        boolean waits = hold(b);
        try {
          lifecycle.boundary(b, waits);
        } finally {
          release();
        }
      } catch (IOException | RuntimeException failure) {
        LOG.warn("c{} boundary failed: {}", b, failure.toString(), failure);
      }
      markProcessed(b);
    }
  }

  /**
   * 手動の学び直しの頼みに従い、予定の学び直しを行う。頼みごとの失敗は警告を書き、その頼みは従ったものとして移す。メモリ切れなどの
   * Error で落ちた頼みも移す（残すと、裏の作業が動くたびに同じ学び直しをやり直す）。学習の途中で取り消された頼みは、関門に通さず、
   * 移さない。
   */
  private void refits(int sealed) throws IOException, InterruptedException {
    if (lifecycle.halted()) return;
    ControlRequest request;
    while (!stopping && (request = nextRefit(sealed)) != null) {
      int slot = ((Request.GrpRefit) request.request()).slot();
      CancelWatch cancel = new CancelWatch(request);
      try {
        GrpRefitter.refit(files, setup, slot, RefitCause.MANUAL, () -> stopping || cancel.cancelled());
        if (stopping) return;
        if (!cancel.cancelled) {
          boolean waits = hold(slot + 1);
          try {
            lifecycle.lateGate(slot, waits);
          } finally {
            release();
          }
        }
      } catch (IOException | RuntimeException | Error failure) {
        LOG.warn("grp-refit request {} for slot {} failed: {}", request.number(), slot, failure.toString(), failure);
      }
      if (cancel.cancelled) {
        LOG.info("Stopped the refit of grp-refit request {} for slot {}: the request was cancelled", request.number(), slot);
      } else {
        requests.retire(request);
        LOG.info("Retired grp-refit request {} for slot {}", request.number(), slot);
      }
      gate.awaitRunning();
    }
    if (setup.selfGrp().grp().refit() == GrpRefit.SCHEDULE && sealed >= 1 && !stopping)
      GrpRefitter.refit(files, setup, sealed, RefitCause.SCHEDULE, () -> stopping);
  }

  /**
   * 次に従う学び直しの頼み。番号の最も小さい頼みのうち、時点のコホートが確定していて、取り消しの頼みが指していないもの。
   *
   * @param sealed 最新の確定済みのコホート
   * @return 無ければ null
   */
  private ControlRequest nextRefit(int sealed) throws IOException {
    List<ControlRequest> pending = requests.pending();
    Set<Integer> cancelled = new HashSet<>();
    for (ControlRequest request : pending)
      if (request.request() instanceof Request.Cancel cancel) cancelled.add(cancel.request());
    for (ControlRequest request : pending) {
      boolean due = request.request() instanceof Request.GrpRefit refit && refit.slot() <= sealed;
      if (due && !cancelled.contains(request.number())) return request;
    }
    return null;
  }

  /**
   * 従っている学び直しの頼みが取り消されたかを見る。取り消されたのは、取り消しの頼みが指しているか、安全点の確かめがもう control/done に
   * 移したとき。学習の毎歩の前に呼ばれるので、control を読むのは {@link #CANCEL_CHECK_NANOS} に 1 回だけにする。1 度取り消されたら
   * 変わらない。control が読めなければ警告を書き、取り消されていないとみなす。
   */
  private final class CancelWatch {
    private final ControlRequest request;
    private long checked = System.nanoTime() - CANCEL_CHECK_NANOS;
    private boolean cancelled;

    CancelWatch(ControlRequest request) {
      this.request = request;
    }

    boolean cancelled() {
      long now = System.nanoTime();
      if (cancelled || now - checked < CANCEL_CHECK_NANOS) return cancelled;
      checked = now;
      try {
        boolean pending = false;
        boolean targeted = false;
        for (ControlRequest other : requests.pending()) {
          pending |= other.number() == request.number();
          targeted |= other.request() instanceof Request.Cancel cancel && cancel.request() == request.number();
        }
        cancelled = targeted || !pending;
      } catch (IOException | RuntimeException failure) {
        LOG.warn("the control requests could not be read during the refit: {}", failure.toString());
      }
      return cancelled;
    }
  }

  /** 未採点のコホート（{@link PhiScorer#pending}）を古い順に採点する。コホートごとの失敗は警告を書いて次に進む。 */
  private void scorePending() throws IOException, InterruptedException {
    for (int cohort : PhiScorer.pending(files)) {
      if (stopping) return;
      gate.awaitRunning();
      try {
        PhiScorer.score(files, setup, cohort).ifPresent(line -> LOG.info("{}", line.summary()));
      } catch (IOException | RuntimeException failure) {
        LOG.warn("c{} was not scored: {}", cohort, failure.toString());
      }
    }
  }

  /** 最新の確定済みのコホート（無ければ 0）。 */
  private int lastSealed() throws IOException {
    int next = PhiHistory.nextIteration(files);
    // state.json の次のコホートが確定済みなら（収集の後、学習の前）、それが最新
    return files.sealed(next) ? next : next - 1;
  }

  private synchronized void markProcessed(int cohort) {
    if (cohort > processed) processed = cohort;
    notifyAll();
  }

  /** 停止のフラグを立て、待っている要求を捨て、実行中の作業が終わるのを待つ。 */
  @Override
  public void close() {
    if (executor == null) return;
    stopping = true;
    synchronized (this) {
      notifyAll();
    }
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

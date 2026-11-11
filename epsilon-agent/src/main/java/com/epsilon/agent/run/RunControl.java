package com.epsilon.agent.run;

import com.epsilon.agent.run.RunStatus.State;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * セーフポイントで、止まるか続けるかを決める。段とジョブを回す JVM が、セーフポイントごとに呼ぶ。
 *
 * <p>1 回の確かめでは、まず取り消しと再開の要求を片付け、次の順に見る。
 *
 * <ol>
 *   <li>このセーフポイントで効く停止と再起動の要求（番号の順で最初のもの）と、停止のシグナル
 *   <li>要求したプロセスが生きている GPU の要求（GPU の使用権を持つときだけ）
 *   <li>このセーフポイントで効く一時停止の要求。あれば {@link #CHECK_INTERVAL} ごとに、同じ順で見直しながら待つ
 * </ol>
 *
 * 止まると決めたら、理由を status.json に書き、従った要求を control/done に移す。停止の要求かシグナルで止まるときは、control/stopped
 * も置く（要求なら、移す前に置く）。
 * 一度決めた後は、何度確かめても同じ理由を返す。一時停止の要求は GPU を譲っても消えないので、起動し直した後もまた一時停止する。
 */
public final class RunControl {

  /** 一時停止している間と段の中で待っている間に、セーフポイントの確かめをやり直す間隔。 */
  public static final Duration CHECK_INTERVAL = Duration.ofSeconds(10);

  private static final Logger LOG = LoggerFactory.getLogger(RunControl.class);

  private final ControlRequests requests;
  private final GpuRequests gpu;
  private final StatusFile status;
  private final Set<String> steps;
  private final BooleanSupplier signalled;
  private final Duration interval;
  private final Set<Integer> warned = new HashSet<>();
  private final ReentrantLock lock = new ReentrantLock();
  private final Condition resumed = lock.newCondition();

  private boolean paused;
  private volatile Halt halt;

  /**
   * @param requests run の制御の要求（run を持たない呼び出しでは null）
   * @param gpu GPU の要求（GPU の使用権を持たなければ null。そのときは GPU を譲らない）
   * @param status status.json のライター（run を持たない呼び出しでは null）
   * @param steps 段の名前（段の前で待つ一時停止の要求を確かめるのに使う）
   * @param signalled 停止のシグナルが来たか
   * @param interval 確かめをやり直す間隔
   */
  RunControl(
      ControlRequests requests,
      GpuRequests gpu,
      StatusFile status,
      Set<String> steps,
      BooleanSupplier signalled,
      Duration interval) {
    this.requests = requests;
    this.gpu = gpu;
    this.status = status;
    this.steps = Set.copyOf(steps);
    this.signalled = signalled;
    this.interval = interval;
  }

  /** run を持たない呼び出し（評価を出力ディレクトリに書くだけのもの）に使う確かめを作る。停止のシグナルだけを見る。 */
  public static RunControl detached(BooleanSupplier signalled) {
    return new RunControl(null, null, null, Set.of(), signalled, CHECK_INTERVAL);
  }

  /**
   * セーフポイントで確かめる。一時停止の要求があれば、それが消えるか、止まると決まるまで待つ。
   *
   * @return 止まるなら、その理由。続けるなら空
   */
  public Optional<Halt> check(SafePoint point) throws IOException, InterruptedException {
    Halt decided = halt;
    if (decided != null) {
      return Optional.of(decided);
    }
    ControlRequest pausing = null;
    try {
      while (true) {
        List<ControlRequest> pending = requests == null ? List.of() : settle(requests.pending());
        Optional<Halt> stop = decide(point, pending);
        if (stop.isPresent()) {
          return stop;
        }
        ControlRequest pause = pause(point, pending);
        if (pause == null) {
          return Optional.empty();
        }
        if (pausing == null || pausing.number() != pause.number()) {
          pausing = pause;
          enterPause(point, pause);
        }
        Thread.sleep(interval.toMillis());
      }
    } finally {
      if (pausing != null) {
        leavePause(point);
      }
    }
  }

  /**
   * 段の中で、条件が成り立つまで待つ。待つ間は {@link #CHECK_INTERVAL} ごとにセーフポイントの確かめを行う。
   *
   * @param point 待っている間のセーフポイント（{@link SafePoint#waiting}）
   * @param ready 待つ条件。渡した時間だけ、成り立つのを待つ
   * @throws Halted 待っている間に止まると決まった（段は完了マーカーを置かずに抜ける）
   */
  public void await(SafePoint point, Ready ready) throws IOException, InterruptedException, Halted {
    while (!ready.await(interval)) {
      Optional<Halt> stop = check(point);
      if (stop.isPresent()) {
        throw new Halted(stop.get());
      }
    }
  }

  /** 段の中で待つ条件。 */
  @FunctionalInterface
  public interface Ready {
    /** @return 待つ時間のうちに条件が成り立てば true */
    boolean await(Duration timeout) throws InterruptedException;
  }

  /** 一時停止している間は待つ。裏で動く処理が、作業の区切りごとに呼ぶ。止まると決まっていれば待たずに戻る。 */
  public void awaitRunning() throws InterruptedException {
    lock.lock();
    try {
      while (paused && halt == null) {
        resumed.await();
      }
    } finally {
      lock.unlock();
    }
  }

  /** 止まると決めた理由。まだ決めていなければ空。 */
  public Optional<Halt> halt() {
    return Optional.ofNullable(halt);
  }

  void step(String step) throws IOException {
    if (status != null) {
      status.step(step);
    }
  }

  void job(String job) throws IOException {
    if (status != null) {
      status.job(job);
    }
  }

  /** 取り消しと再開の要求を片付ける。 @return 残った要求（番号の順） */
  private List<ControlRequest> settle(List<ControlRequest> pending) throws IOException {
    List<ControlRequest> remaining = new ArrayList<>(pending);
    for (ControlRequest request : pending) {
      if (!remaining.contains(request)) {
        continue;
      }
      switch (request.request()) {
        case Request.Cancel cancel -> {
          for (ControlRequest target : List.copyOf(remaining)) {
            if (target.number() == cancel.request() && target != request) {
              requests.retire(target);
              remaining.remove(target);
              LOG.info("Cancelled request {} by request {}", target.fileName(), request.number());
            }
          }
          requests.retire(request);
          remaining.remove(request);
        }
        case Request.Resume resume -> {
          for (ControlRequest target : List.copyOf(remaining)) {
            if (target.request() instanceof Request.Pause && target.number() < request.number()) {
              requests.retire(target);
              remaining.remove(target);
              LOG.info("Removed the pause request {} by request {}", target.number(), request.number());
            }
          }
          requests.retire(request);
          remaining.remove(request);
        }
        default -> {}
      }
    }
    return remaining;
  }

  /** 止めるか、起動し直すか、GPU を譲るかを決める。 */
  private Optional<Halt> decide(SafePoint point, List<ControlRequest> pending) throws IOException {
    for (ControlRequest request : pending) {
      switch (request.request()) {
        case Request.Stop stop when point.reaches(stop.at()) -> {
          String reason = "Stopped at " + point + " by request " + request.number();
          requests.markStopped(reason);
          requests.retire(request);
          return Optional.of(decided(new Halt(Ending.STOPPED, request.number(), reason), State.STOPPED));
        }
        case Request.Restart restart when point.reaches(restart.at()) -> {
          String reason = "Restarting at " + point + " by request " + request.number();
          requests.retire(request);
          return Optional.of(decided(new Halt(Ending.RESTART, request.number(), reason), State.RESTARTING));
        }
        default -> {}
      }
    }
    if (signalled.getAsBoolean()) {
      String reason = "Stopped at " + point + " by a signal";
      if (requests != null) {
        requests.markStopped(reason);
      }
      return Optional.of(decided(new Halt(Ending.STOPPED, null, reason), State.STOPPED));
    }
    if (gpu != null) {
      List<GpuRequests.Request> alive = gpu.alive();
      if (!alive.isEmpty()) {
        GpuRequests.Request first = alive.getFirst();
        String reason =
            "Yielding the GPUs at " + point + " to " + first.name() + " (pid " + first.pid() + ": " + first.command()
                + ")";
        return Optional.of(decided(new Halt(Ending.YIELD, null, reason), State.YIELDED));
      }
    }
    return Optional.empty();
  }

  /** @return このセーフポイントで効く一時停止の要求。無ければ null */
  private ControlRequest pause(SafePoint point, List<ControlRequest> pending) {
    for (ControlRequest request : pending) {
      if (request.request() instanceof Request.Pause pause) {
        if (pause.before() != null && !steps.contains(pause.before()) && warned.add(request.number())) {
          LOG.warn(
              "Pause request {} waits before {}, which is not a step of this run", request.number(), pause.before());
        }
        if (point.pauses(pause)) {
          return request;
        }
      }
    }
    return null;
  }

  private Halt decided(Halt decision, State state) throws IOException {
    lock.lock();
    try {
      halt = decision;
      resumed.signalAll();
    } finally {
      lock.unlock();
    }
    if (status != null) {
      status.state(state, decision.request(), decision.reason());
    }
    LOG.info("{}", decision.reason());
    return decision;
  }

  private void enterPause(SafePoint point, ControlRequest pause) throws IOException {
    lock.lock();
    try {
      paused = true;
    } finally {
      lock.unlock();
    }
    String reason = "Paused at " + point + " by request " + pause.number();
    if (status != null) {
      status.state(State.PAUSED, pause.number(), reason);
    }
    LOG.info("{}", reason);
  }

  private void leavePause(SafePoint point) throws IOException {
    lock.lock();
    try {
      paused = false;
      resumed.signalAll();
    } finally {
      lock.unlock();
    }
    if (halt == null) {
      if (status != null) {
        status.state(State.RUNNING, null, null);
      }
      LOG.info("Resumed at {}", point);
    }
  }
}

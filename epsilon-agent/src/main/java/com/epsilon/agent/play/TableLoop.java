package com.epsilon.agent.play;

import com.epsilon.agent.learn.StageTimer;
import com.epsilon.agent.runtime.InferenceAdmission;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 卓のコア。経路（半荘の主経路か、判断から分けた続き。{@link Run}）を判断の時点まで進め、判断を打ち手と受付の区分ごとに束ねて推論へ出す。
 * 結果は乱数の種と打ち手だけで決まり、ワーカーの数や同時に進める半荘の数に依らない。
 *
 * <p>流れ: 進行のワーカーが経路を判断の時点まで進める → 答えを待つ問い合わせを打ち手と受付の区分（{@link TableSeat#admission}）ごとに
 * まとめて打ち手へ出す → 答えがそろった経路を進行のワーカーへ戻す。数を抑えるのは新しい半荘を始めることだけで、同時に進める半荘の
 * 上限と、系列の処理が許す数（{@link TableHooks#startable}）を超えては始めない。分けた続きは作ったタスクの中で続けて進め、走っている
 * 半荘と同じく数える。ドライバ（{@link #play} を呼んだスレッド）はイベントの受け渡しと受付だけを扱い、経路を進める処理と符号化は
 * ほかのスレッドに任せる。
 *
 * <p>ワーカーのスレッドは卓を閉じるまで使い回す。{@link #play} は 1 度に 1 つのスレッドから呼ぶ。
 */
public final class TableLoop implements AutoCloseable {

  /**
   * {@link #play} を 1 回呼んだ結果。
   *
   * @param games 主経路の半荘の数
   * @param rounds 主経路の局の数
   * @param queriedDecisions 打ち手に問い合わせた判断の数（合法手が 2 つ以上。主経路と分けた続きの全てを数える）
   * @param forcedDecisions 合法手が 1 つしかない強制の判断の数
   * @param stages 卓のコアの段ごとの時間と回数
   * @param admission 受付の集計
   * @param batches 打ち手へ出したバッチの数
   * @param largestBatch 最も大きいバッチの問い合わせの数
   * @param seconds 卓を打つのにかかった時間（秒）
   */
  public record Result(
      long games,
      long rounds,
      long queriedDecisions,
      long forcedDecisions,
      Map<TableStage, StageTimer.Entry> stages,
      InferenceAdmission.Metrics admission,
      long batches,
      int largestBatch,
      double seconds) {}

  /** 進行のワーカー 1 つあたりに出すタスクの数の目安と、1 つのタスクに渡す経路の数の上限。 */
  private static final int TASKS_PER_WORKER = 2, MAX_TASK_SIZE = 64;

  private final TableSettings settings;
  private final ExecutorService workers;
  private final AtomicInteger threadIndex = new AtomicInteger();

  /** ルールごとに、進行のスレッドが 1 つずつ持つヘルパー。 */
  private final Map<Variant, ThreadLocal<Table.Tools>> tools = new EnumMap<>(Variant.class);

  /** @param settings 卓の設定（行を書くスレッドの数は読まない） */
  public TableLoop(TableSettings settings) {
    this.settings = settings;
    workers =
        Executors.newFixedThreadPool(
            settings.workers(),
            task -> {
              Thread thread = new Thread(task, "table-adv-" + threadIndex.getAndIncrement());
              thread.setDaemon(true);
              return thread;
            });
  }

  /**
   * 計画の半荘を並びの順に始めて打つ。系列の処理が打ち切りを返したら（{@link TableHooks#stopped}）、それより後の半荘は始めず、走っている
   * 半荘を打ち終えて返す。打ち手、系列の処理、観測のどれかが失敗したら卓を止め、その例外をそのまま投げる。
   *
   * @throws IllegalStateException 割り込まれたとき、または経路の進行も推論も止まったとき
   */
  public <Q extends Query> Result play(TablePlan<Q> plan, TableHooks<Q> hooks) {
    return new Loop<>(new Table<>(plan, hooks, tools(plan.variant()))).run();
  }

  @Override
  public void close() {
    workers.shutdownNow();
  }

  private synchronized ThreadLocal<Table.Tools> tools(Variant variant) {
    return tools.computeIfAbsent(variant, key -> ThreadLocal.withInitial(() -> new Table.Tools(key)));
  }

  private sealed interface Event permits Advanced, Inferred, IngressReady, Rejected {}

  /**
   * 進めた経路と、それぞれが止まった理由。
   *
   * @param spawned {@code runs} の末尾にある、このタスクで新しく作って進めた続きの数
   */
  private record Advanced<Q extends Query>(List<Run<Q>> runs, List<Run.Status> status, int spawned, Throwable failure)
      implements Event {}

  private record Inferred<Q extends Query>(List<Q> rows, Throwable failure) implements Event {}

  private record IngressReady(Throwable failure) implements Event {}

  /** 打ち手がバッチを受け取る前に失敗した（予約は返してある）。 */
  private record Rejected(Throwable failure) implements Event {}

  private final class Loop<Q extends Query> {
    final Table<Q> table;
    final BlockingQueue<Event> events = new LinkedBlockingQueue<>();
    final InferenceAdmission<TableSeat<Q>, Integer, Q> admission;
    final List<Run<Q>> ready = new ArrayList<>();
    final List<GameKey> games;

    int nextGame, activeGames, inFlightTasks, largest;
    long batches;
    boolean ingressWakeScheduled;

    Loop(Table<Q> table) {
      this.table = table;
      games = table.plan.games();
      admission =
          new InferenceAdmission<>(
              TimeUnit.MICROSECONDS.toNanos(settings.batchWaitMicros()),
              (seat, key) -> seat.maxBatch(key),
              (seat, key) -> seat.tryAcquireIngress(key),
              this::needsSupply);
    }

    /** 満たないバッチでも出すなら true。打ち手の実行先が空いているときか、推論中のバッチが 1 つも無いとき。 */
    private boolean needsSupply(TableSeat<Q> seat) {
      return seat.needsWork() || !admission.hasInFlightBatches();
    }

    Result run() {
      long start = System.nanoTime();
      try {
        while (true) {
          long busy = System.nanoTime();
          refill();
          submitReady();
          dispatch();
          scheduleIngressWake();
          if (finished()) break;
          if (inFlightTasks == 0
              && !admission.hasInFlightBatches()
              && !admission.hasQueuedRows()
              && ready.isEmpty()
              && events.isEmpty())
            throw new IllegalStateException("Table stalled: games=" + activeGames + " inFlight=" + inFlightTasks);
          long waiting = System.nanoTime();
          table.timer.add(TableStage.DRIVER_BUSY, waiting - busy);
          Event event = admission.awaitEvent(events);
          table.timer.add(TableStage.DRIVER_WAIT, System.nanoTime() - waiting);
          if (event != null) {
            handle(event);
            while ((event = events.poll()) != null) handle(event);
          }
        }
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        drain(interrupted);
        throw new IllegalStateException("Table interrupted", interrupted);
      } catch (RuntimeException | Error failure) {
        drain(failure);
        throw failure;
      } catch (Throwable failure) {
        drain(failure);
        throw new IllegalStateException("Table failed", failure);
      }
      return new Result(
          table.games.sum(),
          table.rounds.sum(),
          table.queried.sum(),
          table.forced.sum(),
          table.timer.snapshot(),
          admission.metrics(),
          batches,
          largest,
          (System.nanoTime() - start) / 1e9);
    }

    /** 全ての半荘を始めたか打ち切った後で、走っている半荘とタスクが全て終わったなら true。 */
    boolean finished() {
      return (nextGame == games.size() || table.hooks.stopped()) && activeGames == 0 && inFlightTasks == 0;
    }

    void refill() {
      int limit = Math.min(games.size(), table.hooks.startable());
      int additions = Math.min(limit - nextGame, settings.activeGames() - activeGames);
      // 系列の受け取り先のバックプレッシャーは、走っている半荘があるときだけ掛ける。
      // 全て止めるとライターの進みを知らせるイベントが来なくなり、卓が止まったと誤認する
      if (additions <= 0 || table.hooks.stopped() || (activeGames > 0 && table.hooks.backlogged())) return;
      int first = nextGame;
      nextGame += additions;
      activeGames += additions;
      int size = taskSize(additions);
      for (int from = 0; from < additions; from += size) {
        int start = first + from, end = first + Math.min(additions, from + size);
        submit(
            () -> {
              List<Run<Q>> runs = new ArrayList<>(end - start);
              for (int index = start; index < end; index++) runs.add(new Run<>(table, games.get(index)));
              return runs;
            });
      }
    }

    void submitReady() {
      if (ready.isEmpty()) return;
      List<Run<Q>> runs = List.copyOf(ready);
      ready.clear();
      int size = taskSize(runs.size());
      for (int from = 0; from < runs.size(); from += size) {
        List<Run<Q>> part = runs.subList(from, Math.min(runs.size(), from + size));
        submit(() -> part);
      }
    }

    /** 経路を作るか受け取り、次に止まるところまで進めるタスクを出す。 */
    void submit(Callable<List<Run<Q>>> source) {
      inFlightTasks++;
      try {
        workers.execute(
            () -> {
              Advanced<Q> result;
              try {
                List<Run<Q>> runs = new ArrayList<>(source.call());
                List<Run.Status> status = new ArrayList<>(runs.size());
                int spawned = 0;
                // 経路を進め、そこで分けた続きは一覧の末尾に足して同じタスクで進める（続きからさらに分けた続きも同じ）
                for (int index = 0; index < runs.size(); index++) {
                  Run<Q> run = runs.get(index);
                  status.add(run.advance());
                  if (!run.spawned().isEmpty()) {
                    runs.addAll(run.spawned());
                    spawned += run.spawned().size();
                    run.spawned().clear();
                  }
                }
                result = new Advanced<>(runs, status, spawned, null);
              } catch (Throwable failure) {
                result = new Advanced<>(null, null, 0, failure);
              }
              events.add(result);
            });
      } catch (RuntimeException | Error failure) {
        events.add(new Advanced<Q>(null, null, 0, failure));
      }
    }

    int taskSize(int count) {
      int target = settings.workers() * TASKS_PER_WORKER;
      return Math.max(1, Math.min(MAX_TASK_SIZE, 1 + (count - 1) / target));
    }

    void dispatch() {
      InferenceAdmission.Admitted<TableSeat<Q>, Integer, Q> batch;
      while ((batch = admission.admit(System.nanoTime(), inFlightTasks > 0)) != null) {
        List<Q> rows = batch.rows();
        batches++;
        largest = Math.max(largest, rows.size());
        try {
          batch
              .evaluator()
              .submit(rows, batch.ingress())
              .whenComplete((ignored, failure) -> events.add(new Inferred<>(rows, unwrap(failure))));
        } catch (RuntimeException | Error failure) {
          admission.encodingFailed(batch);
          events.add(new Rejected(failure));
        }
      }
    }

    void scheduleIngressWake() {
      if (!admission.hasQueuedRows() || !admission.waitingForIngress() || ingressWakeScheduled) return;
      ingressWakeScheduled = true;
      admission.ingressAvailable().whenComplete((ignored, failure) -> events.add(new IngressReady(unwrap(failure))));
    }

    @SuppressWarnings("unchecked")
    void handle(Event event) throws Throwable {
      switch (event) {
        case Advanced<?> advanced -> {
          inFlightTasks--;
          if (advanced.failure() != null) throw advanced.failure();
          // 新しく分けた続きも走っている半荘として数える。こうすると、同時に進める半荘の上限で続きのメモリも抑えられる
          activeGames += advanced.spawned();
          long now = System.nanoTime();
          List<Run<Q>> runs = ((Advanced<Q>) advanced).runs();
          for (int index = 0; index < runs.size(); index++) {
            Run<Q> run = runs.get(index);
            run.hooks().settle(run);
            switch (advanced.status().get(index)) {
              case NEEDS_POLICY -> {
                for (Q query : run.outstanding()) admission.add(run.seatOf(query), query.admission(), query, now);
              }
              case FINISHED -> activeGames--;
            }
          }
        }
        case Inferred<?> inferred -> {
          admission.inferenceFinished();
          if (inferred.failure() != null) throw inferred.failure();
          for (Query query : inferred.rows()) {
            Run<Q> run = (Run<Q>) query.owner();
            if (run.requestCompleted()) ready.add(run);
          }
        }
        case IngressReady wake -> {
          ingressWakeScheduled = false;
          if (wake.failure() != null) throw wake.failure();
        }
        case Rejected rejected -> throw rejected.failure();
      }
    }

    /** 失敗の後は新しいタスクを出さず、観測やモデルを借りて使っているタスクと推論が終わるのを待つ。 */
    void drain(Throwable original) {
      boolean interrupted = Thread.interrupted();
      try {
        while (inFlightTasks > 0 || admission.hasInFlightBatches()) {
          try {
            Event event = events.take();
            if (event instanceof Inferred<?> inferred) {
              admission.inferenceFinished();
              suppress(original, inferred.failure());
            } else if (event instanceof Advanced<?> advanced) {
              inFlightTasks--;
              suppress(original, advanced.failure());
            } else if (event instanceof Rejected rejected) {
              suppress(original, rejected.failure());
            }
          } catch (InterruptedException failure) {
            interrupted = true;
          }
        }
      } finally {
        if (interrupted) Thread.currentThread().interrupt();
      }
    }
  }

  private static Throwable unwrap(Throwable failure) {
    return failure instanceof CompletionException && failure.getCause() != null ? failure.getCause() : failure;
  }

  private static void suppress(Throwable original, Throwable additional) {
    if (additional != null && original != additional) original.addSuppressed(additional);
  }
}

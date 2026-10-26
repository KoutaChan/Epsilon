package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.policy.Capacity;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.RowSink;
import com.epsilon.runtime.InferenceAdmission;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * 収集のイベントループ。帳簿だけを扱い、対局の進行・符号化・GRP・保存はほかのスレッドに任せる。
 *
 * <p>流れ: 進行ワーカーが境界まで進める → 非同期の方策の問い合わせを版ごとにまとめて評価器へ → 結果がそろった run を進行ワーカーへ戻す。局末は Φ に
 * 渡し、値が返ったらその局の枠（ラベル行・根の記録）に渡す。新しい半荘の開始だけを上限で止める。写しと分岐は作った仕事の中で深さ優先に進め、
 * 走っている進行として数える。
 */
public final class CollectionDriver implements AutoCloseable {
  /**
   * @param maxActiveGames 同時に進める半荘の数
   * @param advanceWorkers 進行ワーカーのスレッド数
   */
  public record Settings(
      int maxActiveGames, int advanceWorkers, InferenceBatchingSettings batching) {}

  /** 進行ワーカーごとのタスクの数とタスクの大きさの上限。 */
  private static final int TASKS_PER_WORKER = 2, MAX_TASK_SIZE = 64;

  /** 書き終えていない行がこれを超えたら新しい半荘を始めない。 */
  private static final long ROW_BACKLOG = 500_000;

  /** 半荘ごとの席の方策。評価で使う（収集では全席が同じ版）。 */
  @FunctionalInterface
  public interface SeatPolicies {
    PolicyService[] seats(long gameId);
  }

  /** 半荘の終わり。進行ワーカーから呼ばれる。 */
  @FunctionalInterface
  public interface GameObserver {
    void gameEnded(long gameId, int[] finalScores);

    /** 判断（診断用）。state は commit の前で、進行ワーカーから並行に呼ばれる。 */
    default void decided(
        long gameId,
        com.epsilon.core.GameState state,
        com.epsilon.engine.EngineDecisionPoint point,
        com.epsilon.core.Action action) {}

    /** 局の終わり（診断用）。state は局の終わりの局面で、進行ワーカーから並行に呼ばれる。 */
    default void handEnded(
        long gameId, com.epsilon.core.GameState state, com.epsilon.engine.RoundResult result) {}

    /** 牌譜の記録先（診断用。記録しないなら null）。半荘の始めに1回呼ばれ、その半荘の進行ワーカーだけが使う。 */
    default com.epsilon.engine.GameRecorder recorder(long gameId) {
      return null;
    }
  }

  /**
   * 1コホート。半荘 ID は run 全体で一意。収集は sampling（PRISM の率）と全席の行動方策 behaviour で打ち、seating と observer は null。
   * 評価（{@link com.epsilon.prism.eval.Arena}）は {@link Sampling#OFF}・behaviour null（方策の σ）で、seating と observer を使う。
   *
   * @param iterationSeed コホートの種（局のキーと硬貨の鍵の素）
   * @param sampling 収集の率（null なら {@link Sampling#OFF}）
   * @param behaviour 全席の行動方策 b（null なら方策の σ）
   */
  public record Cohort(
      int iteration,
      long runSeed,
      long iterationSeed,
      List<Long> gameIds,
      EpsilonUtilityProfile profile,
      Sampling sampling,
      Behaviour behaviour,
      SeatPolicies seating,
      GameObserver observer) {
    public Cohort {
      if (sampling == null) sampling = Sampling.OFF;
    }
  }

  /**
   * 1 コホートの結果。
   *
   * @param rows 受け口の書いた行（{@link RowSink#stats}。収集が終わった時点の値で、書き手のキューに残る行を含まない。封印した後の値は {@link
   *     #withRows} で差し替える）
   */
  public record Result(
      CollectionMetrics.Snapshot counts,
      Map<StageTimer.Stage, StageTimer.Entry> stages,
      InferenceAdmission.Metrics admission,
      HandValue.Stats phi,
      RowSink.RowStats rows,
      long batches,
      int largestBatch,
      double seconds) {
    /** NN 推論判断/s。 */
    public double modelDecisionsPerSecond() {
      return counts.modelDecisions() / seconds;
    }

    /** 書いた行だけを差し替えた結果（封印の後の manifest の値）。 */
    public Result withRows(RowSink.RowStats written) {
      return new Result(counts, stages, admission, phi, written, batches, largestBatch, seconds);
    }

    /**
     * 収集のログの 1 行: 時間と速度、写しと分岐の判断、半荘と局、受け口へ渡した行、ラベル行（MAIN・ROLLIN）、根（写しと d = played）と床の手で
     * 作らなかった根、分岐、平均の |C ∪ {played}|、ε_C の 1 手、床 ε_b の手。
     */
    public String summary() {
      long labels = 0, copyLabels = 0;
      for (int c = 0; c < counts.labelRows().length; c++) {
        labels += counts.labelRows()[c];
        copyLabels += counts.copyLabelRows()[c];
      }
      long rowsWithValues = labels + copyLabels;
      return String.format(
          Locale.ROOT,
          "collect %.1fs %.0f model decisions/s (copy %d, fork %d) games=%d hands=%d rows=%d"
              + " labels=%d rollin=%d roots=%d (copies %d, same %d, floor skipped %d) forks=%d"
              + " |C|=%.2f explored=%d outside=%d",
          seconds,
          modelDecisionsPerSecond(),
          counts.copyDecisions(),
          counts.forkDecisions(),
          counts.games(),
          counts.hands(),
          counts.rowsWritten(),
          labels,
          copyLabels,
          counts.rootCount(),
          counts.copies(),
          counts.sameDeviations(),
          counts.floorRoots(),
          counts.forks(),
          rowsWithValues == 0 ? 0.0 : counts.measured() / (double) rowsWithValues,
          counts.explored(),
          counts.playedOutside());
    }
  }

  private final Settings settings;
  private final BlockingQueue<Event> events = new LinkedBlockingQueue<>();
  private final ExecutorService workers;
  private final AtomicInteger threadIndex = new AtomicInteger();

  public CollectionDriver(Settings settings) {
    this.settings = settings;
    workers =
        Executors.newFixedThreadPool(
            settings.advanceWorkers(),
            task -> {
              Thread thread = new Thread(task, "prism-adv-" + threadIndex.getAndIncrement());
              thread.setDaemon(true);
              return thread;
            });
  }

  /** ドライバのスレッドで実行するための投入口。Φ などの完了通知に使う。 */
  public Consumer<Runnable> executor() {
    return task -> events.add(new Task(task));
  }

  public Result collect(Cohort cohort, PolicyService policy, HandValue phi, RowSink sink) {
    return collect(cohort, policy, phi, sink, null);
  }

  /**
   * 測定器つきの収集（{@link CandidateProbe}。probe-nano）。probe が null なら {@link #collect(Cohort, PolicyService, HandValue, RowSink)}
   * と同じ。測定器を使うときはロールインと写しの率が 0 でなければならない（根も写しも作らない）。
   */
  public Result collect(
      Cohort cohort, PolicyService policy, HandValue phi, RowSink sink, CandidateProbe probe) {
    Sampling sampling = cohort.sampling();
    if (probe != null
        && (sampling.rollinRateDiscard() > 0
            || sampling.rollinRateCommit() > 0
            || sampling.rhoBranch() > 0))
      throw new IllegalArgumentException("Probed collection needs roll-in and copy rates of 0");
    return new Loop(cohort, policy, phi, sink, probe).run();
  }

  @Override
  public void close() {
    workers.shutdownNow();
  }

  private sealed interface Event {}

  /** 進めた run とその状態。spawned は runs の末尾にある、この仕事で新しく作って進めた写しと分岐の数。 */
  private record Advanced(
      List<GameRunner> runs, List<GameRunner.Status> status, int spawned, Throwable failure)
      implements Event {}

  private record Inferred(List<PolicyRequest> rows, Throwable failure) implements Event {}

  private record Task(Runnable task) implements Event {}

  private record IngressReady(Throwable failure) implements Event {}

  private final class Loop {
    final Cohort cohort;
    final HandValue phi;
    final RowSink sink;
    final CollectContext context;
    final InferenceAdmission<PolicyService, Capacity, PolicyRequest> admission;
    final List<GameRunner> ready = new ArrayList<>();
    int nextGame, activeGames, inFlightTasks, largest;
    long batches;
    boolean ingressWakeScheduled;

    Loop(
        Cohort cohort, PolicyService policy, HandValue phi, RowSink sink, CandidateProbe probe) {
      this.cohort = cohort;
      this.phi = phi;
      this.sink = sink;
      context =
          new CollectContext(
              cohort.iteration(),
              cohort.runSeed(),
              cohort.iterationSeed(),
              policy,
              sink,
              cohort.profile(),
              cohort.sampling(),
              cohort.behaviour(),
              new CollectionMetrics(),
              new StageTimer(),
              cohort.seating(),
              cohort.observer(),
              probe);
      admission =
          new InferenceAdmission<>(
              settings.batching(),
              (evaluator, capacity) -> evaluator.maxBatch(capacity),
              (evaluator, capacity) -> evaluator.tryAcquireIngress(),
              this::needsSupply);
    }

    /** 進行も推論も止まっているなら、満たないバッチでも出して対局を進める。 */
    private boolean needsSupply(PolicyService evaluator) {
      return evaluator.needsWork() || !admission.hasInFlightBatches();
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
              && phi.pending() == 0
              && ready.isEmpty()
              && events.isEmpty())
            throw new IllegalStateException(
                "Collection stalled: games=" + activeGames + " inFlight=" + inFlightTasks);
          long waiting = System.nanoTime();
          context.timer.add(StageTimer.Stage.DRIVER_BUSY, waiting - busy);
          Event event = admission.awaitEvent(events);
          context.timer.add(StageTimer.Stage.DRIVER_WAIT, System.nanoTime() - waiting);
          if (event != null) {
            handle(event);
            while ((event = events.poll()) != null) handle(event);
          }
        }
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        drain(interrupted);
        throw new IllegalStateException("Collection interrupted", interrupted);
      } catch (RuntimeException | Error failure) {
        drain(failure);
        throw failure;
      } catch (Throwable failure) {
        drain(failure);
        throw new IllegalStateException("Collection failed", failure);
      }
      CollectionMetrics.Snapshot counts = context.counts();
      // 予約した行（ラベル行と根）は全て枠がそろって受け口へ渡っているはず。欠けたら論理の誤りなので黙って捨てない
      if (counts.labels() + counts.rootCount() != counts.rowsWritten())
        throw new IllegalStateException(
            "Collection lost rows: labels="
                + counts.labels()
                + " roots="
                + counts.rootCount()
                + " written="
                + counts.rowsWritten());
      return new Result(
          counts,
          context.timer.snapshot(),
          admission.metrics(),
          phi.stats(),
          sink.stats(),
          batches,
          largest,
          (System.nanoTime() - start) / 1e9);
    }

    /** 全部の半荘が終わり、局末の値も出そろった。 */
    boolean finished() {
      return nextGame == cohort.gameIds().size()
          && activeGames == 0
          && inFlightTasks == 0
          && phi.pending() == 0;
    }

    void refill() {
      int additions =
          Math.min(cohort.gameIds().size() - nextGame, settings.maxActiveGames() - activeGames);
      // 行の背圧は走っている半荘があるときだけ掛ける（全部止めると書き手の進みを知る出来事が来ず、停止と誤認する）
      if (additions <= 0 || (activeGames > 0 && sink.backlog() >= ROW_BACKLOG)) return;
      int first = nextGame;
      nextGame += additions;
      activeGames += additions;
      int size = taskSize(additions);
      for (int from = 0; from < additions; from += size) {
        int start = first + from, end = first + Math.min(additions, from + size);
        submit(
            () -> {
              List<GameRunner> runs = new ArrayList<>(end - start);
              for (int index = start; index < end; index++)
                runs.add(new GameRunner(context, cohort.gameIds().get(index)));
              return runs;
            });
      }
    }

    void submitReady() {
      if (ready.isEmpty()) return;
      List<GameRunner> runs = List.copyOf(ready);
      ready.clear();
      int size = taskSize(runs.size());
      for (int from = 0; from < runs.size(); from += size) {
        List<GameRunner> part = runs.subList(from, Math.min(runs.size(), from + size));
        submit(() -> part);
      }
    }

    /** runs を作って（または受け取って）次の停止点まで進める。 */
    void submit(Callable<List<GameRunner>> source) {
      inFlightTasks++;
      try {
        workers.execute(
            () -> {
              Advanced result;
              try {
                List<GameRunner> runs = new ArrayList<>(source.call());
                List<GameRunner.Status> status = new ArrayList<>(runs.size());
                int spawned = 0;
                // 主経路を進め、そこで生まれた写しと分岐は同じ仕事で深さ優先に進める（写しが作った分岐も同じ）
                for (int index = 0; index < runs.size(); index++) {
                  GameRunner run = runs.get(index);
                  status.add(run.advance());
                  if (!run.spawned().isEmpty()) {
                    runs.addAll(run.spawned());
                    spawned += run.spawned().size();
                    run.spawned().clear();
                  }
                }
                result = new Advanced(runs, status, spawned, null);
              } catch (Throwable failure) {
                result = new Advanced(null, null, 0, failure);
              }
              events.add(result);
            });
      } catch (RuntimeException | Error failure) {
        events.add(new Advanced(null, null, 0, failure));
      }
    }

    int taskSize(int count) {
      int target = settings.advanceWorkers() * TASKS_PER_WORKER;
      return Math.max(1, Math.min(MAX_TASK_SIZE, 1 + (count - 1) / target));
    }

    void dispatch() {
      InferenceAdmission.AdmittedBatch<PolicyService, Capacity, PolicyRequest> batch;
      while ((batch = admission.startNextEncoding(System.nanoTime(), inFlightTasks > 0)) != null) {
        List<PolicyRequest> rows = batch.rows();
        batches++;
        largest = Math.max(largest, rows.size());
        try {
          batch
              .evaluator()
              .submit(rows, batch.ingress())
              .whenComplete((ignored, failure) -> events.add(new Inferred(rows, unwrap(failure))));
        } catch (RuntimeException | Error failure) {
          events.add(new Inferred(rows, failure));
        }
      }
    }

    void scheduleIngressWake() {
      if (!admission.hasQueuedRows() || !admission.isWaitingForIngress() || ingressWakeScheduled)
        return;
      ingressWakeScheduled = true;
      admission
          .ingressAvailable()
          .whenComplete((ignored, failure) -> events.add(new IngressReady(unwrap(failure))));
    }

    void handle(Event event) throws Throwable {
      switch (event) {
        case Advanced advanced -> {
          inFlightTasks--;
          if (advanced.failure() != null) throw advanced.failure();
          // 新しく生まれた写しと分岐は走っている進行として数える（半荘の投入の上限でそのメモリも抑える）
          activeGames += advanced.spawned();
          long now = System.nanoTime();
          for (int index = 0; index < advanced.runs().size(); index++) {
            GameRunner run = advanced.runs().get(index);
            for (HandEnd end : run.handEnds()) settle(end);
            run.handEnds().clear();
            switch (advanced.status().get(index)) {
              case NEEDS_POLICY -> {
                for (PolicyRequest request : run.outstanding())
                  admission.add(run.policyFor(request), request.capacity(), request, now);
              }
              case FINISHED -> activeGames--;
            }
          }
        }
        case Inferred inferred -> {
          admission.inferenceFinished();
          if (inferred.failure() != null) throw inferred.failure();
          for (PolicyRequest request : inferred.rows()) {
            GameRunner run = (GameRunner) request.owner();
            if (run.requestCompleted()) ready.add(run);
          }
        }
        case Task task -> task.task().run();
        case IngressReady wake -> {
          ingressWakeScheduled = false;
          if (wake.failure() != null) throw wake.failure();
        }
      }
    }

    /** 局末の値を Φ に求め（半荘末は最終効用をそのまま）、その局の行に渡す。 */
    void settle(HandEnd end) {
      if (end.values() != null) {
        accept(end.targets(), end.values());
        return;
      }
      phi.request(end.prefix(), end.scores(), values -> accept(end.targets(), values));
    }

    private static void accept(List<Slot> targets, double[] values) {
      for (Slot slot : targets) slot.target().accept(slot.action(), values);
    }

    /** 失敗後は新しい仕事を出さず、借用中の観測やモデルを使う処理が終わるのを待つ。 */
    void drain(Throwable original) {
      boolean interrupted = Thread.interrupted();
      try {
        while (inFlightTasks > 0 || admission.hasInFlightBatches()) {
          try {
            Event event = events.take();
            if (event instanceof Inferred inferred) {
              admission.inferenceFinished();
              suppress(original, inferred.failure());
            } else if (event instanceof Advanced advanced) {
              inFlightTasks--;
              suppress(original, advanced.failure());
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
    return failure instanceof CompletionException && failure.getCause() != null
        ? failure.getCause()
        : failure;
  }

  private static void suppress(Throwable original, Throwable additional) {
    if (additional != null && original != additional) original.addSuppressed(additional);
  }
}

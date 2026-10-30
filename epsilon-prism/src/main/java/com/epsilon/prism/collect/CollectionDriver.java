package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameRecorder;
import com.epsilon.engine.RoundResult;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.policy.Capacity;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.runtime.InferenceAdmission;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
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
 * <p>流れ: 進行ワーカーが境界まで進める → 非同期の方策の問い合わせを版と受付の区分（{@link PolicyService#admissionKey}）ごとにまとめて評価器へ →
 * 結果がそろった run を進行ワーカーへ戻す。局末は Φ に渡し、値が返ったらその局の枠（ラベル行・根の記録）に渡す。新しい半荘の開始だけを上限で
 * 止める。写しと分岐は作った仕事の中で深さ優先に進め、走っている進行として数える。
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
    default void decided(long gameId, GameState state, EngineDecisionPoint point, Action action) {}

    /** 局の終わり（診断用）。state は局の終わりの局面で、進行ワーカーから並行に呼ばれる。 */
    default void handEnded(long gameId, GameState state, RoundResult result) {}

    /**
     * 主経路の局末の評価の入力（診断用。GRP の監査）。半荘が続くなら次局 root の公開系列（Φ の入力）、終わったなら null。scores は精算後の
     * 点数。{@link #handEnded} の後、半荘が終わる局では {@link #gameEnded} の前に呼ばれる。
     */
    default void handValued(long gameId, float[] nextRootPrefix, int[] scores) {}

    /** 牌譜の記録先（診断用。記録しないなら null）。半荘の始めに1回呼ばれ、その半荘の進行ワーカーだけが使う。 */
    default GameRecorder recorder(long gameId) {
      return null;
    }
  }

  /**
   * 1コホート。半荘 ID は run 全体で一意。収集は sampling（PRISM の率）と全席の行動方策 behaviour で打ち、seating と observer は null。
   * 評価（{@link Arena}）は {@link Sampling#OFF}・behaviour null（方策の σ）で、seating と observer を使う。
   *
   * @param iterationSeed コホートの種（局のキーと硬貨の鍵の素）
   * @param sampling 収集の率（null なら {@link Sampling#OFF}）
   * @param behaviour 全席の行動方策 b（null なら方策の σ）
   */
  public record Cohort(
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
   * 書いた行。
   *
   * @param rows 行の数
   * @param bytes 圧縮後のバイト数
   */
  public record RowStats(long rows, long bytes) {}

  /**
   * 1 コホートの結果。
   *
   * @param rows 受け口に渡した行の数（バイトは 0。封印した後の目録の値は {@link #withRows} で差し替える）
   */
  public record Result(
      CollectionMetrics.Snapshot counts,
      Map<StageTimer.Stage, StageTimer.Entry> stages,
      InferenceAdmission.Metrics admission,
      HandValue.Stats phi,
      RowStats rows,
      long batches,
      int largestBatch,
      double seconds) {
    /** NN 推論判断/s。 */
    public double modelDecisionsPerSecond() {
      return counts.modelDecisions() / seconds;
    }

    /** 書いた行だけを差し替えた結果（封印の後の目録の値）。 */
    public Result withRows(RowStats written) {
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
              + " |C|=%.2f explored=%d outside=%d mainNonGreedy=%d",
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
          counts.playedOutside(),
          counts.mainNonGreedy());
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

  /**
   * 行を作らない対局（評価）。率は {@link Sampling#OFF} でなければならない。
   *
   * @throws IllegalArgumentException 率が {@link Sampling#OFF} でないとき
   */
  public Result collect(Cohort cohort, PolicyService policy, HandValue phi) {
    if (cohort.sampling() != Sampling.OFF)
      throw new IllegalArgumentException("A collection without a row sink plays with Sampling.OFF");
    return new Loop(cohort, policy, phi, null, null).run();
  }

  /**
   * 収集: ラベル行と根の記録を rows に書く。行の記録の失敗は収集を止める。結果の rows は書き手に渡した行の数（バイトは封印した目録の値に差し替える）。
   *
   * @param rows 行の受け口
   */
  public Result collect(Cohort cohort, PolicyService policy, HandValue phi, RowSink rows) {
    return collect(cohort, policy, phi, rows, null);
  }

  /**
   * 運の網 Ω の標本も集める収集。luck があれば、全ての経路の偶然の出来事（ツモ・槓ドラの表示）から標本を引き、その経路の局末の値を Y にして書く
   * （{@link LuckSampleSink}）。主経路で枠のない局も、標本があれば局末の値を求める。標本は進行・乱数・行を変えず、標本が失敗しても収集は続ける（失敗は
   * {@link LuckSampleSink#failure}）。
   *
   * @param rows 行の受け口
   * @param luck 運の網の標本の受け口（null なら標本を集めない）
   */
  public Result collect(
      Cohort cohort, PolicyService policy, HandValue phi, RowSink rows, LuckSampleSink luck) {
    return new Loop(cohort, policy, phi, Objects.requireNonNull(rows, "rows"), luck).run();
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
    final CollectContext context;
    final InferenceAdmission<PolicyService, Capacity, PolicyRequest> admission;
    final List<GameRunner> ready = new ArrayList<>();

    /** 局末の 3 語の書き手（行を書く収集で、ドライバのスレッドだけが使う）。 */
    final OutcomeCodec outcomes = new OutcomeCodec();

    int nextGame, activeGames, inFlightTasks, largest;
    long batches;
    boolean ingressWakeScheduled;

    Loop(Cohort cohort, PolicyService policy, HandValue phi, RowSink rows, LuckSampleSink luck) {
      this.cohort = cohort;
      this.phi = phi;
      context =
          new CollectContext(
              cohort.runSeed(),
              cohort.iterationSeed(),
              policy,
              cohort.profile(),
              cohort.sampling(),
              cohort.behaviour(),
              new CollectionMetrics(),
              new StageTimer(),
              cohort.seating(),
              cohort.observer(),
              rows,
              luck);
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
          new RowStats(context.rows == null ? 0 : context.rows.rows(), 0),
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
      if (additions <= 0 || (activeGames > 0 && backlog() >= ROW_BACKLOG)) return;
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

    /** 書き終えていない行（行を書かない評価は 0）。 */
    long backlog() {
      return context.rows == null ? 0 : context.rows.backlog();
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
                for (PolicyRequest request : run.outstanding()) {
                  PolicyService policy = run.policyFor(request);
                  admission.add(policy, policy.admissionKey(request), request, now);
                }
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

    /** 局末の値を Φ に求め（半荘末は最終効用をそのまま）、その局の行と運の網の標本に渡す。 */
    void settle(HandEnd end) {
      if (end.values() != null) {
        accept(end, end.values());
        return;
      }
      phi.request(end.prefix(), end.scores(), values -> accept(end, values));
    }

    /**
     * 局の枠ごとに、帳簿 → 局末の事実（行を書く収集）→ 値の順に渡し、運の網の標本があれば同じ値を Y にして書く。標本が書けなければ標本だけを
     * 止める。
     */
    private void accept(HandEnd end, double[] values) {
      List<Slot> targets = end.targets();
      HandEndFacts facts = end.facts();
      long outcomeNanos = 0;
      for (Slot slot : targets) {
        if (slot.ledger() != null) slot.target().acceptLedger(slot.action(), slot.ledger());
        if (facts != null) {
          long started = System.nanoTime();
          slot.target().acceptOutcome(slot.action(), facts, outcomes);
          outcomeNanos += System.nanoTime() - started;
        }
        slot.target().accept(slot.action(), values);
      }
      if (facts != null) context.timer.add(StageTimer.Stage.V2_OUTCOMES, outcomeNanos, targets.size());
      if (end.luck() != null)
        context.luck.settle(end.luck(), values, targets.isEmpty() && end.values() == null);
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

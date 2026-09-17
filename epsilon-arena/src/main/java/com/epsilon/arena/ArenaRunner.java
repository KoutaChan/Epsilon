package com.epsilon.arena;

import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.runtime.InferenceAdmission;
import com.epsilon.runtime.InferenceAdmission.AdmittedBatch;
import com.epsilon.runtime.MeasurementStatus;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * モデル系列に共通するインターフェースを使って、複数の対局とバッチ推論を進める。
 *
 * <p>各エンジンを操作するワーカーは常に一つとする。全員の応答が揃うまで行動を局面へ反映せず、その間の観測を読み取り専用で借用する。
 */
public final class ArenaRunner {
  private final List<Participant> participants;
  private final RunSettings settings;

  public ArenaRunner(List<Participant> participants, RunSettings settings) {
    if (participants.size() != 4)
      throw new IllegalArgumentException("Exactly four participants required");
    this.participants = List.copyOf(participants);
    this.settings = settings;
  }

  /** 呼び出し元スレッドへの割り込み で中止できる。発行済み推論は回収してから返る。 */
  public ArenaResult run() throws InterruptedException {
    return new Run().execute();
  }

  private final class Run {
    // 生成数は concurrentGames とバッチ数で制限される。外部要求を投入する処理は持たない。
    private final LinkedBlockingQueue<Runnable> events = new LinkedBlockingQueue<>();
    private final ExecutorService workers = Executors.newFixedThreadPool(settings.workers());
    private final InferenceAdmission<BatchedPolicy, Object, Ticket> admission =
        new InferenceAdmission<>(
            new InferenceBatchingSettings(settings.maxBatchWaitMicros()),
            BatchedPolicy::maxBatchSize,
            this::acquire,
            BatchedPolicy::needsInferenceWork);
    private final IdentityHashMap<BatchedPolicy, Metrics> metrics = new IdentityHashMap<>();
    private final List<CompletableFuture<int[]>> inFlight = new ArrayList<>();
    private final AtomicBoolean stopped = new AtomicBoolean();
    private final MatchStatistics statistics = new MatchStatistics();
    private int launched;
    private int completed;
    private int advancingGames;
    private boolean ingressWakeScheduled;
    private Throwable failure;

    ArenaResult execute() throws InterruptedException {
      long start = System.nanoTime();
      try {
        for (Participant participant : participants) {
          metrics.computeIfAbsent(
              participant.policy(),
              ignored -> new Metrics(participant.name(), participant.policy().timings()));
        }
        while (launched < Math.min(settings.games(), settings.concurrentGames())) launch();
        while (completed < settings.games()) {
          Runnable event = admission.awaitEvent(events);
          if (event != null) {
            event.run();
            while ((event = events.poll()) != null) event.run();
          }
          if (failure != null) throw new CompletionException("Arena failed", failure);
          dispatch();
        }
        return aggregate((System.nanoTime() - start) / 1e9);
      } finally {
        stopped.set(true);
        workers.shutdown();
        // cancel() は ネイティブ処理の終了を保証しないので使用しない。
        // Future が完了するまでは観測・実行環境・デバイスを生存させる。
        boolean interrupted = Thread.interrupted();
        for (CompletableFuture<int[]> batch : inFlight) {
          try {
            batch.join();
          } catch (CompletionException | CancellationException ignored) {
            /* 元の失敗を保持する。 */
          }
        }
        while (!workers.isTerminated()) {
          try {
            workers.awaitTermination(1, TimeUnit.SECONDS);
          } catch (InterruptedException ignored) {
            interrupted = true;
          }
        }
        if (interrupted) Thread.currentThread().interrupt();
      }
    }

    private void launch() {
      int index = launched++;
      advance(new Game(index));
    }

    private void advance(Game game) {
      advancingGames++;
      workers.execute(
          () -> {
            if (stopped.get()) return;
            try {
              GameStepResult step =
                  game.boundary == null
                      ? game.engine.stepHanchan()
                      : game.engine.commitDecisions(game.selections);
              game.selections.clear();
              while (!stopped.get()) {
                switch (step) {
                  case GameStepResult.AwaitingDecisions awaiting -> {
                    game.boundary = awaiting.decisions();
                    boolean allForced = true;
                    for (EngineDecisionPoint point : game.boundary) {
                      if (point.legalActions().size() != 1) {
                        allForced = false;
                        break;
                      }
                    }
                    if (!allForced) {
                      events.add(
                          () -> {
                            advancingGames--;
                            enqueue(game);
                          });
                      return;
                    }
                    for (EngineDecisionPoint point : game.boundary)
                      game.selections.add(point.id(), point.legalActions().getFirst());
                    step = game.engine.commitDecisions(game.selections);
                    game.selections.clear();
                  }
                  case GameStepResult.RoundSettled ignored -> step = game.engine.stepHanchan();
                  case GameStepResult.HanchanEnded ended -> {
                    int[] finalScores = ended.snapshotFinalScores();
                    events.add(
                        () -> {
                          advancingGames--;
                          finish(game, finalScores);
                        });
                    return;
                  }
                  case GameStepResult.RoundEnded ignored ->
                      throw new IllegalStateException("Standalone round in arena");
                }
              }
            } catch (Throwable error) {
              events.add(
                  () -> {
                    advancingGames--;
                    failure = error;
                  });
            }
          });
    }

    private void enqueue(Game game) {
      game.remaining = game.boundary.size();
      long queuedAt = System.nanoTime();
      for (int index = 0; index < game.boundary.size(); index++) {
        EngineDecisionPoint point = game.boundary.get(index);
        if (point.legalActions().size() == 1) {
          game.selected[index] = 0;
          game.remaining--;
          continue;
        }
        BatchedPolicy policy = participants.get((point.player() - game.rotation + 4) % 4).policy();
        DecisionRequest request =
            new DecisionRequest(
                point.id(),
                point.player(),
                point.kind(),
                game.engine.observation(point.player()),
                point.legalActions());
        Object key = policy.batchKey(request);
        admission.add(policy, key, new Ticket(game, index, request, queuedAt), queuedAt);
      }
    }

    private DecisionInferenceIngress.Attempt acquire(BatchedPolicy policy, Object key) {
      CompletableFuture<Void> available = new CompletableFuture<>();
      BatchedPolicy.Ingress ingress = policy.tryAcquire(key, () -> available.complete(null));
      return ingress == null
          ? DecisionInferenceIngress.Attempt.blocked(available)
          : DecisionInferenceIngress.Attempt.acquired(
              new DecisionInferenceIngress(policy, ingress, ingress::close));
    }

    private void dispatch() {
      AdmittedBatch<BatchedPolicy, Object, Ticket> batch;
      while ((batch = admission.startNextEncoding(System.nanoTime(), advancingGames > 0)) != null) {
        submit(batch);
      }
      if (admission.hasQueuedRows() && admission.isWaitingForIngress() && !ingressWakeScheduled) {
        ingressWakeScheduled = true;
        admission
            .ingressAvailable()
            .whenComplete(
                (ignored, error) -> {
                  if (!stopped.get())
                    events.add(
                        () -> {
                          ingressWakeScheduled = false;
                          if (error != null) failure = error;
                        });
                });
      }
    }

    private void submit(AdmittedBatch<BatchedPolicy, Object, Ticket> batch) {
      BatchedPolicy policy = batch.evaluator();
      List<Ticket> tickets = batch.rows();
      int count = tickets.size();
      List<DecisionRequest> requests = new ArrayList<>(count);
      Metrics metric = metrics.get(policy);
      long batchStart = System.nanoTime();
      for (Ticket ticket : tickets) {
        requests.add(ticket.request);
        metric.queueNanos += batchStart - ticket.queuedAt;
      }
      CompletableFuture<int[]> future = new CompletableFuture<>();
      inFlight.add(future);
      metric.rows += count;
      metric.batches++;
      metric.capacity += policy.maxBatchSize(batch.key());
      future.whenComplete(
          (selected, error) ->
              events.add(
                  () -> {
                    inFlight.remove(future);
                    admission.inferenceFinished();
                    metric.batchNanos += System.nanoTime() - batchStart;
                    if (error != null) failure = error;
                    else apply(tickets, selected);
                  }));
      BatchedPolicy.Ingress ingress =
          (BatchedPolicy.Ingress) batch.ingress().handoff(policy).move();
      // 予約済みバッチだけをワーカーへ渡す。観測は結果の反映まで借用する。
      workers.execute(
          () -> {
            try (ingress) {
              if (stopped.get()) {
                future.completeExceptionally(
                    new IllegalStateException("Arena stopped before submission"));
                return;
              }
              ingress
                  .submit(requests)
                  .whenComplete(
                      (selected, error) -> {
                        if (error == null) future.complete(selected);
                        else future.completeExceptionally(error);
                      });
            } catch (Throwable error) {
              future.completeExceptionally(error);
            }
          });
    }

    private void apply(List<Ticket> tickets, int[] selected) {
      if (selected.length != tickets.size())
        throw new IllegalArgumentException("Policy result row count mismatch");
      // モデル出力を全行検証した後にだけゲームの選択へ反映する。
      for (int row = 0; row < selected.length; row++) {
        if (selected[row] < 0 || selected[row] >= tickets.get(row).request.legalActions().size())
          throw new IllegalArgumentException(
              "Policy returned an invalid legal action index: " + selected[row]);
      }
      for (int row = 0; row < selected.length; row++) {
        Ticket ticket = tickets.get(row);
        Game game = ticket.game;
        game.selected[ticket.decisionIndex] = selected[row];
        if (--game.remaining == 0) {
          for (int index = 0; index < game.boundary.size(); index++) {
            EngineDecisionPoint point = game.boundary.get(index);
            game.selections.add(point.id(), point.legalActions().get(game.selected[index]));
          }
          advance(game);
        }
      }
    }

    private void finish(Game game, int[] finalScores) {
      statistics.add(game.index, finalScores);
      completed++;
      if (launched < settings.games()) launch();
    }

    private ArenaResult aggregate(double seconds) {
      List<ArenaResult.InferenceResult> inferenceResults = new ArrayList<>();
      for (var entry : metrics.entrySet()) {
        Metrics metric = entry.getValue();
        inferenceResults.add(
            new ArenaResult.InferenceResult(
                metric.name,
                metric.rows,
                metric.batches,
                metric.rows / seconds,
                metric.capacity == 0 ? 0 : (double) metric.rows / metric.capacity,
                metric.rows == 0 ? 0 : metric.queueNanos / 1000.0 / metric.rows,
                metric.batches == 0 ? 0 : metric.batchNanos / 1e6 / metric.batches,
                providerTimings(metric.before, entry.getKey().timings())));
      }
      return new ArenaResult(
          settings,
          seconds,
          settings.games() / seconds,
          statistics.players(participants.stream().map(Participant::name).toList()),
          statistics.paired(),
          inferenceResults);
    }
  }

  private final class Game {
    final int index;
    final int rotation;
    final GameEngine engine;
    final EngineSelectionBuffer selections = new EngineSelectionBuffer();
    final int[] selected = new int[3];
    List<EngineDecisionPoint> boundary;
    int remaining;

    Game(int index) {
      this.index = index;
      rotation = index % 4;
      engine = new GameEngine(settings.wallSeed(index));
    }
  }

  private record Ticket(Game game, int decisionIndex, DecisionRequest request, long queuedAt) {}

  private static final class Metrics {
    final String name;
    final BatchedPolicy.Timings before;
    long rows, batches, capacity, queueNanos, batchNanos;

    Metrics(String name, BatchedPolicy.Timings before) {
      this.name = name;
      this.before = before;
    }
  }

  private static ArenaResult.ProviderTimings providerTimings(
      BatchedPolicy.Timings before, BatchedPolicy.Timings after) {
    if (!before.available() || !after.available()) {
      return new ArenaResult.ProviderTimings(MeasurementStatus.UNAVAILABLE, 0, null, null, null);
    }
    long batches = after.batches() - before.batches();
    double divisor = batches == 0 ? 1 : 1e6 * batches;
    return new ArenaResult.ProviderTimings(
        MeasurementStatus.AVAILABLE,
        batches,
        (after.encoderQueueNanos() - before.encoderQueueNanos()) / divisor,
        (after.encodingNanos() - before.encodingNanos()) / divisor,
        (after.inferenceRoundTripNanos() - before.inferenceRoundTripNanos()) / divisor);
  }
}

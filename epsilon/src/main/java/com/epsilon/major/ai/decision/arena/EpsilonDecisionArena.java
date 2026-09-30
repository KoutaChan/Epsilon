package com.epsilon.major.ai.decision.arena;

import com.epsilon.ai.decision.DecisionBranchBudget;
import com.epsilon.ai.decision.DecisionBranchComparison;
import com.epsilon.ai.decision.EpsilonDecisionSeeds;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.ai.grp.EpsilonGrpInferenceBatcher;
import com.epsilon.ai.grp.EpsilonGrpRankPredictor;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionTrainArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.RoundPublicStateIndex;
import com.epsilon.engine.EngineCommitResult;
import com.epsilon.engine.EngineDecisionOutcome;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineDecisionSelection;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import com.epsilon.major.ai.decision.data.EpsilonDecisionCompletedGame;
import com.epsilon.major.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.major.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.major.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionRequestBatcher;
import com.epsilon.runtime.ArenaAdvanceExecutor;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.runtime.InferenceAdmission;
import com.epsilon.runtime.InferenceAdmission.AdmittedBatch;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 複数の自己対局から推論要求をまとめ、行動選択をプレイヤーへ委譲する。 */
public final class EpsilonDecisionArena {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionArena.class);
  private static final AtomicInteger COHORT_THREAD_SEQUENCE = new AtomicInteger();

  private EpsilonDecisionArena() {}

  static Metrics collectPopulationGamesWithPayloadStore(
      int games,
      long seedBase,
      int gamesInFlight,
      EpsilonDecisionEvaluator actorEvaluator,
      long actorSnapshotId,
      long[][] opponentIdsBySeat,
      DecisionSnapshotEvaluatorProvider snapshotEvaluatorProvider,
      EpsilonDecisionPlayer.RolloutConfig actorRolloutConfig,
      EpsilonDecisionPlayer.RolloutConfig opponentRolloutConfig,
      CompletedGameSink completedGameSink,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore,
      EpsilonGrpInference grpInference,
      DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession,
      GameIndexPolicy gameIndexPolicy,
      SettingsLoader config)
      throws Exception {
    if (grpInference == null) {
      return collectPopulationGamesWithRankPredictor(
          games,
          seedBase,
          gamesInFlight,
          actorEvaluator,
          actorSnapshotId,
          opponentIdsBySeat,
          snapshotEvaluatorProvider,
          actorRolloutConfig,
          opponentRolloutConfig,
          completedGameSink,
          trajectoryPayloadStore,
          null,
          adaptiveExplorationSession,
          gameIndexPolicy,
          config);
    }
    EpsilonGrpInferenceBatcher batcher =
        EpsilonGrpInferenceBatcher.create(
            grpInference,
            config
                .bind(com.epsilon.config.settings.GrpInferenceSettings.class)
                .asyncBatchingEnabled(),
            config.bind(com.epsilon.config.settings.GrpInferenceSettings.class).maxBatch(),
            config
                .bind(com.epsilon.config.settings.GrpInferenceSettings.class)
                .coalesceWaitMicros());
    try {
      return collectPopulationGamesWithRankPredictor(
          games,
          seedBase,
          gamesInFlight,
          actorEvaluator,
          actorSnapshotId,
          opponentIdsBySeat,
          snapshotEvaluatorProvider,
          actorRolloutConfig,
          opponentRolloutConfig,
          completedGameSink,
          trajectoryPayloadStore,
          batcher,
          adaptiveExplorationSession,
          gameIndexPolicy,
          config);
    } finally {
      batcher.close();
      log.info("Decision GRP boundary inference complete: {}", batcher.metrics().summary());
    }
  }

  static Metrics collectPopulationGamesWithRankPredictor(
      int games,
      long seedBase,
      int gamesInFlight,
      EpsilonDecisionEvaluator actorEvaluator,
      long actorSnapshotId,
      long[][] opponentIdsBySeat,
      DecisionSnapshotEvaluatorProvider snapshotEvaluatorProvider,
      EpsilonDecisionPlayer.RolloutConfig actorRolloutConfig,
      EpsilonDecisionPlayer.RolloutConfig opponentRolloutConfig,
      CompletedGameSink completedGameSink,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore,
      EpsilonGrpRankPredictor grpInference,
      DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession,
      GameIndexPolicy gameIndexPolicy,
      SettingsLoader config)
      throws Exception {
    if (games < 0) {
      throw new IllegalArgumentException("games must be non-negative");
    }
    if (gamesInFlight <= 0) {
      throw new IllegalArgumentException("gamesInFlight must be positive");
    }

    ArenaRun run =
        new ArenaRun(
            games,
            seedBase,
            actorEvaluator,
            actorSnapshotId,
            opponentIdsBySeat,
            snapshotEvaluatorProvider,
            actorRolloutConfig,
            opponentRolloutConfig,
            completedGameSink,
            trajectoryPayloadStore,
            grpInference,
            adaptiveExplorationSession,
            gameIndexPolicy,
            new Object(),
            new Object(),
            config);
    DecisionTrainArenaSettings arenaSettings = config.bind(DecisionTrainArenaSettings.class);
    List<CohortPlan> plans = cohortPlans(games, gamesInFlight, arenaSettings.cohorts());
    if (plans.isEmpty()) {
      return emptyMetrics();
    }
    boolean asyncInferencePipelineEnabled = arenaSettings.asyncInferencePipelineEnabled();
    long maxBatchWaitMicros = config.bind(InferenceBatchingSettings.class).maxBatchWaitMicros();
    boolean streamingSchedulerEnabled =
        asyncInferencePipelineEnabled && arenaSettings.streamingSchedulerEnabled();
    var branchSettings = config.bind(DecisionBranchComparisonSettings.class);
    if (branchSettings.enabled() && (!streamingSchedulerEnabled || grpInference == null)) {
      throw new IllegalArgumentException(
          "Branch comparison requires the streaming arena and a frozen GRP predictor");
    }
    if (asyncInferencePipelineEnabled) {
      log.info(
          "Decision arena async inference pipeline enabled: configuredCohorts={}"
              + " effectiveCohorts={} maxBatchWaitMicros={}",
          plans.size(),
          streamingSchedulerEnabled ? 1 : plans.size(),
          maxBatchWaitMicros);
    }
    if (streamingSchedulerEnabled) {
      CohortPlan streamingPlan = new CohortPlan(0, 1, Math.min(games, gamesInFlight));
      log.info(
          "Decision arena streaming scheduler enabled: capacity={} configuredCohorts={}"
              + " maxBatchWaitMicros={}",
          streamingPlan.capacity(),
          plans.size(),
          maxBatchWaitMicros);
      return new StreamingScheduler(run, streamingPlan, arenaSettings).collect();
    }
    return plans.size() == 1
        ? collectWaveCohortGames(
            run, plans.getFirst(), arenaSettings.advanceWorkers(), asyncInferencePipelineEnabled)
        : collectParallelCohorts(
            run, plans, arenaSettings.advanceWorkers(), asyncInferencePipelineEnabled);
  }

  private static Metrics collectParallelCohorts(
      ArenaRun run,
      List<CohortPlan> plans,
      int advanceWorkers,
      boolean asyncInferencePipelineEnabled)
      throws Exception {
    List<Integer> advanceWorkersByCohort = distributeAdvanceWorkers(advanceWorkers, plans.size());
    log.info(
        "Decision arena cohort pipeline enabled: cohorts={} capacities={}" + " advanceWorkers={}",
        plans.size(),
        plans.stream().map(CohortPlan::capacity).toList(),
        advanceWorkersByCohort);
    ExecutorService executor = Executors.newFixedThreadPool(plans.size(), cohortThreadFactory());
    ExecutorCompletionService<Metrics> completion = new ExecutorCompletionService<>(executor);
    ArrayList<Future<Metrics>> futures = new ArrayList<>(plans.size());
    boolean completed = false;
    try {
      for (int cohort = 0; cohort < plans.size(); cohort++) {
        CohortPlan plan = plans.get(cohort);
        int cohortAdvanceWorkers = advanceWorkersByCohort.get(cohort);
        futures.add(
            completion.submit(
                () ->
                    collectWaveCohortGames(
                        run, plan, cohortAdvanceWorkers, asyncInferencePipelineEnabled)));
      }
      Metrics aggregate = emptyMetrics();
      for (int i = 0; i < plans.size(); i++) {
        try {
          aggregate = mergeMetrics(aggregate, completion.take().get());
        } catch (ExecutionException e) {
          throw cohortFailure(e.getCause());
        }
      }
      completed = true;
      return aggregate;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw e;
    } finally {
      stopCohortWorkers(executor, futures, !completed);
    }
  }

  static List<Integer> distributeAdvanceWorkers(int configuredWorkers, int cohorts) {
    int totalWorkers = Math.max(configuredWorkers, cohorts);
    int workersPerCohort = totalWorkers / cohorts;
    int extraWorkers = totalWorkers % cohorts;
    ArrayList<Integer> distribution = new ArrayList<>(cohorts);
    for (int cohort = 0; cohort < cohorts; cohort++) {
      distribution.add(workersPerCohort + (cohort < extraWorkers ? 1 : 0));
    }
    return distribution;
  }

  private static ThreadFactory cohortThreadFactory() {
    return task -> {
      Thread thread =
          new Thread(
              task, "epsilon-decision-arena-cohort-" + COHORT_THREAD_SEQUENCE.getAndIncrement());
      thread.setDaemon(true);
      return thread;
    };
  }

  private static void stopCohortWorkers(
      ExecutorService executor, List<Future<Metrics>> futures, boolean cancel) {
    if (cancel) {
      for (Future<Metrics> future : futures) {
        future.cancel(true);
      }
      executor.shutdownNow();
    } else {
      executor.shutdown();
    }
    boolean interrupted = false;
    while (!executor.isTerminated()) {
      try {
        if (!executor.awaitTermination(1L, TimeUnit.SECONDS) && cancel) {
          // 失敗した対局群が共有の評価器/保存先を使い続けた状態で上位へ戻さない。
          // 通常完了では終了済みワーカーをそのまま終了待ちする。
          executor.shutdownNow();
        }
      } catch (InterruptedException e) {
        interrupted = true;
        executor.shutdownNow();
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private static Exception cohortFailure(Throwable cause) {
    if (cause instanceof Error error) {
      throw error;
    }
    if (cause instanceof Exception exception) {
      return exception;
    }
    return new IllegalStateException("Decision arena cohort failed", cause);
  }

  private static Throwable unwrapCompletionFailure(Throwable failure) {
    if (failure instanceof CompletionException completionException
        && completionException.getCause() != null) {
      return completionException.getCause();
    }
    return failure;
  }

  private static Metrics collectWaveCohortGames(
      ArenaRun run, CohortPlan plan, int advanceWorkers, boolean asyncInferencePipelineEnabled)
      throws Exception {
    ArenaMetrics metrics = new ArenaMetrics();
    ArrayList<GameContext> active = new ArrayList<>(Math.min(plan.capacity(), run.games()));
    long nextGame = plan.firstGameIndex();
    try (ArenaAdvanceExecutor advanceExecutor = new ArenaAdvanceExecutor(advanceWorkers)) {
      while (nextGame < run.games() || !active.isEmpty()) {
        requireCohortRunning();
        while (nextGame < run.games() && active.size() < plan.capacity()) {
          requireCohortRunning();

          active.add(run.startGame(Math.toIntExact(nextGame)));
          nextGame += plan.gameIndexStride();
        }

        ArrayList<DecisionRequest> requests = new ArrayList<>();
        for (GameContext context : active) {
          collectRequests(context, requests);
        }
        evaluatePredictionRequests(requests, metrics, asyncInferencePipelineEnabled);
        requireCohortRunning();
        selectActions(requests);

        advanceActiveGames(active, advanceExecutor, run, metrics);
      }
    }
    return metrics.snapshot(InferenceAdmission.Metrics.empty());
  }

  /**
   * 全ゲーム共通の評価器・型付き容量区分キューを保ったまま、完了した境界だけを先に進めるスケジューラー。
   *
   * <p>容量区分ごとの端数を保持して上限まで詰めたバッチを優先する。ホスト符号化は一バッチ内で並列化し、符号化完了後は先行GPU推論を待たずに
   * 次バッチを構築する。したがってバッチを厚くする入力抑制は持つが、全対局を揃えるbarrierは作らない。
   */
  private static final class StreamingScheduler {

    private final ArenaRun run;
    private final CohortPlan plan;
    private final int advanceWorkers;
    private final int advanceTasksPerWorker;
    private final int maximumAdvanceTaskSize;
    private final ArenaMetrics metrics = new ArenaMetrics();
    private final InferenceAdmission<EpsilonDecisionEvaluator, DecisionBucket, DecisionRequest>
        inferenceAdmission;
    private final BlockingQueue<SchedulerEvent> events = new LinkedBlockingQueue<>();
    private final ArrayList<GameEvaluation> readyEvaluations = new ArrayList<>();
    private final ArenaAdvanceExecutor advanceExecutor;
    private final DecisionBatchEncoder encodingExecutor;
    private long nextGame;
    private int activeGames;
    private int inFlightAdvanceTasks;
    private boolean ingressWakeScheduled;
    private final DecisionBranchBudget branchBudget;
    private final ArrayDeque<GameContext> pendingBranches;

    private StreamingScheduler(ArenaRun run, CohortPlan plan, DecisionTrainArenaSettings settings) {
      this.run = run;
      var branchSettings = run.config().bind(DecisionBranchComparisonSettings.class);
      branchBudget = branchSettings.enabled() ? new DecisionBranchBudget(branchSettings) : null;
      pendingBranches = branchBudget == null ? null : new ArrayDeque<>();
      this.plan = plan;
      this.advanceWorkers = settings.advanceWorkers();
      this.advanceTasksPerWorker = settings.advanceTasksPerWorker();
      this.maximumAdvanceTaskSize = settings.maximumAdvanceTaskSize();
      inferenceAdmission =
          new InferenceAdmission<>(
              run.config().bind(InferenceBatchingSettings.class),
              EpsilonDecisionEvaluator::preferredStreamingBatchSize,
              (evaluator, key) -> evaluator.tryAcquireInferenceIngress(),
              EpsilonDecisionEvaluator::needsInferenceWork);
      this.nextGame = plan.firstGameIndex();
      this.advanceExecutor = new ArenaAdvanceExecutor(advanceWorkers);
      this.encodingExecutor = new DecisionBatchEncoder(advanceWorkers);
    }

    private Metrics collect() throws Exception {
      try {
        startInitialGames();
        while (activeGames > 0) {
          requireCohortRunning();
          if (!readyEvaluations.isEmpty()) {
            submitReadyAdvances();
          }
          dispatchAvailableEncodings();
          enqueuePendingBranches();
          dispatchAvailableEncodings();
          scheduleIngressWakeIfBlocked();
          if (!hasInFlightWork() && !inferenceAdmission.hasQueuedRows()) {
            throw new IllegalStateException(
                "Streaming arena has active games without pending work");
          }

          SchedulerEvent event = inferenceAdmission.awaitEvent(events);
          if (event != null) {
            handleReadyEvents(event);
          }
        }
        if (branchBudget != null) {
          drainPending(null);
          log.info("Decision branch comparison complete: {}", branchBudget.summary());
        }
        return metrics.snapshot(inferenceAdmission.metrics());
      } catch (Throwable failure) {
        drainPending(failure);
        if (failure instanceof Error error) {
          throw error;
        }
        if (failure instanceof Exception exception) {
          throw exception;
        }
        throw new IllegalStateException("Decision arena streaming scheduler failed", failure);
      } finally {
        encodingExecutor.close();
        advanceExecutor.close();
      }
    }

    /** 待機せず取得できる完了イベントをまとめて適用し、次の対局進行と符号化を十分な粒度へ保つ。 */
    private void handleReadyEvents(SchedulerEvent first) throws Exception {
      handle(first);
      SchedulerEvent ready;
      while ((ready = events.poll()) != null) {
        handle(ready);
      }
    }

    private void startInitialGames() throws Exception {
      ArrayList<GameContext> contexts = new ArrayList<>(plan.capacity());
      while (nextGame < run.games() && activeGames < plan.capacity()) {
        contexts.add(startGame());
      }
      enqueue(contexts);
      dispatchAvailableEncodings();
    }

    private GameContext startGame() throws Exception {
      requireCohortRunning();

      GameContext context = run.startGame(Math.toIntExact(nextGame));
      nextGame += plan.gameIndexStride();
      activeGames++;
      return context;
    }

    private void refill(int limit, ArrayList<GameContext> contexts) throws Exception {
      int started = 0;
      while (started < limit && nextGame < run.games() && activeGames < plan.capacity()) {
        requireCohortRunning();
        contexts.add(startGame());
        started++;
      }
    }

    private void enqueue(List<GameContext> contexts) {
      if (contexts.isEmpty()) {
        return;
      }

      long enqueuedNanos = System.nanoTime();
      for (GameContext context : contexts) {
        GameEvaluation evaluation = collectEvaluation(context);
        if (branchBudget != null) {
          int rows = 0;
          for (DecisionRequest request : evaluation.requests)
            if (request.requiresEvaluation()) {
              rows++;
            }
          branchBudget.recordMainRows(rows);
        }
        enqueueEvaluation(evaluation, enqueuedNanos);
      }
    }

    private void enqueueEvaluation(GameEvaluation evaluation, long enqueuedNanos) {
      for (DecisionRequest request : evaluation.requests) {
        if (!request.requiresEvaluation()) {
          if (evaluation.predictionAttached()) {
            readyEvaluations.add(evaluation);
          }
        } else {
          inferenceAdmission.add(
              request.inferenceEvaluator,
              REQUEST_ENCODER.selectBucket(request),
              request,
              enqueuedNanos);
        }
      }
    }

    /** 満杯と期限超過のバッチは直ちに符号化し、未充填の補充バッチはCPU対局進行による後続入力の生成が終わるまで待つ。 */
    private void dispatchAvailableEncodings() {
      AdmittedBatch<EpsilonDecisionEvaluator, DecisionBucket, DecisionRequest> batch;
      while ((batch =
              inferenceAdmission.startNextEncoding(System.nanoTime(), inFlightAdvanceTasks > 0))
          != null) {
        submit(batch);
      }
    }

    /** 他スケジューラーが占有する格納枠を待つ場合もイベントループを停止させず、空き容量が次に更新されるまで待つ。 */
    private void scheduleIngressWakeIfBlocked() {
      if (!inferenceAdmission.hasQueuedRows()
          || !inferenceAdmission.isWaitingForIngress()
          || ingressWakeScheduled) {
        return;
      }
      ingressWakeScheduled = true;
      inferenceAdmission
          .ingressAvailable()
          .whenComplete(
              (ignored, failure) ->
                  events.add(new IngressAvailable(unwrapCompletionFailure(failure))));
    }

    /** 一つのホスト側バッチを並列符号化し、スケジューラーを止めずに後続行を型付きキューへ蓄積する。 */
    private void submit(
        AdmittedBatch<EpsilonDecisionEvaluator, DecisionBucket, DecisionRequest> batch) {
      recordGroupMetrics(
          metrics,
          new EpsilonDecisionRequestBatcher.BatchMetrics(
              1, batch.rows().size(), batch.rows().size()));

      try {
        encodingExecutor
            .encodeAsync(
                batch.rows(),
                batch.key(),
                (requests, builder, from, to) -> {
                  try (var session = builder.openEncoding()) {
                    for (int row = from; row < to; row++) {
                      DecisionRequest request = requests.get(row);
                      session.encodeInferenceRow(
                          row, request.context.engine, request.decision, request.boundaryContext());
                    }
                  }
                })
            .whenComplete(
                (hostBatch, failure) -> {
                  events.add(
                      new CompletedEncoding(batch, hostBatch, unwrapCompletionFailure(failure)));
                });
      } catch (RuntimeException | Error failure) {
        events.add(new CompletedEncoding(batch, null, failure));
      }
    }

    private CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> submitRetained(
        AdmittedBatch<EpsilonDecisionEvaluator, DecisionBucket, DecisionRequest> batch,
        DecisionHostBatch hostBatch) {
      return submitEvaluation(batch.evaluator(), hostBatch, batch.ingress());
    }

    private boolean hasInFlightWork() {
      return inferenceAdmission.hasInFlightBatches()
          || inFlightAdvanceTasks > 0
          || ingressWakeScheduled;
    }

    private void handle(SchedulerEvent event) throws Exception {
      if (event instanceof CompletedEncoding completed) {
        submitEncoded(completed);
      } else if (event instanceof CompletedEvaluatorChunk completed) {
        attach(completed);
      } else if (event instanceof CompletedAdvanceTask completed) {
        applyAdvanceTask(completed);
      } else if (event instanceof IngressAvailable available) {
        ingressWakeScheduled = false;
        if (available.failure() != null) {
          throw cohortFailure(available.failure());
        }
      }
    }

    private void submitEncoded(CompletedEncoding completed) throws Exception {
      if (completed.failure() != null) {
        inferenceAdmission.encodingFailed(completed.batch());
        throw cohortFailure(completed.failure());
      }
      try {
        for (int row = 0; row < completed.batch().rows().size(); row++) {
          completed.batch().rows().get(row).retainInput(completed.hostBatch(), row);
        }
      } catch (RuntimeException | Error failure) {
        inferenceAdmission.encodingFailed(completed.batch());
        throw failure;
      }

      CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> future;
      try {
        future = submitRetained(completed.batch(), completed.hostBatch());
      } catch (RuntimeException | Error failure) {
        inferenceAdmission.inferenceFinished();
        throw failure;
      }
      future.whenComplete(
          (predictions, failure) ->
              events.add(
                  new CompletedEvaluatorChunk(
                      completed.batch(), predictions, unwrapCompletionFailure(failure))));
    }

    private void attach(CompletedEvaluatorChunk completed) throws Exception {
      try {
        if (completed.failure() != null) {
          throw cohortFailure(completed.failure());
        }
        List<EpsilonDecisionInferenceServer.Prediction> predictions = completed.predictions();
        List<DecisionRequest> requests = completed.batch().rows();
        int expected = requests.size();
        if (predictions.size() != expected) {
          throw new IllegalStateException(
              "Prediction count mismatch: requested=" + expected + " got=" + predictions.size());
        }
        for (int row = 0; row < predictions.size(); row++) {
          DecisionRequest request = requests.get(row);
          request.attachPrediction(predictions.get(row));
          if (request.evaluation.predictionAttached()) {
            readyEvaluations.add(request.evaluation);
          }
        }
      } finally {
        inferenceAdmission.inferenceFinished();
      }
    }

    private void submitReadyAdvances() {
      GameEvaluation[] evaluations = readyEvaluations.toArray(GameEvaluation[]::new);
      readyEvaluations.clear();
      int targetTasks = Math.max(1, advanceWorkers * advanceTasksPerWorker);
      int taskSize =
          Math.max(
              1,
              Math.min(
                  maximumAdvanceTaskSize, (evaluations.length + targetTasks - 1) / targetTasks));
      for (int start = 0; start < evaluations.length; start += taskSize) {
        int from = start;
        int to = Math.min(start + taskSize, evaluations.length);
        CompletableFuture<List<GameAdvanceResult>> future =
            advanceExecutor.submit(() -> advanceBatch(evaluations, from, to));
        inFlightAdvanceTasks++;
        future.whenComplete(
            (results, failure) ->
                events.add(new CompletedAdvanceTask(results, unwrapCompletionFailure(failure))));
      }
    }

    private List<GameAdvanceResult> advanceBatch(GameEvaluation[] evaluations, int from, int to)
        throws Exception {
      // 同じゲームの選択順を保ち、方策標本化と対局中の行動履歴書込みも対局進行ワーカーへ渡す。

      for (int index = from; index < to; index++) {
        requireCohortRunning();
        selectActions(evaluations[index].requests);
      }

      ArrayList<GameAdvanceResult> results = new ArrayList<>(to - from);
      for (int index = from; index < to; index++) {
        requireCohortRunning();
        GameContext context = evaluations[index].context;
        if (branchBudget != null && context.branch == null) {
          context.createdBranch = forkComparison(context);
        }
        GameStepResult nextStep = commitAndAdvance(context, context.selections);
        results.add(new GameAdvanceResult(context, nextStep));
      }
      return results;
    }

    private void applyAdvanceTask(CompletedAdvanceTask completed) throws Exception {
      inFlightAdvanceTasks--;
      if (completed.failure() != null) {
        throw cohortFailure(completed.failure());
      }
      ArrayList<GameContext> nextBoundaries = new ArrayList<>(completed.results().size() * 2);
      for (GameAdvanceResult result : completed.results()) {
        requireCohortRunning();
        GameContext context = result.context();
        context.step = result.step();
        if (context.branch != null) {
          finishOrQueueBranch(context);
          continue;
        }
        if (context.createdBranch != null) {
          finishOrQueueBranch(context.createdBranch);
          context.createdBranch = null;
        }
        if (context.step instanceof GameStepResult.HanchanEnded ended) {
          finishGame(context, ended, run.completedGameSink(), run.completedGameLock(), metrics);
          activeGames--;
        } else {
          nextBoundaries.add(context);
        }
      }
      refill(completed.results().size(), nextBoundaries);
      enqueue(nextBoundaries);
    }

    private GameContext forkComparison(GameContext main) {
      EpsilonDecisionPlayer.BranchCandidate candidate = main.actorPlayer.takeBranchCandidate();
      if (candidate == null || !branchBudget.admit(main.branchCount)) {
        return null;
      }
      main.branchCount++;
      var comparison = main.actorPlayer.admitBranch(candidate, branchBudget);
      GameEngine engine = main.engine.forkAtDecisionBoundary();
      long seed = EpsilonDecisionSeeds.branch(main.seed, candidate.decisionId());
      EpsilonDecisionPlayer[] players = new EpsilonDecisionPlayer[GameState.NUM_PLAYERS];
      for (int seat = 0; seat < players.length; seat++) {
        players[seat] =
            main.players[seat].branchPlayer(EpsilonDecisionSeeds.player(seed, seat), run.config());
      }
      GameContext extra =
          new GameContext(
              main.gameIndex,
              main.gameId,
              seed,
              engine,
              null,
              players,
              players[candidate.playerSeat()]);
      extra.branch = new BranchRun(main, candidate, comparison);
      for (EngineDecisionSelection selection : main.selections) {
        extra.selections.add(
            selection.decisionId(),
            selection.decisionId() == candidate.decisionId()
                ? candidate.alternative()
                : selection.action());
      }
      extra.step = engine.commitDecisionsWithOutcomes(extra.selections).step();
      return extra;
    }

    private void finishOrQueueBranch(GameContext extra) {
      BranchRun branch = extra.branch;
      if (branch.comparison().cancelled()) {
        branchBudget.release();
        return;
      }
      RoundSettlement settlement =
          switch (extra.step) {
            case GameStepResult.RoundSettled settled -> settled.settlement();
            case GameStepResult.HanchanEnded ended -> ended.settlement();
            default -> null;
          };
      if (settlement != null) {
        if (settlement.transition() instanceof RoundTransition.NextRound
            && !branchBudget.takeRows(1)) {
          branch.comparison().cancel();
          branchBudget.release();
          return;
        }
        branch
            .comparison()
            .completeExtra(
                branch.owner().actorPlayer.branchUtility(settlement, branch.candidate()));
        branchBudget.release();
      } else {
        pendingBranches.addLast(extra);
      }
    }

    /** 通常対局の投入後、予算内の追加枝も同じキューへ渡す。混雑時も枝の進行とキャンセル済み枠の回収を止めない。 */
    private void enqueuePendingBranches() {
      if (pendingBranches == null) {
        return;
      }
      while (!pendingBranches.isEmpty()) {
        GameContext extra = pendingBranches.removeFirst();
        if (extra.branch.comparison().cancelled()) {
          branchBudget.release();
          continue;
        }
        GameEvaluation evaluation = collectEvaluation(extra);
        int rows = 0;
        for (DecisionRequest request : evaluation.requests)
          if (request.requiresEvaluation()) {
            rows++;
          }
        if (!branchBudget.takeRows(rows)) {
          extra.branch.comparison().cancel();
          branchBudget.release();
          continue;
        }
        enqueueEvaluation(evaluation, System.nanoTime());
      }
    }

    private void drainPending(Throwable primaryFailure) {
      boolean interrupted = Thread.interrupted();
      ingressWakeScheduled = false;
      while (hasInFlightWork()) {
        SchedulerEvent event;
        try {
          event = events.take();
        } catch (InterruptedException ignored) {
          interrupted = true;
          continue;
        }
        try {
          if (event instanceof CompletedEncoding completed) {
            inferenceAdmission.encodingFailed(completed.batch());
          } else if (event instanceof CompletedEvaluatorChunk completed) {
            inferenceAdmission.inferenceFinished();
          } else if (event instanceof CompletedAdvanceTask) {
            inFlightAdvanceTasks--;
          } else if (event instanceof IngressAvailable) {
            ingressWakeScheduled = false;
          }
        } catch (RuntimeException | Error cleanupFailure) {
          if (primaryFailure == null) {
            throw cleanupFailure;
          }
          primaryFailure.addSuppressed(cleanupFailure);
        }
      }
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }

  private static void advanceActiveGames(
      ArrayList<GameContext> active,
      ArenaAdvanceExecutor advanceExecutor,
      ArenaRun run,
      ArenaMetrics metrics)
      throws Exception {
    int activeSize = active.size();
    GameAdvance[] advances = new GameAdvance[activeSize];
    for (int i = 0; i < activeSize; i++) {
      GameContext context = active.get(i);
      // collectRequests が全有効なゲームに 1 件以上の判断を保証するため必ず存在する。
      advances[i] = new GameAdvance(context);
    }

    GameStepResult[] nextSteps = executeAdvances(advances, advanceExecutor);

    int survivorStart = activeSize;
    // 完了順ではなく従来と同じ有効な逆順で適用し、受け取り先の公開順を維持する。
    for (int i = activeSize - 1; i >= 0; i--) {
      requireCohortRunning();
      GameContext context = advances[i].context();
      context.step = nextSteps[i];
      if (context.step instanceof GameStepResult.HanchanEnded ended) {
        finishGame(context, ended, run.completedGameSink(), run.completedGameLock(), metrics);
      } else {
        active.set(--survivorStart, context);
      }
    }
    compactSurvivors(active, survivorStart, activeSize);
  }

  private static GameStepResult[] executeAdvances(
      GameAdvance[] advances, ArenaAdvanceExecutor advanceExecutor) throws Exception {
    GameStepResult[] nextSteps = new GameStepResult[advances.length];

    advanceExecutor.invoke(
        advances.length,
        index -> {
          GameAdvance advance = advances[index];
          GameContext context = advance.context();
          nextSteps[index] = commitAndAdvance(context, context.selections);
        });

    return nextSteps;
  }

  private static void requireCohortRunning() throws InterruptedException {
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("Decision arena cohort was cancelled");
    }
  }

  static List<CohortPlan> cohortPlans(int games, int gamesInFlight, int cohorts) {
    if (games < 0) {
      throw new IllegalArgumentException("games must be non-negative");
    }
    if (gamesInFlight <= 0) {
      throw new IllegalArgumentException("gamesInFlight must be positive");
    }
    if (cohorts <= 0) {
      throw new IllegalArgumentException("cohorts must be positive");
    }
    if (games == 0) {
      return List.of();
    }
    int lanes = Math.min(cohorts, Math.min(games, gamesInFlight));
    if (lanes == 1) {
      return List.of(new CohortPlan(0, 1, Math.min(games, gamesInFlight)));
    }
    int baseCapacity = gamesInFlight / lanes;
    int remainder = gamesInFlight % lanes;
    ArrayList<CohortPlan> plans = new ArrayList<>(lanes);
    for (int lane = 0; lane < lanes; lane++) {
      plans.add(new CohortPlan(lane, lanes, baseCapacity + (lane < remainder ? 1 : 0)));
    }
    return plans;
  }

  static Metrics mergeMetrics(Metrics left, Metrics right) {
    return new Metrics(
        Math.addExact(left.games(), right.games()),
        Math.addExact(left.inferenceBatchCount(), right.inferenceBatchCount()),
        Math.addExact(left.inferenceRequestCount(), right.inferenceRequestCount()),
        Math.max(left.maxInferenceBatchSize(), right.maxInferenceBatchSize()),
        left.batching().plus(right.batching()));
  }

  private static Metrics emptyMetrics() {
    return new Metrics(0, 0L, 0L, 0, InferenceAdmission.Metrics.empty());
  }

  private static void finishGame(
      GameContext context,
      GameStepResult.HanchanEnded ended,
      CompletedGameSink completedGameSink,
      Object completedGameLock,
      ArenaMetrics metrics)
      throws Exception {

    EpsilonDecisionCompletedGame game;

    game = context.finishTrajectory(ended.snapshotFinalScores());

    synchronized (completedGameLock) {
      completedGameSink.accept(context.gameIndex, context.seed, game);
      metrics.completedGames++;
    }
  }

  private static GameContext startGame(
      int gameIndex,
      long seedBase,
      EpsilonDecisionEvaluator actorEvaluator,
      long actorSnapshotId,
      long[][] opponentIdsBySeat,
      DecisionSnapshotEvaluatorProvider snapshotEvaluatorProvider,
      EpsilonDecisionPlayer.RolloutConfig actorRolloutConfig,
      EpsilonDecisionPlayer.RolloutConfig opponentRolloutConfig,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore,
      EpsilonGrpRankPredictor grpInference,
      DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession,
      GameIndexPolicy gameIndexPolicy,
      SettingsLoader config)
      throws Exception {
    GameAssignment assignment = gameAssignment(gameIndex, seedBase, gameIndexPolicy);
    long seed = assignment.seed();
    int actorSeat = assignment.actorSeat();
    long[] opponentIds = opponentIdsBySeat[actorSeat];
    EpsilonDecisionPlayer[] players = new EpsilonDecisionPlayer[GameState.NUM_PLAYERS];
    EpsilonDecisionPlayer actorPlayer = null;
    for (int seat = 0; seat < players.length; seat++) {
      long playerSeed = EpsilonDecisionSeeds.player(seed, seat);
      if (seat == actorSeat) {
        actorPlayer =
            EpsilonDecisionPlayer.trainingRollout(
                actorEvaluator,
                actorSnapshotId,
                playerSeed,
                actorRolloutConfig,
                trajectoryPayloadStore,
                grpInference,
                adaptiveExplorationSession,
                config);
        players[seat] = actorPlayer;
        continue;
      }
      long opponentSnapshotId = opponentIds[seat < actorSeat ? seat : seat - 1];
      players[seat] =
          EpsilonDecisionPlayer.rolloutActionOnly(
              snapshotEvaluatorProvider.evaluatorFor(opponentSnapshotId),
              opponentSnapshotId,
              playerSeed,
              opponentRolloutConfig);
    }
    GameEngine engine = new GameEngine(seed);
    GameStepResult step = advanceToBoundary(engine, engine.stepHanchan(), players);
    return new GameContext(
        Math.toIntExact(assignment.globalGameIndex()),
        assignment.gameId(),
        seed,
        engine,
        step,
        players,
        actorPlayer);
  }

  static GameAssignment gameAssignment(
      int localGameIndex, long seedBase, GameIndexPolicy gameIndexPolicy) {
    if (localGameIndex < 0) {
      throw new IllegalArgumentException("localGameIndex must be non-negative: " + localGameIndex);
    }
    long globalGameIndex = Math.addExact(gameIndexPolicy.globalGameIndexOffset(), localGameIndex);
    int actorSeat =
        gameIndexPolicy.forceFourSeatRotation()
            ? Math.floorMod(globalGameIndex, GameState.NUM_PLAYERS)
            : Math.floorMod(localGameIndex, GameState.NUM_PLAYERS);
    long seed = EpsilonDecisionSeeds.trainGame(seedBase, localGameIndex);
    return new GameAssignment(globalGameIndex, seed, seed, actorSeat);
  }

  private static GameStepResult advanceToBoundary(
      GameEngine engine, GameStepResult step, EpsilonDecisionPlayer[] players) {
    GameStepResult current = step;
    while (current instanceof GameStepResult.RoundSettled settled) {
      notifyRoundEnd(players, settled.settlement());
      current = engine.stepHanchan();
    }
    if (current instanceof GameStepResult.HanchanEnded ended) {
      notifyRoundEnd(players, ended.settlement());
    }
    return current;
  }

  private static GameStepResult commitAndAdvance(
      GameContext context, List<EngineDecisionSelection> selections) {
    EngineCommitResult committed = context.engine.commitDecisionsWithOutcomes(selections);
    notifyDecisionOutcomes(context.players, committed.decisionOutcomes());
    return context.branch == null
        ? advanceToBoundary(context.engine, committed.step(), context.players)
        : committed.step();
  }

  private static void notifyDecisionOutcomes(
      EpsilonDecisionPlayer[] players, List<EngineDecisionOutcome> outcomes) {
    for (EngineDecisionOutcome outcome : outcomes) {
      players[outcome.player()].recordDecisionOutcome(
          outcome.decisionId(), outcome.selectedAction(), outcome.learningRole());
    }
  }

  private static void notifyRoundEnd(EpsilonDecisionPlayer[] players, RoundSettlement settlement) {
    for (EpsilonDecisionPlayer player : players) {
      player.onRoundSettled(settlement);
    }
  }

  private static void collectRequests(GameContext context, List<DecisionRequest> requests) {
    requests.addAll(collectEvaluation(context).requests);
  }

  private static GameEvaluation collectEvaluation(GameContext context) {
    if (!(context.step instanceof GameStepResult.AwaitingDecisions awaiting)) {
      throw new IllegalStateException("Expected decisions, got " + context.step);
    }
    List<EngineDecisionPoint> decisions = awaiting.decisions();
    if (decisions.isEmpty()) {
      throw new IllegalStateException("Decision boundary has no decisions");
    }
    GameState state = context.engine.getState();
    context.selections.clear();
    CompletableFuture<DecisionBoundaryContext> actorBoundaryContext =
        context.actorPlayer.prepareBoundaryContextAsync(state);
    GameEvaluation evaluation = new GameEvaluation(context, decisions.size());
    for (EngineDecisionPoint decision : decisions) {
      EpsilonDecisionPlayer player = context.players[decision.player()];
      CompletableFuture<DecisionBoundaryContext> boundaryContext =
          player == context.actorPlayer
              ? actorBoundaryContext
              : CompletableFuture.completedFuture(DecisionBoundaryContext.uniform());
      evaluation.requests.add(
          new DecisionRequest(context, decision, player, evaluation, boundaryContext));
    }
    return evaluation;
  }

  private static void evaluatePredictionRequests(
      List<DecisionRequest> requests, ArenaMetrics metrics, boolean asyncInferencePipelineEnabled)
      throws Exception {
    IdentityHashMap<EpsilonDecisionEvaluator, List<DecisionRequest>> byEvaluator =
        new IdentityHashMap<>();
    for (DecisionRequest request : requests) {
      if (!request.requiresEvaluation()) {
        continue;
      }
      byEvaluator
          .computeIfAbsent(request.inferenceEvaluator, ignored -> new ArrayList<>())
          .add(request);
    }
    if (asyncInferencePipelineEnabled) {
      ArrayList<PendingEvaluatorGroup> pending = new ArrayList<>(byEvaluator.size());
      for (Map.Entry<EpsilonDecisionEvaluator, List<DecisionRequest>> entry :
          byEvaluator.entrySet()) {
        EvaluatorGroup group = new EvaluatorGroup(entry.getKey(), entry.getValue());
        PendingEvaluatorGroup submitted = submitGroup(group);
        pending.add(submitted);
        recordGroupMetrics(metrics, batchMetrics(submitted));
      }
      for (PendingEvaluatorGroup group : pending) {
        awaitGroup(group);
      }
      return;
    }
    for (Map.Entry<EpsilonDecisionEvaluator, List<DecisionRequest>> entry :
        byEvaluator.entrySet()) {
      recordGroupMetrics(
          metrics, evaluateGroup(new EvaluatorGroup(entry.getKey(), entry.getValue())));
    }
  }

  private static void recordGroupMetrics(
      ArenaMetrics metrics, EpsilonDecisionRequestBatcher.BatchMetrics result) {
    metrics.recordInferenceBatches(
        result.batchCount(), result.requestCount(), result.maximumBatchSize());
  }

  private static EpsilonDecisionRequestBatcher.BatchMetrics evaluateGroup(EvaluatorGroup group) {
    return EpsilonDecisionRequestBatcher.evaluateRequests(
        group.evaluator(),
        group.requests(),
        REQUEST_ENCODER,
        (request, prediction, hostBatch, rowIndex) -> {
          request.attachPrediction(prediction);
          request.retainInput(hostBatch, rowIndex);
        });
  }

  private static EpsilonDecisionRequestBatcher.BatchMetrics batchMetrics(
      PendingEvaluatorGroup group) {
    long requests = 0L;
    int maximumBatchSize = 0;
    for (PendingEvaluatorChunk chunk : group.chunks()) {
      int size = chunk.requests().size();
      requests += size;
      maximumBatchSize = Math.max(maximumBatchSize, size);
    }
    return new EpsilonDecisionRequestBatcher.BatchMetrics(
        group.chunks().size(), requests, maximumBatchSize);
  }

  private static final EpsilonDecisionRequestBatcher.RequestEncoder<DecisionRequest>
      REQUEST_ENCODER =
          new EpsilonDecisionRequestBatcher.RequestEncoder<>() {
            @Override
            public DecisionBucket selectBucket(DecisionRequest request) {
              return DecisionBatchBuilder.selectInferenceBucket(request.decision.legalActions());
            }

            @Override
            public void encodeRow(DecisionRequest request, DecisionBatchBuilder batchBuilder) {
              batchBuilder.addInferenceRow(
                  request.context.engine, request.decision, request.boundaryContext());
            }
          };

  private static void selectActions(List<DecisionRequest> requests) {
    for (DecisionRequest request : requests) {
      Action action =
          request.player.selectActionFromPrediction(
              request.decision.id(),
              request.decision.legalActions(),
              request.inputForSelection(),
              request.prediction);
      request.context.selections.add(request.decision.id(), action);
    }
  }

  private static <T> void compactSurvivors(ArrayList<T> active, int survivorStart, int oldSize) {
    int survivors = oldSize - survivorStart;
    for (int i = 0; i < survivors; i++) {
      active.set(i, active.get(survivorStart + i));
    }
    active.subList(survivors, oldSize).clear();
  }

  private record EvaluatorGroup(
      EpsilonDecisionEvaluator evaluator, List<DecisionRequest> requests) {}

  private record PendingEvaluatorGroup(List<PendingEvaluatorChunk> chunks) {}

  private record PendingEvaluatorChunk(
      List<DecisionRequest> requests,
      CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> future) {}

  private sealed interface SchedulerEvent
      permits CompletedEncoding, CompletedEvaluatorChunk, CompletedAdvanceTask, IngressAvailable {}

  private record CompletedEncoding(
      AdmittedBatch<EpsilonDecisionEvaluator, DecisionBucket, DecisionRequest> batch,
      DecisionHostBatch hostBatch,
      Throwable failure)
      implements SchedulerEvent {}

  private record CompletedEvaluatorChunk(
      AdmittedBatch<EpsilonDecisionEvaluator, DecisionBucket, DecisionRequest> batch,
      List<EpsilonDecisionInferenceServer.Prediction> predictions,
      Throwable failure)
      implements SchedulerEvent {}

  private record CompletedAdvanceTask(List<GameAdvanceResult> results, Throwable failure)
      implements SchedulerEvent {}

  private record IngressAvailable(Throwable failure) implements SchedulerEvent {}

  private record GameAdvance(GameContext context) {}

  private record GameAdvanceResult(GameContext context, GameStepResult step) {}

  private static final class GameEvaluation {
    private final GameContext context;
    private final ArrayList<DecisionRequest> requests;
    private int remaining;

    private GameEvaluation(GameContext context, int requestCount) {
      this.context = context;
      this.requests = new ArrayList<>(requestCount);
      this.remaining = requestCount;
    }

    private RoundPublicStateIndex publicStateContext() {
      return context.engine.getState().publicState();
    }

    private boolean predictionAttached() {
      return --remaining == 0;
    }
  }

  record CohortPlan(int firstGameIndex, int gameIndexStride, int capacity) {

    CohortPlan {
      if (firstGameIndex < 0) {
        throw new IllegalArgumentException("firstGameIndex must be non-negative");
      }
      if (gameIndexStride <= 0) {
        throw new IllegalArgumentException("gameIndexStride must be positive");
      }
      if (capacity <= 0) {
        throw new IllegalArgumentException("capacity must be positive");
      }
    }
  }

  private static PendingEvaluatorGroup submitGroup(EvaluatorGroup group) {
    LinkedHashMap<DecisionBucket, List<DecisionRequest>> byBucket = new LinkedHashMap<>();
    for (DecisionRequest request : group.requests()) {
      byBucket
          .computeIfAbsent(REQUEST_ENCODER.selectBucket(request), ignored -> new ArrayList<>())
          .add(request);
    }
    ArrayList<PendingEvaluatorChunk> chunks = new ArrayList<>();
    for (var entry : byBucket.entrySet()) {
      int chunkSize = group.evaluator().preferredBatchSize(entry.getKey());
      List<DecisionRequest> bucketRequests = entry.getValue();
      for (int start = 0; start < bucketRequests.size(); start += chunkSize) {
        int count = Math.min(chunkSize, bucketRequests.size() - start);
        List<DecisionRequest> requests = bucketRequests.subList(start, start + count);
        chunks.add(encodeAndSubmit(group.evaluator(), requests, entry.getKey()));
      }
    }
    return new PendingEvaluatorGroup(chunks);
  }

  private static PendingEvaluatorChunk encodeAndSubmit(
      EpsilonDecisionEvaluator evaluator, List<DecisionRequest> requests, DecisionBucket bucket) {
    DecisionBatchBuilder builder = DecisionBatchBuilder.inference(requests.size(), bucket);
    for (DecisionRequest request : requests) {
      REQUEST_ENCODER.encodeRow(request, builder);
    }
    DecisionHostBatch batch = builder.build();
    return retainInputsAndSubmit(evaluator, requests, batch);
  }

  private static PendingEvaluatorChunk retainInputsAndSubmit(
      EpsilonDecisionEvaluator evaluator, List<DecisionRequest> requests, DecisionHostBatch batch) {
    for (int row = 0; row < requests.size(); row++) {
      requests.get(row).retainInput(batch, row);
    }
    return new PendingEvaluatorChunk(requests, submitEvaluation(evaluator, batch));
  }

  private static CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>>
      submitEvaluation(EpsilonDecisionEvaluator evaluator, DecisionHostBatch batch) {
    return evaluator.submitBatch(batch);
  }

  private static CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>>
      submitEvaluation(
          EpsilonDecisionEvaluator evaluator,
          DecisionHostBatch batch,
          DecisionInferenceIngress ingress) {
    return evaluator.submitBatch(batch, ingress);
  }

  private static void awaitGroup(PendingEvaluatorGroup group) throws Exception {
    for (PendingEvaluatorChunk chunk : group.chunks()) {
      List<EpsilonDecisionInferenceServer.Prediction> predictions;
      try {
        predictions = chunk.future().get();
      } catch (ExecutionException e) {
        throw cohortFailure(e.getCause());
      }
      if (predictions.size() != chunk.requests().size()) {
        throw new IllegalStateException(
            "Prediction count mismatch: requested="
                + chunk.requests().size()
                + " got="
                + predictions.size());
      }
      for (int row = 0; row < predictions.size(); row++) {
        chunk.requests().get(row).attachPrediction(predictions.get(row));
      }
    }
  }

  private record ArenaRun(
      int games,
      long seedBase,
      EpsilonDecisionEvaluator actorEvaluator,
      long actorSnapshotId,
      long[][] opponentIdsBySeat,
      DecisionSnapshotEvaluatorProvider snapshotEvaluatorProvider,
      EpsilonDecisionPlayer.RolloutConfig actorRolloutConfig,
      EpsilonDecisionPlayer.RolloutConfig opponentRolloutConfig,
      CompletedGameSink completedGameSink,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore,
      EpsilonGrpRankPredictor grpInference,
      DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession,
      GameIndexPolicy gameIndexPolicy,
      Object startGameLock,
      Object completedGameLock,
      SettingsLoader config) {

    private GameContext startGame(int gameIndex) throws Exception {
      // DecisionSnapshotEvaluatorSet は評価器参照を変更可能な Map で遅延生成するため、
      // 提供元を通る対局初期化だけを直列化する。一括処理の符号化/対局進行は直列化しない。
      synchronized (startGameLock) {
        requireCohortRunning();
        return EpsilonDecisionArena.startGame(
            gameIndex,
            seedBase,
            actorEvaluator,
            actorSnapshotId,
            opponentIdsBySeat,
            snapshotEvaluatorProvider,
            actorRolloutConfig,
            opponentRolloutConfig,
            trajectoryPayloadStore,
            grpInference,
            adaptiveExplorationSession,
            gameIndexPolicy,
            config);
      }
    }
  }

  private record BranchRun(
      GameContext owner,
      EpsilonDecisionPlayer.BranchCandidate candidate,
      DecisionBranchComparison comparison) {}

  private static final class GameContext {
    private final int gameIndex;
    private final long gameId;
    private final long seed;
    private final GameEngine engine;
    private final EpsilonDecisionPlayer[] players;
    private final EpsilonDecisionPlayer actorPlayer;
    private final EngineSelectionBuffer selections = new EngineSelectionBuffer();
    private GameStepResult step;
    private BranchRun branch;
    private GameContext createdBranch;
    private int branchCount;

    private GameContext(
        int gameIndex,
        long gameId,
        long seed,
        GameEngine engine,
        GameStepResult step,
        EpsilonDecisionPlayer[] players,
        EpsilonDecisionPlayer actorPlayer) {
      this.gameIndex = gameIndex;
      this.gameId = gameId;
      this.seed = seed;
      this.engine = engine;
      this.step = step;
      this.players = players;
      this.actorPlayer = actorPlayer;
    }

    private EpsilonDecisionCompletedGame finishTrajectory(int[] finalScores) {
      return actorPlayer.flushTrajectoryDeferred(gameId, finalScores);
    }
  }

  private static final class DecisionRequest {
    private final GameContext context;
    private final EngineDecisionPoint decision;
    private final EpsilonDecisionPlayer player;
    private final EpsilonDecisionEvaluator inferenceEvaluator;
    private final GameEvaluation evaluation;
    private final CompletableFuture<DecisionBoundaryContext> boundaryContext;
    private EpsilonDecisionInferenceServer.Prediction prediction;
    private DecisionHostBatch.RowSlice selectionInput;

    private DecisionRequest(
        GameContext context,
        EngineDecisionPoint decision,
        EpsilonDecisionPlayer player,
        GameEvaluation evaluation,
        CompletableFuture<DecisionBoundaryContext> boundaryContext) {
      this.context = context;
      this.decision = decision;
      this.player = player;
      this.evaluation = evaluation;
      this.boundaryContext = boundaryContext;
      if (decision.legalActions().size() == 1 && !player.collectsTrajectory()) {
        prediction = EpsilonDecisionInferenceServer.Prediction.forcedAction();
        inferenceEvaluator = player.evaluator();
      } else {
        inferenceEvaluator = player.evaluator().resolveInferenceEvaluator(decision.legalActions());
      }
    }

    private boolean requiresEvaluation() {
      return prediction == null;
    }

    private DecisionHostBatch.RowSlice inputForSelection() {
      if (!player.collectsTrajectory()) {
        return null;
      }
      if (selectionInput == null) {
        throw new IllegalStateException("Typed input row was not retained");
      }
      return selectionInput;
    }

    private DecisionBoundaryContext boundaryContext() {
      return boundaryContext.join();
    }

    private void attachPrediction(EpsilonDecisionInferenceServer.Prediction nextPrediction) {
      prediction = nextPrediction;
    }

    private void retainInput(DecisionHostBatch batch, int row) {
      if (player.collectsTrajectory()) {
        selectionInput = batch.sliceRows(row, 1);
      }
    }
  }

  private static final class ArenaMetrics {
    private long inferenceBatchCount;
    private long inferenceRequestCount;
    private int maxInferenceBatchSize;
    private int completedGames;

    private void recordInferenceBatches(int count, long requests, int maximumBatchSize) {
      inferenceBatchCount += count;
      inferenceRequestCount += requests;
      maxInferenceBatchSize = Math.max(maxInferenceBatchSize, maximumBatchSize);
    }

    private Metrics snapshot(InferenceAdmission.Metrics batching) {
      return new Metrics(
          completedGames,
          inferenceBatchCount,
          inferenceRequestCount,
          maxInferenceBatchSize,
          batching);
    }
  }

  /** まとまり-局所的なインデックスを全実行の対局インデックスへ写像する。標準方式は従来動作を完全に維持する。 */
  public record GameIndexPolicy(long globalGameIndexOffset, boolean forceFourSeatRotation) {

    public GameIndexPolicy {
      if (globalGameIndexOffset < 0L) {
        throw new IllegalArgumentException(
            "globalGameIndexOffset must be non-negative: " + globalGameIndexOffset);
      }
    }

    public static GameIndexPolicy standard() {
      return new GameIndexPolicy(0L, false);
    }

    public static GameIndexPolicy fourSeatRotation(long publishedGames) {
      return new GameIndexPolicy(publishedGames, true);
    }
  }

  record GameAssignment(long globalGameIndex, long gameId, long seed, int actorSeat) {}

  /** 完了対局を対局インデックス順の後続処理収集処理へ渡すコールバック。 */
  @FunctionalInterface
  public interface CompletedGameSink {
    /**
     * 完了した一対局のサンプル群を受け取る。
     *
     * @param gameIndex 実行内の確定対局インデックス
     * @param seed 対局生成に使用した乱数シード
     * @param game 終局順位と境界付随情報を一度だけ持つ完了ゲーム
     * @throws Exception 後続処理保存または集約に失敗した場合
     */
    void accept(int gameIndex, long seed, EpsilonDecisionCompletedGame game) throws Exception;
  }

  /**
   * 対局実行処理の推論バッチ処理指標。
   *
   * @param games 完了対局数
   * @param inferenceBatchCount 実行した物理推論バッチ数
   * @param inferenceRequestCount バッチへ投入した要求数
   * @param maxInferenceBatchSize 観測した最大物理バッチ行数
   */
  public record Metrics(
      int games,
      long inferenceBatchCount,
      long inferenceRequestCount,
      int maxInferenceBatchSize,
      InferenceAdmission.Metrics batching) {

    /**
     * 物理順伝播一回あたりの平均要求数を返す。
     *
     * @return 順伝播未実行なら0、それ以外は要求件数 / バッチ件数
     */
    public double averageInferenceBatchSize() {
      return inferenceBatchCount == 0
          ? 0.0
          : (double) inferenceRequestCount / (double) inferenceBatchCount;
    }
  }
}

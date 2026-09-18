package com.epsilon.ai.decision.duel;

import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.runtime.ArenaAdvanceExecutor;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.runtime.InferenceAdmission;
import com.epsilon.runtime.InferenceAdmission.AdmittedBatch;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * 固定牌山で席を入れ替える対戦評価について、方策推論と対局進行を並列に実行する。
 *
 * <p>スケジューラースレッドが推論待ちの要求と対局の状態を管理する。ワーカーは完了イベントだけを返す。 同じ対局の入力符号化と {@link GameEngine}
 * の進行は重ならず、推論が完了するまで公開観測を借用できる。
 *
 * <p>要求を方策とバッチ分類キーごとにまとめ、待機期限、バッチの充填状況、実行先の空きに応じて発行する。 入力符号化と推論器への投入は系列の {@link BatchedPolicy}
 * が担当する。
 */
final class EpsilonDecisionDuelScheduler {

  private final BatchedPolicy candidateEvaluator;
  private final BatchedPolicy opponentEvaluator;
  private final int totalGames;
  private final long seedBase;
  private final long firstWallFamilyId;
  private final int gamesInFlight;
  private final int advanceWorkers;
  private final int advanceTasksPerWorker;
  private final int maximumAdvanceTaskSize;
  private final Consumer<CompletedGame> completedGameSink;
  private final MetricsBuilder metrics = new MetricsBuilder();
  private final InferenceAdmission<BatchedPolicy, Object, ActionRequest> inferenceAdmission;
  private final BlockingQueue<CompletionEvent> completedWork = new LinkedBlockingQueue<>();
  private final ArrayList<DecisionBoundary> boundariesReadyToAdvance = new ArrayList<>();
  private final AtomicInteger nextGameIndex = new AtomicInteger();
  private final ArenaAdvanceExecutor advanceExecutor;
  private int activeGameCount;
  private int inFlightAdvanceTaskCount;
  private boolean ingressWakeScheduled;

  private EpsilonDecisionDuelScheduler(
      BatchedPolicy candidateEvaluator,
      BatchedPolicy opponentEvaluator,
      int totalGames,
      long seedBase,
      long firstWallFamilyId,
      int gamesInFlight,
      Consumer<CompletedGame> completedGameSink,
      DecisionDuelArenaSettings settings,
      InferenceBatchingSettings batchingSettings) {
    this.candidateEvaluator = candidateEvaluator;
    this.opponentEvaluator = opponentEvaluator;
    this.totalGames = totalGames;
    this.seedBase = seedBase;
    this.firstWallFamilyId = firstWallFamilyId;
    this.gamesInFlight = Math.min(gamesInFlight, totalGames);
    advanceWorkers = settings.advanceWorkers();
    advanceTasksPerWorker = settings.advanceTasksPerWorker();
    maximumAdvanceTaskSize = settings.maximumAdvanceTaskSize();
    this.completedGameSink = completedGameSink;
    inferenceAdmission =
        new InferenceAdmission<>(
            batchingSettings,
            BatchedPolicy::maxBatchSize,
            EpsilonDecisionDuelScheduler::acquire,
            BatchedPolicy::needsInferenceWork);
    advanceExecutor = new ArenaAdvanceExecutor(advanceWorkers);
  }

  static Execution run(
      BatchedPolicy candidateEvaluator,
      BatchedPolicy opponentEvaluator,
      int totalGames,
      long seedBase,
      long firstWallFamilyId,
      int gamesInFlight,
      Consumer<CompletedGame> completedGameSink,
      DecisionDuelArenaSettings settings,
      InferenceBatchingSettings batchingSettings) {
    return new EpsilonDecisionDuelScheduler(
            candidateEvaluator,
            opponentEvaluator,
            totalGames,
            seedBase,
            firstWallFamilyId,
            gamesInFlight,
            completedGameSink,
            settings,
            batchingSettings)
        .execute();
  }

  private Execution execute() {
    try {
      startInitialGames();
      while (activeGameCount > 0) {
        requireRunning();
        scheduleAvailableWork();
        awaitAndApplyCompletion();
      }
      return new Execution(metrics.snapshot(inferenceAdmission.metrics()));
    } catch (Throwable failure) {
      drainPendingWork(failure);
      throw asSchedulerFailure(failure);
    } finally {
      advanceExecutor.close();
    }
  }

  /** 準備済みの境界を進め、予約できた容量へ推論バッチを発行する。 */
  private void scheduleAvailableWork() {
    submitReadyAdvances();
    dispatchAvailableInference();
    scheduleIngressWakeIfBlocked();
    if (hasInFlightWork() || inferenceAdmission.hasQueuedRows()) {
      return;
    }
    throw new IllegalStateException("Duel has active games without pending work");
  }

  private boolean hasInFlightWork() {
    return inferenceAdmission.hasInFlightBatches()
        || inFlightAdvanceTaskCount > 0
        || ingressWakeScheduled;
  }

  private void awaitAndApplyCompletion() throws Exception {
    CompletionEvent event = inferenceAdmission.awaitEvent(completedWork);
    if (event == null) return;
    applyCompletion(event);
    CompletionEvent ready;
    while ((ready = completedWork.poll()) != null) {
      applyCompletion(ready);
    }
  }

  private void applyCompletion(CompletionEvent event) throws Exception {
    if (event instanceof InferenceCompleted inference) {
      applyInferenceCompletion(inference);
    } else if (event instanceof AdvanceCompleted advance) {
      applyAdvanceCompletion(advance);
    } else if (event instanceof IngressAvailable available) {
      ingressWakeScheduled = false;
      if (available.failure() != null) {
        throw asWorkerException(available.failure());
      }
    }
  }

  private void startInitialGames() throws Exception {
    ActiveGame[] initialGames = new ActiveGame[gamesInFlight];
    advanceExecutor.invoke(initialGames.length, index -> initialGames[index] = startGame(index));
    nextGameIndex.set(initialGames.length);
    activeGameCount = initialGames.length;
    enqueueDecisionBoundaries(Arrays.asList(initialGames));
  }

  /** 完了した半荘の枠へ次の対局インデックスを一度だけ割り当てる。 */
  private ActiveGame startReplacementGame() {
    int gameIndex = nextGameIndex.getAndIncrement();
    return gameIndex < totalGames ? startGame(gameIndex) : null;
  }

  private ActiveGame startGame(int gameIndex) {

    long firstGameIndex = Math.multiplyExact(firstWallFamilyId, (long) GameState.NUM_PLAYERS);
    EpsilonDecisionDuelArena.Rotation rotation =
        EpsilonDecisionDuelArena.rotationForGame(
            Math.addExact(firstGameIndex, gameIndex), seedBase);
    BatchedPolicy[] evaluators = new BatchedPolicy[GameState.NUM_PLAYERS];
    for (int seat = 0; seat < evaluators.length; seat++) {
      evaluators[seat] = seat == rotation.candidateSeat() ? candidateEvaluator : opponentEvaluator;
    }
    GameEngine engine = new GameEngine(rotation.wallSeed());
    GameStepResult boundary = advanceToDecisionBoundary(engine, engine.stepHanchan());
    return new ActiveGame(
        rotation.wallIndex(),
        rotation.wallSeed(),
        rotation.candidateSeat(),
        engine,
        boundary,
        evaluators);
  }

  private void enqueueDecisionBoundaries(Iterable<ActiveGame> games) {
    long enqueuedNanos = System.nanoTime();
    for (ActiveGame game : games) {
      DecisionBoundary boundary = DecisionBoundary.from(game);
      for (ActionRequest request : boundary.requests) {
        if (!request.needsInference()) {
          if (boundary.markActionResolved()) {
            boundariesReadyToAdvance.add(boundary);
          }
          continue;
        }
        inferenceAdmission.add(request.evaluator, request.bucket, request, enqueuedNanos);
      }
    }
  }

  /** 満杯と期限超過のバッチは直ちに符号化し、未充填の補充バッチはCPU対局進行による後続入力の生成が終わるまで待つ。 */
  private void dispatchAvailableInference() {
    AdmittedBatch<BatchedPolicy, Object, ActionRequest> batch;
    while ((batch =
            inferenceAdmission.startNextEncoding(System.nanoTime(), inFlightAdvanceTaskCount > 0))
        != null) {
      submitInference(batch);
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
                completedWork.add(new IngressAvailable(unwrapAsyncFailure(failure))));
  }

  private static DecisionInferenceIngress.Attempt acquire(BatchedPolicy policy, Object key) {
    CompletableFuture<Void> available = new CompletableFuture<>();
    BatchedPolicy.Ingress ingress = policy.tryAcquire(key, () -> available.complete(null));
    return ingress == null
        ? DecisionInferenceIngress.Attempt.blocked(available)
        : DecisionInferenceIngress.Attempt.acquired(
            new DecisionInferenceIngress(policy, ingress, ingress::close));
  }

  private void submitInference(AdmittedBatch<BatchedPolicy, Object, ActionRequest> batch) {
    metrics.recordInferenceBatch(batch.rows().size());
    CompletableFuture<int[]> future;
    BatchedPolicy.Ingress ingress =
        (BatchedPolicy.Ingress) batch.ingress().handoff(batch.evaluator()).move();
    try (ingress) {
      List<DecisionRequest> requests = new ArrayList<>(batch.rows().size());
      for (ActionRequest request : batch.rows()) requests.add(request.input);
      future = ingress.submit(requests);
    } catch (RuntimeException | Error failure) {
      inferenceAdmission.inferenceFinished();
      throw failure;
    }
    future.whenComplete(
        (selectedSlots, failure) ->
            completedWork.add(
                new InferenceCompleted(batch, selectedSlots, unwrapAsyncFailure(failure))));
  }

  private void applyInferenceCompletion(InferenceCompleted completed) throws Exception {
    try {
      if (completed.failure() != null) {
        throw asWorkerException(completed.failure());
      }
      int[] selectedSlots = completed.selectedSlots();
      List<ActionRequest> requests = completed.batch().rows();
      if (selectedSlots.length != requests.size()) {
        throw new IllegalStateException(
            "Greedy action count mismatch: requested="
                + requests.size()
                + " got="
                + selectedSlots.length);
      }
      for (int row = 0; row < selectedSlots.length; row++) {
        ActionRequest request = requests.get(row);
        request.select(selectedSlots[row]);
        if (request.boundary.markActionResolved()) {
          boundariesReadyToAdvance.add(request.boundary);
        }
      }
    } finally {
      inferenceAdmission.inferenceFinished();
    }
  }

  private void submitReadyAdvances() {
    if (boundariesReadyToAdvance.isEmpty()) {
      return;
    }
    DecisionBoundary[] boundaries = boundariesReadyToAdvance.toArray(DecisionBoundary[]::new);
    boundariesReadyToAdvance.clear();

    int targetTaskCount = Math.max(1, advanceWorkers * advanceTasksPerWorker);
    int taskSize =
        Math.max(
            1,
            Math.min(
                maximumAdvanceTaskSize,
                (boundaries.length + targetTaskCount - 1) / targetTaskCount));
    for (int start = 0; start < boundaries.length; start += taskSize) {
      int from = start;
      int to = Math.min(start + taskSize, boundaries.length);
      CompletableFuture<List<AdvanceResult>> future =
          advanceExecutor.submit(() -> advanceBoundaries(boundaries, from, to));
      inFlightAdvanceTaskCount++;
      future.whenComplete(
          (results, failure) ->
              completedWork.add(new AdvanceCompleted(results, unwrapAsyncFailure(failure))));
    }
  }

  private List<AdvanceResult> advanceBoundaries(DecisionBoundary[] boundaries, int from, int to) {
    ArrayList<AdvanceResult> results = new ArrayList<>(to - from);
    for (int index = from; index < to; index++) {
      requireRunning();
      DecisionBoundary boundary = boundaries[index];
      ActiveGame game = boundary.game;
      EngineSelectionBuffer selections = boundary.selections;
      selections.clear();

      for (ActionRequest request : boundary.requests) {
        Action action = request.selectedAction();
        selections.add(request.decision.id(), action);
      }

      GameStepResult nextBoundary;

      nextBoundary =
          advanceToDecisionBoundary(game.engine, game.engine.commitDecisions(selections));

      ActiveGame replacement =
          nextBoundary instanceof GameStepResult.HanchanEnded ? startReplacementGame() : null;
      results.add(new AdvanceResult(game, nextBoundary, replacement));
    }
    return results;
  }

  private void applyAdvanceCompletion(AdvanceCompleted completed) throws Exception {
    inFlightAdvanceTaskCount--;
    if (completed.failure() != null) {
      throw asWorkerException(completed.failure());
    }
    ArrayList<ActiveGame> nextBoundaries = new ArrayList<>(completed.results().size());
    for (AdvanceResult result : completed.results()) {
      ActiveGame game = result.game();
      game.boundary = result.boundary();
      if (result.boundary() instanceof GameStepResult.HanchanEnded ended) {
        recordCompletedGame(game, ended);
        if (result.replacement() == null) {
          activeGameCount--;
        } else {
          nextBoundaries.add(result.replacement());
        }
      } else {
        nextBoundaries.add(game);
      }
    }
    enqueueDecisionBoundaries(nextBoundaries);
  }

  private void recordCompletedGame(ActiveGame game, GameStepResult.HanchanEnded ended) {

    completedGameSink.accept(
        new CompletedGame(
            game.wallIndex, game.wallSeed, game.candidateSeat, ended.snapshotFinalScores()));
    metrics.recordCompletedGame();
  }

  /** スケジューラー失敗後も、発行済みワーカーが返るまでイベントを回収して実行処理を安全に閉じる。 */
  private void drainPendingWork(Throwable primaryFailure) {
    boolean interrupted = Thread.interrupted();
    ingressWakeScheduled = false;
    while (hasInFlightWork()) {
      try {
        CompletionEvent event = completedWork.take();
        if (event instanceof InferenceCompleted) {
          inferenceAdmission.inferenceFinished();
        } else if (event instanceof AdvanceCompleted) {
          inFlightAdvanceTaskCount--;
        } else if (event instanceof IngressAvailable) {
          ingressWakeScheduled = false;
        }
      } catch (InterruptedException ignored) {
        interrupted = true;
      } catch (RuntimeException | Error cleanupFailure) {
        primaryFailure.addSuppressed(cleanupFailure);
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private static GameStepResult advanceToDecisionBoundary(GameEngine engine, GameStepResult step) {
    GameStepResult current = step;
    while (current instanceof GameStepResult.RoundSettled) {
      current = engine.stepHanchan();
    }
    return current;
  }

  private static void requireRunning() {
    if (Thread.currentThread().isInterrupted()) {
      throw new CompletionException(new InterruptedException("Duel scheduler was cancelled"));
    }
  }

  private static Throwable unwrapAsyncFailure(Throwable failure) {
    Throwable current = failure;
    while ((current instanceof CompletionException || current instanceof ExecutionException)
        && current.getCause() != null) {
      current = current.getCause();
    }
    return current;
  }

  private static Exception asWorkerException(Throwable failure) {
    if (failure instanceof Error error) {
      throw error;
    }
    return failure instanceof Exception exception
        ? exception
        : new IllegalStateException("Duel worker failed", failure);
  }

  private static RuntimeException asSchedulerFailure(Throwable failure) {
    if (failure instanceof Error error) {
      throw error;
    }
    if (failure instanceof RuntimeException runtime) {
      return runtime;
    }
    if (failure instanceof InterruptedException) {
      Thread.currentThread().interrupt();
    }
    return new IllegalStateException("Decision duel scheduler failed", failure);
  }

  record Execution(Metrics metrics) {}

  record CompletedGame(long wallIndex, long wallSeed, int candidateSeat, int[] finalScores) {}

  record Metrics(
      int completedGames,
      long inferenceBatches,
      long inferenceRequests,
      int maximumInferenceBatch,
      InferenceAdmission.Metrics batching) {}

  private sealed interface CompletionEvent
      permits InferenceCompleted, AdvanceCompleted, IngressAvailable {}

  private record InferenceCompleted(
      AdmittedBatch<BatchedPolicy, Object, ActionRequest> batch,
      int[] selectedSlots,
      Throwable failure)
      implements CompletionEvent {}

  private record AdvanceCompleted(List<AdvanceResult> results, Throwable failure)
      implements CompletionEvent {}

  private record IngressAvailable(Throwable failure) implements CompletionEvent {}

  private record AdvanceResult(ActiveGame game, GameStepResult boundary, ActiveGame replacement) {}

  /** 一つの半荘と、現在停止しているエンジン境界を保持する。 */
  private static final class ActiveGame {

    private final long wallIndex;
    private final long wallSeed;
    private final int candidateSeat;
    private final GameEngine engine;
    private final BatchedPolicy[] evaluators;
    private GameStepResult boundary;

    private ActiveGame(
        long wallIndex,
        long wallSeed,
        int candidateSeat,
        GameEngine engine,
        GameStepResult boundary,
        BatchedPolicy[] evaluators) {
      this.wallIndex = wallIndex;
      this.wallSeed = wallSeed;
      this.candidateSeat = candidateSeat;
      this.engine = engine;
      this.boundary = boundary;
      this.evaluators = evaluators;
    }
  }

  /** 同じエンジン境界で同時に解決すべき全プレイヤーの行動要求。 */
  private static final class DecisionBoundary {

    private final ActiveGame game;
    private final ArrayList<ActionRequest> requests;
    private final EngineSelectionBuffer selections = new EngineSelectionBuffer();
    private int unresolvedActionCount;

    private DecisionBoundary(ActiveGame game, int actionCount) {
      this.game = game;
      requests = new ArrayList<>(actionCount);
      unresolvedActionCount = actionCount;
    }

    private static DecisionBoundary from(ActiveGame game) {
      if (!(game.boundary instanceof GameStepResult.AwaitingDecisions awaiting)) {
        throw new IllegalStateException("Expected decisions, got " + game.boundary);
      }
      DecisionBoundary boundary = new DecisionBoundary(game, awaiting.decisions().size());
      for (EngineDecisionPoint decision : awaiting.decisions()) {
        boundary.requests.add(
            new ActionRequest(boundary, decision, game.evaluators[decision.player()]));
      }
      return boundary;
    }

    private boolean markActionResolved() {
      return --unresolvedActionCount == 0;
    }
  }

  /** 合法手集合と、その集合を評価する評価器を結び付けた一行の推論要求。 */
  private static final class ActionRequest {

    private final DecisionBoundary boundary;
    private final EngineDecisionPoint decision;
    private final BatchedPolicy evaluator;
    private final Object bucket;
    private final DecisionRequest input;
    private int selectedActionSlot = -1;

    private ActionRequest(
        DecisionBoundary boundary, EngineDecisionPoint decision, BatchedPolicy evaluator) {
      this.boundary = boundary;
      this.decision = decision;
      this.evaluator = evaluator;
      if (decision.legalActions().size() == 1) {
        selectedActionSlot = 0;
        input = null;
        bucket = null;
      } else {
        input =
            new DecisionRequest(
                decision.id(),
                decision.player(),
                decision.kind(),
                boundary.game.engine.observation(decision.player()),
                decision.legalActions());
        bucket = evaluator.batchKey(input);
      }
    }

    private boolean needsInference() {
      return selectedActionSlot < 0;
    }

    private void select(int actionSlot) {
      selectedActionSlot = actionSlot;
    }

    private Action selectedAction() {
      return decision.legalActions().get(selectedActionSlot);
    }
  }

  private static final class MetricsBuilder {

    private long inferenceBatches;
    private long inferenceRequests;
    private int maximumInferenceBatch;
    private int completedGames;

    private void recordInferenceBatch(int rows) {
      inferenceBatches++;
      inferenceRequests += rows;
      maximumInferenceBatch = Math.max(maximumInferenceBatch, rows);
    }

    private void recordCompletedGame() {
      completedGames++;
    }

    private Metrics snapshot(InferenceAdmission.Metrics batching) {
      return new Metrics(
          completedGames, inferenceBatches, inferenceRequests, maximumInferenceBatch, batching);
    }
  }
}

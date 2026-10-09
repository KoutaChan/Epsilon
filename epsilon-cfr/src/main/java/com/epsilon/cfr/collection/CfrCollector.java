package com.epsilon.cfr.collection;

import com.epsilon.ai.decision.DecisionBranchBudget;
import com.epsilon.ai.decision.DecisionBranchGate;
import com.epsilon.ai.decision.EpsilonDecisionSeeds;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.cfr.data.CfrBranchOutcome;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.ScoreRanking;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.runtime.ArenaAdvanceExecutor;
import com.epsilon.runtime.InferenceAdmission;
import com.epsilon.util.SeedMixer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletionException;
import java.util.concurrent.LinkedBlockingQueue;

/** 全席を対象とするアウトカムサンプリング。判断待ちの複数対局をまとめて推論する。乱数シードと選択行動は対局・判断の識別子で決まり、処理の完了順やバッチサイズには依存しない。 */
public final class CfrCollector {
  public enum Mode {
    HAND_GRP,
    MATCH_MC
  }

  public record Config(
      Mode mode,
      double epsilon,
      int maxBatchSize,
      int maxConcurrentGames,
      long seed,
      boolean centerUtility,
      boolean retainDecisions,
      int multiTransitionMaxBatch,
      DecisionDuelArenaSettings arena,
      InferenceBatchingSettings batching,
      EpsilonUtilityProfile utilityProfile) {
    public Config {
      Objects.requireNonNull(mode, "mode");
      if (utilityProfile == null || !utilityProfile.rankBased())
        throw new IllegalArgumentException("CFR collection requires a rank utility profile");
      SettingsLoader.validate(Objects.requireNonNull(arena, "arena"));
      Objects.requireNonNull(batching, "batching");
      if (!Double.isFinite(epsilon)
          || epsilon < 0
          || epsilon > 1
          || maxBatchSize < 1
          || multiTransitionMaxBatch < 1
          || maxConcurrentGames < 1) {
        throw new IllegalArgumentException("Invalid collection configuration");
      }
    }
  }

  /** 半荘内の教師経路。HAND_GRPでは局ごとに返し、最後の経路だけに半荘の最終順位を付ける。 */
  public record CollectedEpisode(
      CfrEpisode episode,
      int trajectoryIndex,
      boolean gameEnded,
      List<RootState> observedRoots,
      int[] finalRanks) {
    public CollectedEpisode {
      observedRoots = List.copyOf(observedRoots);
      finalRanks = finalRanks == null ? null : finalRanks.clone();
    }

    @Override
    public int[] finalRanks() {
      return finalRanks == null ? null : finalRanks.clone();
    }
  }

  /** 主経路の半荘数・実局数・教師経路数を分けて記録し、枝探索の仕事量も別に数える。 */
  public record CollectionStats(
      long games,
      long hands,
      long trajectories,
      long decisions,
      long inferenceBatches,
      int largestBatch,
      double elapsedSeconds,
      long branchComparisons,
      long branchEpisodes,
      long branchInferenceRows) {}

  private final Config config;
  private final DecisionBranchComparisonSettings branchSettings;
  private final double minimumUtility;
  private final double maximumUtility;

  public CfrCollector(Config config, DecisionBranchComparisonSettings branchSettings) {
    this.config = Objects.requireNonNull(config);
    this.branchSettings = Objects.requireNonNull(branchSettings);
    double minimum = Double.POSITIVE_INFINITY, maximum = Double.NEGATIVE_INFINITY;
    for (double utility : config.utilityProfile().rankUtilityDouble()) {
      minimum = Math.min(minimum, utility);
      maximum = Math.max(maximum, utility);
    }
    minimumUtility = minimum;
    maximumUtility = maximum;
    if (branchSettings.enabled() && !config.retainDecisions())
      throw new IllegalArgumentException("Branch comparison requires retained training decisions");
  }

  /** 確定した経路を空きのある保存先へ渡しながら、他の半荘と推論を進める。受け付け待ちの経路を持つ半荘も同時進行数に含め、保存されていない局のデータが増え続けるのを防ぐ。 */
  public CollectionStats collect(
      List<Long> gameIds, CfrPolicy policy, BoundaryValue boundary, CfrEpisodeSink completed) {
    Objects.requireNonNull(policy, "policy");
    Objects.requireNonNull(completed, "completed");
    if (config.mode() == Mode.HAND_GRP) {
      Objects.requireNonNull(boundary, "HAND_GRP requires a frozen boundary value");
    }
    HashSet<Long> unique = new HashSet<>();
    for (Long id : gameIds) {
      if (id == null || id < 0 || !unique.add(id)) {
        throw new IllegalArgumentException("Planned game IDs must be distinct and nonnegative");
      }
    }
    return new Scheduler(gameIds, policy, boundary, completed).collect();
  }

  /** 推論、対局進行、次の入力生成を対局ごとの完了イベントで接続する。 */
  private final class Scheduler {
    private final List<Long> gameIds;
    private final CfrPolicy policy;
    private final BoundaryValue boundary;
    private final RootState initialRoot = RootState.initial();
    private double[] initialValue;
    private final CfrEpisodeSink completed;
    private final BlockingQueue<Event> events = new LinkedBlockingQueue<>();
    private final List<Run> readyAdvances = new ArrayList<>();
    private final List<Run> completedRuns = new ArrayList<>();
    private final List<Run> waitingForBranches = new ArrayList<>();
    private final ArrayDeque<CompletedTrajectory> pendingCompletions = new ArrayDeque<>();
    private final DecisionBranchBudget branchBudget;
    private final InferenceAdmission<CfrPolicy, InferenceCapacity, Pending> admission;
    private final int advanceWorkers;
    private ArenaAdvanceExecutor advanceExecutor;
    private int next, activeMain, inFlightAdvances, largest;
    private boolean ingressWakeScheduled;
    private boolean completionWakeScheduled;
    private long finished, decisionCount, batchCount, branchEpisodes, branchRows, branchComparisons;
    private long hands, trajectories;
    private long nextBranchId = -1;

    Scheduler(
        List<Long> gameIds, CfrPolicy policy, BoundaryValue boundary, CfrEpisodeSink completed) {
      this.gameIds = gameIds;
      this.policy = policy;
      this.boundary = boundary;
      this.completed = completed;
      advanceWorkers = Math.min(config.maxConcurrentGames(), config.arena().advanceWorkers());
      branchBudget = branchSettings.enabled() ? new DecisionBranchBudget(branchSettings) : null;
      admission =
          new InferenceAdmission<>(
              config.batching(),
              (evaluator, capacity) ->
                  capacity == InferenceCapacity.MULTIPLE_TRANSITIONS
                      ? Math.min(config.maxBatchSize(), config.multiTransitionMaxBatch())
                      : config.maxBatchSize(),
              (evaluator, capacity) -> evaluator.tryAcquireInferenceIngress(),
              this::needsSupply);
    }

    private boolean needsSupply(CfrPolicy evaluator) {
      // 入力生成も推論も進んでいない場合は、バッチが満たなくても待機中の入力を処理して対局を進める。
      return evaluator.needsInferenceWork() || !admission.hasInFlightBatches();
    }

    CollectionStats collect() {
      long start = System.nanoTime();
      try (var executor = new ArenaAdvanceExecutor(advanceWorkers)) {
        advanceExecutor = executor;
        try {
          initialValue =
              config.mode() == Mode.HAND_GRP && config.centerUtility()
                  ? evaluateBoundaries(boundary, List.of(initialRoot.publicPrefix())).getFirst()
                  : new double[4];
          refill();
          while (finished < gameIds.size()) {
            requireRunning();
            publishCompleted();
            refill();
            submitReadyAdvances();
            dispatchAvailable();
            scheduleIngressWake();
            if (finished == gameIds.size()) break;
            if (inFlightAdvances == 0
                && !admission.hasInFlightBatches()
                && !admission.hasQueuedRows()
                && !completionWakeScheduled)
              throw new IllegalStateException(
                  "CFR collection has active games without pending work");
            Event event = admission.awaitEvent(events);
            if (event != null) {
              handle(event);
              while ((event = events.poll()) != null) handle(event);
            }
          }
        } catch (Throwable failure) {
          drain(failure);
          throw failure;
        }
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("CFR collection interrupted", interrupted);
      } catch (RuntimeException | Error failure) {
        throw failure;
      } catch (Throwable failure) {
        throw new IllegalStateException("CFR game advancement failed", failure);
      }
      return new CollectionStats(
          finished,
          hands,
          trajectories,
          decisionCount,
          batchCount,
          largest,
          (System.nanoTime() - start) / 1_000_000_000.0,
          branchComparisons,
          branchEpisodes,
          branchRows);
    }

    private void refill() {
      int additions = Math.min(gameIds.size() - next, config.maxConcurrentGames() - activeMain);
      if (additions == 0) return;
      int first = next;
      next += additions;
      activeMain += additions;
      int taskSize = taskSize(additions);
      for (int start = 0; start < additions; start += taskSize) {
        int from = start, to = Math.min(additions, start + taskSize);
        submitAdvance(
            () -> {
              List<RunBoundary> result = new ArrayList<>(to - from);
              for (int index = from; index < to; index++) {
                requireRunning();
                var run = new Run(gameIds.get(first + index), initialRoot, initialValue);
                result.add(prepare(run));
              }
              return result;
            });
      }
    }

    private void submitAdvance(java.util.concurrent.Callable<List<RunBoundary>> work) {
      inFlightAdvances++;
      try {
        advanceExecutor
            .submit(work)
            .whenComplete((runs, failure) -> events.add(new Advanced(runs, unwrap(failure))));
      } catch (RuntimeException | Error failure) {
        events.add(new Advanced(null, failure));
      }
    }

    private void dispatchAvailable() {
      InferenceAdmission.AdmittedBatch<CfrPolicy, InferenceCapacity, Pending> batch;
      while ((batch = admission.startNextEncoding(System.nanoTime(), inFlightAdvances > 0))
          != null) {
        var requests = batch.rows();
        var contexts = requests.stream().map(Pending::context).toList();
        largest = Math.max(largest, contexts.size());
        batchCount++;
        try {
          policy
              .submitBatch(contexts, batch.ingress())
              .whenComplete(
                  (predictions, failure) ->
                      events.add(new Inferred(requests, predictions, unwrap(failure))));
        } catch (RuntimeException | Error failure) {
          events.add(new Inferred(requests, null, failure));
        }
      }
    }

    private void scheduleIngressWake() {
      if (!admission.hasQueuedRows() || !admission.isWaitingForIngress() || ingressWakeScheduled)
        return;
      ingressWakeScheduled = true;
      admission
          .ingressAvailable()
          .whenComplete((ignored, failure) -> events.add(new IngressReady(unwrap(failure))));
    }

    private void handle(Event event) throws Throwable {
      if (event instanceof Inferred inferred) {
        try {
          if (inferred.failure() != null) throw inferred.failure();
          if (inferred.predictions() == null
              || inferred.predictions().size() != inferred.requests().size())
            throw new IllegalStateException("Policy returned a different batch size");
          for (int index = 0; index < inferred.requests().size(); index++) {
            Pending request = inferred.requests().get(index);
            Run run = request.run();
            run.predictions.add(new Selected(request.context(), inferred.predictions().get(index)));
            if (run.branch == null) {
              decisionCount++;
              if (branchBudget != null) branchBudget.recordMainRows(1);
            } else {
              branchRows++;
              branchBudget.recordExtraRows(1);
            }
            if (--run.pendingPredictions == 0) readyAdvances.add(run);
          }
        } finally {
          admission.inferenceFinished();
        }
      } else if (event instanceof Advanced advanced) {
        inFlightAdvances--;
        if (advanced.failure() != null) throw advanced.failure();
        // 子の完了を適用する前に親の保留数を更新し、共有状態の更新が進行ワーカーと競合しないようにする。
        for (RunBoundary result : advanced.runs()) {
          Run run = result.run();
          if (run.createdOutcome != null) {
            run.branchOutcomes.add(run.createdOutcome);
            run.createdOutcome = null;
          }
          if (run.preparedBranch != null) {
            if (run.createdBranch == null) branchBudget.release();
            else {
              run.pendingBranches++;
              branchEpisodes++;
              run.createdBranch = null;
            }
            run.preparedBranch = null;
          }
        }
        long now = System.nanoTime();
        for (RunBoundary result : advanced.runs()) {
          Run run = result.run();
          if (result.contexts().isEmpty()) {
            completedRuns.add(run);
          } else {
            run.pendingPredictions = result.contexts().size();
            for (DecisionContext context : result.contexts())
              admission.add(
                  policy,
                  InferenceCapacity.forActions(context.legalActions()),
                  new Pending(run, context),
                  now);
          }
        }
      } else if (event instanceof IngressReady ready) {
        ingressWakeScheduled = false;
        if (ready.failure() != null) throw ready.failure();
      } else if (event instanceof CompletionReady ready) {
        completionWakeScheduled = false;
        if (ready.failure() != null) throw ready.failure();
      }
    }

    private void publishCompleted() {
      if (!completedRuns.isEmpty()) {
        var ready = complete(completedRuns, waitingForBranches, boundary, branchBudget);
        completedRuns.clear();
        for (Run run : ready)
          pendingCompletions.addLast(
              new CompletedTrajectory(
                  run,
                  new CollectedEpisode(
                      new CfrEpisode(run.id, run.decisions, run.utilities, run.branchOutcomes),
                      run.trajectoryIndex,
                      run.gameEnded,
                      run.observedRoots,
                      run.finalRanks)));
      }
      while (!pendingCompletions.isEmpty()) {
        CompletedTrajectory trajectory = pendingCompletions.getFirst();
        CollectedEpisode episode = trajectory.episode();
        if (!completed.tryAccept(episode)) {
          if (!completionWakeScheduled) {
            completionWakeScheduled = true;
            completed
                .whenAvailable()
                .whenComplete(
                    (ignored, failure) -> events.add(new CompletionReady(unwrap(failure))));
          }
          return;
        }
        pendingCompletions.removeFirst();
        branchComparisons += episode.episode().branchOutcomes().size();
        trajectories++;
        Run run = trajectory.run();
        if (episode.gameEnded()) {
          finished++;
          hands += run.handCount;
          activeMain--;
        } else {
          // 保存先へ渡した局だけを解放して次局へ進む。半荘の同時進行枠は終局まで保持する。
          run.continueHand = true;
          readyAdvances.add(run);
        }
      }
    }

    private void submitReadyAdvances() {
      if (readyAdvances.isEmpty()) return;
      List<Run> runs = List.copyOf(readyAdvances);
      readyAdvances.clear();
      for (Run run : runs) {
        if (run.continueHand) continue;
        run.predictions.sort(Comparator.comparingLong(row -> row.context().decisionId()));
        if (branchBudget == null
            || run.branch != null
            || run.branchCount >= branchSettings.maxBranchesPerGame()) continue;
        for (Selected selected : run.predictions) {
          if (!hasBranchCandidate(selected.context().legalActions())) continue;
          var plan =
              CfrBranchSelection.plan(
                  selected.context().legalActions(),
                  normalizedPolicy(selected.context(), selected.prediction()),
                  config.epsilon(),
                  branchSettings);
          if (plan == null) continue;
          if (branchBudget.admit(run.branchCount)) {
            run.branchCount++;
            run.preparedBranch = new PreparedBranch(selected.context(), plan, nextBranchId--);
          }
          break;
        }
      }
      int taskSize = taskSize(runs.size());
      for (int start = 0; start < runs.size(); start += taskSize) {
        int from = start, to = Math.min(runs.size(), start + taskSize);
        submitAdvance(
            () -> {
              List<RunBoundary> results = new ArrayList<>(2 * (to - from));
              for (int index = from; index < to; index++) {
                requireRunning();
                Run run = runs.get(index);
                if (run.continueHand) run.beginNextHand();
                else advance(run);
                results.add(prepare(run));
                if (run.createdBranch != null) results.add(prepare(run.createdBranch));
              }
              return results;
            });
      }
    }

    private int taskSize(int count) {
      int targetTasks = Math.multiplyExact(advanceWorkers, config.arena().advanceTasksPerWorker());
      return Math.max(
          1, Math.min(config.arena().maximumAdvanceTaskSize(), 1 + (count - 1) / targetTasks));
    }

    private RunBoundary prepare(Run run) {
      if (run.advance()) return new RunBoundary(run, List.of());
      run.selections.clear();
      var awaiting = (GameStepResult.AwaitingDecisions) run.step;
      List<DecisionContext> contexts = new ArrayList<>(awaiting.decisions().size());
      for (EngineDecisionPoint point : awaiting.decisions()) {
        contexts.add(
            new DecisionContext(
                run.id,
                point.id(),
                point.player(),
                run.engine.observation(point.player()),
                point.legalActions(),
                run.root.roundOrdinal()));
      }
      return new RunBoundary(run, List.copyOf(contexts));
    }

    /** 失敗後は新たな対局処理を開始せず、借用中の観測やモデルを使う処理の完了を待つ。 */
    private void drain(Throwable original) {
      boolean interrupted = Thread.interrupted();
      try {
        while (inFlightAdvances > 0 || admission.hasInFlightBatches()) {
          try {
            Event event = events.take();
            if (event instanceof Inferred inferred) {
              admission.inferenceFinished();
              suppress(original, inferred.failure());
            } else if (event instanceof Advanced advanced) {
              inFlightAdvances--;
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

  private sealed interface Event {}

  private record Inferred(List<Pending> requests, List<Prediction> predictions, Throwable failure)
      implements Event {}

  private record Advanced(List<RunBoundary> runs, Throwable failure) implements Event {}

  private record IngressReady(Throwable failure) implements Event {}

  private record CompletionReady(Throwable failure) implements Event {}

  private record RunBoundary(Run run, List<DecisionContext> contexts) {}

  private record CompletedTrajectory(Run run, CollectedEpisode episode) {}

  private static void requireRunning() throws InterruptedException {
    if (Thread.currentThread().isInterrupted())
      throw new InterruptedException("CFR collection interrupted");
  }

  private static Throwable unwrap(Throwable failure) {
    return failure instanceof CompletionException && failure.getCause() != null
        ? failure.getCause()
        : failure;
  }

  private static void suppress(Throwable original, Throwable additional) {
    if (additional != null && original != additional) original.addSuppressed(additional);
  }

  private List<double[]> evaluateBoundaries(BoundaryValue boundary, List<float[]> prefixes) {
    List<double[]> values = new ArrayList<>(prefixes.size());
    for (int from = 0; from < prefixes.size(); from += config.maxBatchSize()) {
      List<float[]> part =
          prefixes.subList(from, Math.min(prefixes.size(), from + config.maxBatchSize()));
      List<double[]> predicted = boundary.evaluateBatch(part);
      if (predicted == null || predicted.size() != part.size()) {
        throw new IllegalStateException("Boundary value returned a different batch size");
      }
      for (double[] value : predicted) values.add(validatedUtilities(value));
    }
    return values;
  }

  private boolean hasBranchCandidate(List<Action> actions) {
    for (Action action : actions) {
      if (branchSettings.kyushu() && DecisionBranchGate.KYUSHU.accepts(action)
          || branchSettings.winDecline() && DecisionBranchGate.RON.accepts(action)) return true;
    }
    return false;
  }

  private List<Run> complete(
      List<Run> runs,
      List<Run> waitingForBranches,
      BoundaryValue boundary,
      DecisionBranchBudget branchBudget) {
    List<Run> bootstrapped = runs.stream().filter(run -> run.terminalPrefix != null).toList();
    if (!bootstrapped.isEmpty()) {
      List<double[]> values =
          evaluateBoundaries(
              boundary, bootstrapped.stream().map(run -> run.terminalPrefix).toList());
      for (int index = 0; index < bootstrapped.size(); index++) {
        Run run = bootstrapped.get(index);
        run.utilities = values.get(index);
        if (run.branch == null) run.nextStartValue = run.utilities.clone();
      }
    }
    for (Run run : runs) {
      run.center();
      if (run.branch == null) {
        run.decisions.sort(Comparator.comparingLong(CfrDecision::decisionId));
        waitingForBranches.add(run);
      } else {
        BranchLink link = run.branch;
        link.owner()
            .branchOutcomes
            .add(
                new CfrBranchOutcome(
                    link.decisionId(),
                    link.choice().mainInclusionProbability(),
                    link.choice().alternativeSlot(),
                    link.choice().alternativeInclusionProbability(),
                    run.utilities[link.seat()],
                    run.logSuffix));
        link.owner().pendingBranches--;
        branchBudget.recordCompletedComparison();
        branchBudget.release();
      }
    }
    List<Run> ready = new ArrayList<>();
    waitingForBranches.removeIf(
        run -> {
          if (run.pendingBranches != 0) return false;
          run.branchOutcomes.sort(Comparator.comparingLong(CfrBranchOutcome::decisionId));
          ready.add(run);
          return true;
        });
    return ready;
  }

  private void advance(Run run) {
    List<SampledAction> selectedActions =
        run.preparedBranch == null ? null : new ArrayList<>(run.predictions.size());
    for (Selected selected : run.predictions) {
      SampledAction sampled =
          run.select(selected.context(), selected.prediction(), selectedActions != null);
      if (selectedActions != null) selectedActions.add(sampled);
    }
    if (run.preparedBranch != null) run.forkBranch(selectedActions);
    run.predictions.clear();
    run.step = run.engine.commitDecisions(run.selections);
  }

  public static double[] terminalUtilities(int[] scores, EpsilonUtilityProfile profile) {
    if (scores.length != 4) throw new IllegalArgumentException("Four scores required");
    int[] ranks = ScoreRanking.byScoreThenSeat(scores);
    double[] utilities = new double[4];
    for (int seat = 0; seat < 4; seat++)
      utilities[seat] = EpsilonUtilityTargets.utilityDouble(profile, scores[seat], ranks[seat]);
    return utilities;
  }

  private double[] validatedUtilities(double[] values) {
    if (values == null || values.length != 4)
      throw new IllegalStateException("Expected four boundary utilities");
    values = values.clone();
    for (double value : values) {
      if (!Double.isFinite(value)
          || value < minimumUtility - .000001
          || value > maximumUtility + .000001) {
        throw new IllegalStateException(
            "Boundary utility is outside the rank utility range: " + value);
      }
    }
    return values;
  }

  /** 受付は遷移数による行数上限だけを分け、実際の行動容量は入力エンコーダーで揃える。 */
  private enum InferenceCapacity {
    SINGLE_TRANSITION,
    MULTIPLE_TRANSITIONS;

    static InferenceCapacity forActions(List<Action> actions) {
      return DecisionBatchBuilder.selectInferenceBucket(actions).actionTransitionCapacity() > 1
          ? MULTIPLE_TRANSITIONS
          : SINGLE_TRANSITION;
    }
  }

  private record Pending(Run run, DecisionContext context) {}

  private record Selected(DecisionContext context, Prediction prediction) {}

  private record PreparedBranch(
      DecisionContext context, CfrBranchSelection.Plan plan, long branchId) {}

  private record SampledAction(DecisionContext context, int slot, double sigma, double behavior) {
    Action action() {
      return context.legalActions().get(slot);
    }

    double logRatio() {
      return sigma == 0 ? Double.NEGATIVE_INFINITY : Math.log(sigma) - Math.log(behavior);
    }
  }

  private record BranchLink(
      Run owner, long decisionId, int seat, CfrBranchSelection.Choice choice) {}

  private static double[] normalizedPolicy(DecisionContext context, Prediction prediction) {
    Objects.requireNonNull(prediction, "prediction");
    double[] sigma = prediction.targetPolicy();
    if (sigma.length != context.legalActions().size() || sigma.length == 0)
      throw new IllegalStateException("Policy/engine legal action mismatch");
    double sum = 0;
    for (double probability : sigma) {
      if (!Double.isFinite(probability) || probability < 0 || probability > 1)
        throw new IllegalStateException("Invalid target probability");
      sum += probability;
    }
    if (Math.abs(sum - 1) > 1e-6)
      throw new IllegalStateException("Target probabilities do not sum to one");
    for (int i = 0; i < sigma.length; i++) sigma[i] /= sum;
    return sigma;
  }

  private final class Run {
    final long id;
    RootState root;
    final List<RootState> observedRoots = new ArrayList<>();
    final GameEngine engine;
    final List<CfrDecision> decisions = new ArrayList<>();
    final List<Selected> predictions = new ArrayList<>(4);
    final EngineSelectionBuffer selections = new EngineSelectionBuffer();
    double[] startValue;
    double[] nextStartValue;
    RootState nextRoot;
    final long actionSeed;
    final BranchLink branch;
    final List<CfrBranchOutcome> branchOutcomes = new ArrayList<>();
    PreparedBranch preparedBranch;
    Run createdBranch;
    CfrBranchOutcome createdOutcome;
    int pendingPredictions;
    int branchCount;
    int pendingBranches;
    int trajectoryIndex;
    int handCount;
    boolean gameEnded;
    boolean continueHand;
    double logSuffix;
    GameStepResult step;
    double[] utilities;
    int[] finalRanks;
    float[] terminalPrefix;

    Run(long id, RootState root, double[] startValue) {
      this.id = id;
      branch = null;
      actionSeed = SeedMixer.mix64(config.seed() ^ SeedMixer.mix64(id) ^ 0x616374696f6eL);
      this.root = root;
      observedRoots.add(root);
      engine =
          new GameEngine(
              SeedMixer.mix64(config.seed() ^ SeedMixer.mix64(id) ^ 0x77616c6cL),
              null,
              root.engineStart());
      this.startValue = startValue;
      step = engine.stepHanchan();
    }

    Run(
        long id,
        GameEngine engine,
        RootState root,
        long actionSeed,
        BranchLink branch,
        double logSuffix,
        EngineSelectionBuffer forced) {
      this.id = id;
      this.engine = engine;
      this.root = root;
      this.actionSeed = actionSeed;
      this.branch = branch;
      this.logSuffix = logSuffix;
      startValue = branch.owner().startValue;
      step = engine.commitDecisions(forced);
    }

    void forkBranch(List<SampledAction> selected) {
      PreparedBranch prepared = preparedBranch;
      long decisionId = prepared.context().decisionId();
      SampledAction main =
          selected.stream()
              .filter(row -> row.context().decisionId() == decisionId)
              .findFirst()
              .orElseThrow();
      long branchSeed = EpsilonDecisionSeeds.branch(actionSeed, decisionId);
      CfrBranchSelection.Choice choice =
          prepared.plan().afterSelected(main.slot(), new SplittableRandom(branchSeed).nextDouble());
      if (choice.alternativeSlot() < 0) {
        if (prepared.plan().gate() == DecisionBranchGate.RON)
          createdOutcome = new CfrBranchOutcome(decisionId, 1, -1, 0, 0, Double.NEGATIVE_INFINITY);
        return;
      }
      GameEngine fork = engine.forkAtDecisionBoundary();
      EngineSelectionBuffer forced = new EngineSelectionBuffer();
      double initialSuffix = 0;
      for (SampledAction action : selected) {
        Action selectedAction =
            action.context().decisionId() == decisionId
                ? action.context().legalActions().get(choice.alternativeSlot())
                : action.action();
        forced.add(action.context().decisionId(), selectedAction);
        if (action.context().decisionId() > decisionId) initialSuffix += action.logRatio();
      }
      createdBranch =
          new Run(
              prepared.branchId(),
              fork,
              root,
              SeedMixer.mix64(branchSeed ^ 0x737566666978L),
              new BranchLink(this, decisionId, main.context().seat(), choice),
              initialSuffix,
              forced);
    }

    boolean advance() {
      while (!(step instanceof GameStepResult.AwaitingDecisions)) {
        if (step instanceof GameStepResult.HanchanEnded ended) {
          handCount++;
          gameEnded = true;
          int[] scores = ended.snapshotFinalScores();
          finalRanks = ScoreRanking.byScoreThenSeat(scores);
          utilities = terminalUtilities(scores, config.utilityProfile());
          return true;
        }
        if (step instanceof GameStepResult.RoundSettled settled) {
          handCount++;
          RoundSettlement settlement = settled.settlement();
          RoundTransition.NextRound next = (RoundTransition.NextRound) settlement.transition();
          nextRoot = root.next(next, settlement.snapshotFinalScores());
          if (config.mode() == Mode.HAND_GRP) {
            terminalPrefix = nextRoot.publicPrefix();
            return true;
          }
          root = nextRoot;
          if (branch == null) observedRoots.add(root);
          step = engine.stepHanchan();
          continue;
        }
        throw new IllegalStateException("Unexpected engine boundary " + step);
      }
      return false;
    }

    /** 局内教師を引き渡した後で、到達した次局を同じエンジンから再開する。 */
    void beginNextHand() {
      root = nextRoot;
      nextRoot = null;
      startValue = config.centerUtility() ? nextStartValue : new double[4];
      nextStartValue = null;
      decisions.clear();
      branchOutcomes.clear();
      observedRoots.clear();
      observedRoots.add(root);
      utilities = null;
      terminalPrefix = null;
      trajectoryIndex++;
      continueHand = false;
      step = engine.stepHanchan();
    }

    void center() {
      for (int seat = 0; seat < 4; seat++) utilities[seat] -= startValue[seat];
    }

    SampledAction select(DecisionContext context, Prediction prediction, boolean retainSelection) {
      double[] sigma = normalizedPolicy(context, prediction);
      double draw =
          new SplittableRandom(SeedMixer.mix64(actionSeed ^ context.decisionId())).nextDouble();
      double cumulative = 0;
      int selected = -1;
      for (int slot = 0; slot < sigma.length; slot++) {
        double mu = (1 - config.epsilon()) * sigma[slot] + config.epsilon() / sigma.length;
        if (mu > 0) selected = slot;
        cumulative += mu;
        if (draw < cumulative) {
          selected = slot;
          break;
        }
      }
      if (selected < 0) throw new IllegalStateException("No sampled action");
      double behavior = (1 - config.epsilon()) * sigma[selected] + config.epsilon() / sigma.length;
      if (config.retainDecisions() && branch == null) {
        int[] actionIds = new int[sigma.length];
        for (int slot = 0; slot < actionIds.length; slot++)
          actionIds[slot] = context.legalActions().get(slot).toIndex();
        decisions.add(
            new CfrDecision(
                context.decisionId(),
                context.seat(),
                actionIds,
                selected,
                sigma,
                behavior,
                prediction.observationPayload()));
      }
      Action action = context.legalActions().get(selected);
      selections.add(context.decisionId(), action);
      if (branch != null)
        logSuffix +=
            sigma[selected] == 0
                ? Double.NEGATIVE_INFINITY
                : Math.log(sigma[selected]) - Math.log(behavior);
      return retainSelection
          ? new SampledAction(context, selected, sigma[selected], behavior)
          : null;
    }
  }
}

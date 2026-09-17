package com.epsilon.nano.ai.decision.audit;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonDecisionSeeds;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpCheckpointBundle;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineCommitResult;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.EngineDecisionOutcome;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineDecisionSelection;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.EpsilonUtilityTargets;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionRequestBatcher;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointBundle;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.ai.network.NetworkDevices;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 基準モデルがパス、候補モデルがチーまたはポンを選ぶ局面で対局状態を複製し、鳴く場合と鳴かない場合の終局結果を比較する。
 *
 * <p>比較する行動だけを変更し、それ以降は両方の対局の全席で基準モデルの最大確率の行動を選ぶ。牌山と分岐時点の状態を複製するため、他の方策の違いを混ぜずに追加の鳴き1回の影響を測れる。
 * 各半荘から比較局面を最大1件、リザーバサンプリングで選び、同一半荘の複数局面を独立標本として数えない。
 */
public final class EpsilonDecisionCallCounterfactualAudit {

  private static final Logger log =
      LoggerFactory.getLogger(EpsilonDecisionCallCounterfactualAudit.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final Gson LINE_GSON = new Gson();
  private static final String SCHEMA = "epsilon-decision-call-counterfactual-audit-v3";
  private static final int COMMON_RANDOM_CONTROL_ROOTS = 16;
  private static final double Z_95 = 1.96;
  private static final long ROOT_RESERVOIR_SALT = 0x6A09E667F3BCC909L;

  private EpsilonDecisionCallCounterfactualAudit() {}

  /**
   * 新しく生成した比較元の最大確率行動半荘から行動の不一致比較局面を採取し、PASSとCHI・PONを分岐評価する。
   *
   * @param reportFile 集約JSON レポートの新規出力先
   * @param traceFile 比較局面別JSONL トレースの新規出力先
   * @param grpDecisionCheckpointDir 重みを固定したGRPとDecision チェックポイントを含むルートディレクトリ
   * @param parentCheckpointDir 比較元 Decision チェックポイント
   * @param candidateCheckpointDir 比較する候補 Decision チェックポイント
   * @param baseGames 行動の不一致を探索する新しく生成した半荘数
   * @param seedBase 基準対局乱数シードの先頭
   * @param gamesInFlight 同時進行させる基準対局数
   * @return 対応をそろえた分岐差と診断を集約した監査レポート
   * @throws Exception チェックポイント読込、対局実行、または新規ファイル出力に失敗した場合
   */
  public static AuditReport run(
      Path reportFile,
      Path traceFile,
      Path grpDecisionCheckpointDir,
      Path parentCheckpointDir,
      Path candidateCheckpointDir,
      int baseGames,
      long seedBase,
      int gamesInFlight)
      throws Exception {
    return run(
        reportFile,
        traceFile,
        grpDecisionCheckpointDir,
        parentCheckpointDir,
        candidateCheckpointDir,
        baseGames,
        seedBase,
        gamesInFlight,
        EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static AuditReport run(
      Path reportFile,
      Path traceFile,
      Path grpDecisionCheckpointDir,
      Path parentCheckpointDir,
      Path candidateCheckpointDir,
      int baseGames,
      long seedBase,
      int gamesInFlight,
      SettingsLoader snapshot)
      throws Exception {
    Objects.requireNonNull(reportFile, "reportFile");
    Objects.requireNonNull(traceFile, "traceFile");
    Objects.requireNonNull(grpDecisionCheckpointDir, "grpDecisionCheckpointDir");
    Objects.requireNonNull(parentCheckpointDir, "parentCheckpointDir");
    Objects.requireNonNull(candidateCheckpointDir, "candidateCheckpointDir");
    if (baseGames <= 0) {
      throw new IllegalArgumentException("baseGames must be positive");
    }
    if (gamesInFlight <= 0) {
      throw new IllegalArgumentException("gamesInFlight must be positive");
    }

    Path report = newOutputFile(reportFile, "reportFile");
    Path trace = newOutputFile(traceFile, "traceFile");
    if (report.equals(trace)) {
      throw new IOException("reportFile and traceFile must be different");
    }
    Path parent = requireCheckpoint(parentCheckpointDir, "parent");
    Path candidate = requireCheckpoint(candidateCheckpointDir, "candidate");
    if (parent.equals(candidate)) {
      throw new IOException("parent and candidate checkpoints must differ");
    }
    requireOutputOutsideCheckpoint(report, parent, candidate);
    requireOutputOutsideCheckpoint(trace, parent, candidate);

    Path grpRoot = grpDecisionCheckpointDir.toRealPath();
    Path grpCheckpoint = EpsilonGrpCheckpointManager.resolveExisting(grpRoot.resolve("grp"));
    if (grpCheckpoint == null) {
      throw new IOException("Frozen GRP checkpoint not found below " + grpRoot.resolve("grp"));
    }
    grpCheckpoint = grpCheckpoint.toRealPath();

    EpsilonDecisionCheckpointBundle parentManifest =
        EpsilonDecisionCheckpointManager.loadManifest(parent);
    EpsilonDecisionCheckpointBundle candidateManifest =
        EpsilonDecisionCheckpointManager.loadManifest(candidate);
    EpsilonGrpCheckpointBundle grpManifest =
        EpsilonGrpCheckpointManager.loadManifest(grpCheckpoint);

    NetworkDevices inferenceDevices =
        NetworkFactory.getInferenceDevices(snapshot.bind(DeviceSettings.class));
    Device parentDevice = inferenceDevices.primary();
    Device candidateDevice = inferenceDevices.get(Math.min(1, inferenceDevices.size() - 1));
    Evaluation evaluation;
    String parentDevices;
    String candidateDevices;
    try (var context = new DecisionExecutionContext();
        EpsilonDecisionEvaluatorFactory.Handle parentHandle =
            EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                parent, NetworkDevices.of(parentDevice), context, snapshot);
        EpsilonDecisionEvaluatorFactory.Handle candidateHandle =
            EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                candidate, NetworkDevices.of(candidateDevice), context, snapshot)) {
      parentDevices = parentHandle.devices();
      candidateDevices = candidateHandle.devices();
      evaluation =
          evaluate(
              parentHandle.evaluator(),
              candidateHandle.evaluator(),
              baseGames,
              seedBase,
              gamesInFlight,
              snapshot);
    }

    List<RootOutcome> outcomes = evaluation.outcomes();
    Files.writeString(
        trace,
        traceLines(outcomes),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE_NEW,
        StandardOpenOption.WRITE);
    PairedMetrics overall = pairedMetrics(outcomes);
    AuditReport audit =
        new AuditReport(
            SCHEMA,
            Instant.now().toString(),
            false,
            false,
            new Protocol(
                "fresh parent-greedy hanchan",
                "parent greedy PASS and candidate greedy CHI/PON; all other simultaneous parent"
                    + " selections are PASS",
                "one uniformly reservoir-sampled eligible root per base hanchan",
                "forced PASS versus forced candidate CHI/PON",
                "parent-greedy all seats after the root",
                "deep-copied engine state and remaining wall",
                "final hanchan rank and score; positive rank delta means the call is worse"),
            baseGames,
            seedBase,
            gamesInFlight,
            snapshot.bind(DecisionSettings.class).utilityProfile().name(),
            checkpointIdentity(parent, parentManifest, parentDevices),
            checkpointIdentity(candidate, candidateManifest, candidateDevices),
            new GrpIdentity(
                grpRoot.toString(),
                grpCheckpoint.toString(),
                grpManifest.iteration,
                "metadata only; no GRP mutation or inference"),
            evaluation.search(),
            evaluation.branchMetrics(),
            overall,
            grouped(outcomes, outcome -> outcome.callType().name()),
            grouped(outcomes, outcome -> "rank" + outcome.rootRank()),
            grouped(outcomes, outcome -> outcome.allLast() ? "all-last" : "before-all-last"),
            trace.toString());
    Files.writeString(
        report,
        GSON.toJson(audit) + System.lineSeparator(),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE_NEW,
        StandardOpenOption.WRITE);
    log.info(
        "Call counterfactual audit complete: report={} trace={} roots={} rankDelta={} ci=[{},{}]"
            + " signP={}",
        report,
        trace,
        overall.roots(),
        overall.callMinusPassRankMean(),
        overall.callMinusPassRankLower95(),
        overall.callMinusPassRankUpper95(),
        overall.twoSidedSignP());
    return audit;
  }

  private static Evaluation evaluate(
      EpsilonDecisionEvaluator parentEvaluator,
      EpsilonDecisionEvaluator candidateEvaluator,
      int baseGames,
      long seedBase,
      int gamesInFlight,
      SettingsLoader snapshot) {
    RootCollection roots =
        collectRoots(parentEvaluator, candidateEvaluator, baseGames, seedBase, gamesInFlight);
    return evaluateRoots(
        parentEvaluator, roots, snapshot.bind(DecisionSettings.class).utilityProfile());
  }

  public static ProbeDataset evaluateProbeDataset(
      EpsilonDecisionEvaluator parentEvaluator,
      EpsilonDecisionEvaluator candidateEvaluator,
      int baseGames,
      long seedBase,
      int gamesInFlight) {
    return evaluateProbeDataset(
        parentEvaluator,
        candidateEvaluator,
        baseGames,
        seedBase,
        gamesInFlight,
        EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static ProbeDataset evaluateProbeDataset(
      EpsilonDecisionEvaluator parentEvaluator,
      EpsilonDecisionEvaluator candidateEvaluator,
      int baseGames,
      long seedBase,
      int gamesInFlight,
      SettingsLoader snapshot) {
    Evaluation evaluation =
        evaluate(parentEvaluator, candidateEvaluator, baseGames, seedBase, gamesInFlight, snapshot);
    return new ProbeDataset(
        evaluation.probeSamples(), evaluation.search(), evaluation.branchMetrics());
  }

  private static RootCollection collectRoots(
      EpsilonDecisionEvaluator parentEvaluator,
      EpsilonDecisionEvaluator candidateEvaluator,
      int baseGames,
      long seedBase,
      int gamesInFlight) {
    ArrayList<BaseContext> active = new ArrayList<>(Math.min(baseGames, gamesInFlight));
    ArrayList<RootSnapshot> roots = new ArrayList<>();
    SearchAccumulator metrics = new SearchAccumulator(baseGames);
    int nextGame = 0;
    while (nextGame < baseGames && active.size() < gamesInFlight) {
      active.add(startBaseGame(nextGame++, seedBase));
    }

    while (!active.isEmpty()) {
      ArrayList<BaseRequest> requests = new ArrayList<>();
      for (BaseContext context : active) {
        collectBaseRequests(context, requests);
      }
      metrics.parentInferenceRequests += requests.size();
      metrics.parentInferenceCalls +=
          EpsilonDecisionRequestBatcher.evaluateRequests(
                  parentEvaluator,
                  requests,
                  BASE_REQUEST_ENCODER,
                  (request, prediction, hostBatch, rowIndex) -> {
                    request.parentPrediction = prediction;
                    request.input = hostBatch.copyRow(rowIndex);
                  })
              .batchCount();

      IdentityHashMap<BaseContext, List<EngineDecisionSelection>> selectionsByGame =
          parentSelections(requests, metrics);
      ArrayList<BaseRequest> candidateRequests =
          candidateRequests(requests, selectionsByGame, metrics);
      metrics.candidateInferenceRequests += candidateRequests.size();
      metrics.candidateInferenceCalls +=
          EpsilonDecisionRequestBatcher.evaluateRequests(
                  candidateEvaluator,
                  candidateRequests,
                  BASE_REQUEST_ENCODER,
                  (request, prediction, hostBatch, rowIndex) ->
                      request.candidatePrediction = prediction)
              .batchCount();
      collectEligibleRoots(candidateRequests, selectionsByGame, metrics);

      for (int index = active.size() - 1; index >= 0; index--) {
        BaseContext context = active.get(index);
        GameStepResult next = context.engine.commitDecisions(selectionsByGame.get(context));
        context.step = advanceToBoundary(context.engine, next);
        if (context.step instanceof GameStepResult.HanchanEnded) {
          metrics.completedBaseGames++;
          if (context.selectedRoot != null) {
            roots.add(context.selectedRoot);
          }
          active.remove(index);
        }
      }
      while (nextGame < baseGames && active.size() < gamesInFlight) {
        active.add(startBaseGame(nextGame++, seedBase));
      }
    }
    roots.sort(Comparator.comparingInt(RootSnapshot::gameIndex));
    metrics.baseGamesWithEligibleRoot = roots.size();
    log.info(
        "Call counterfactual root search complete: baseGames={} eligible={} selected={}"
            + " responseDecisions={} candidateRequests={}",
        baseGames,
        metrics.eligibleDisagreements,
        roots.size(),
        metrics.responseDecisions,
        metrics.candidateInferenceRequests);
    return new RootCollection(List.copyOf(roots), metrics.snapshot());
  }

  private static BaseContext startBaseGame(int gameIndex, long seedBase) {
    long wallSeed = EpsilonDecisionSeeds.evalVsGame(seedBase, gameIndex);
    GameEngine engine = new GameEngine(wallSeed);
    return new BaseContext(
        gameIndex,
        wallSeed,
        new SplittableRandom(wallSeed ^ ROOT_RESERVOIR_SALT),
        engine,
        advanceToBoundary(engine, engine.stepHanchan()));
  }

  private static void collectBaseRequests(BaseContext context, List<BaseRequest> requests) {
    if (!(context.step instanceof GameStepResult.AwaitingDecisions awaiting)) {
      throw new IllegalStateException("Expected base decisions, got " + context.step);
    }
    for (EngineDecisionPoint decision : awaiting.decisions()) {
      requests.add(new BaseRequest(context, decision));
    }
  }

  private static final EpsilonDecisionRequestBatcher.RequestEncoder<BaseRequest>
      BASE_REQUEST_ENCODER =
          new EpsilonDecisionRequestBatcher.RequestEncoder<>() {
            @Override
            public DecisionBucket selectBucket(BaseRequest request) {
              return DecisionBatchBuilder.selectInferenceBucket(request.decision.legalActions());
            }

            @Override
            public void encodeRow(BaseRequest request, DecisionBatchBuilder batchBuilder) {
              batchBuilder.addInferenceRow(
                  request.context.engine, request.decision, DecisionBoundaryContext.uniform());
            }
          };

  private static IdentityHashMap<BaseContext, List<EngineDecisionSelection>> parentSelections(
      List<BaseRequest> requests, SearchAccumulator metrics) {
    IdentityHashMap<BaseContext, List<EngineDecisionSelection>> out = new IdentityHashMap<>();
    for (BaseRequest request : requests) {
      Action action = greedyAction(request.decision, request.parentPrediction);
      request.parentAction = action;
      if (request.decision.kind() == EngineDecisionKind.RESPONSE) {
        metrics.responseDecisions++;
      }
      out.computeIfAbsent(request.context, ignored -> new ArrayList<>())
          .add(new EngineDecisionSelection(request.decision.id(), action));
    }
    return out;
  }

  private static ArrayList<BaseRequest> candidateRequests(
      List<BaseRequest> requests,
      IdentityHashMap<BaseContext, List<EngineDecisionSelection>> selectionsByGame,
      SearchAccumulator metrics) {
    ArrayList<BaseRequest> out = new ArrayList<>();
    for (BaseRequest request : requests) {
      if (request.decision.kind() != EngineDecisionKind.RESPONSE
          || request.parentAction.type() != Action.Type.PASS
          || !hasChiOrPon(request.decision.legalActions())) {
        continue;
      }
      metrics.parentPassWithCallLegal++;
      if (!otherSelectionsArePass(selectionsByGame.get(request.context), request.decision.id())) {
        metrics.excludedByOtherParentResponse++;
        continue;
      }
      out.add(request);
    }
    return out;
  }

  private static void collectEligibleRoots(
      List<BaseRequest> requests,
      IdentityHashMap<BaseContext, List<EngineDecisionSelection>> selectionsByGame,
      SearchAccumulator metrics) {
    for (BaseRequest request : requests) {
      Action candidateAction = greedyAction(request.decision, request.candidatePrediction);
      if (!isChiOrPon(candidateAction)) {
        continue;
      }
      metrics.candidateCallDisagreements++;
      metrics.eligibleDisagreements++;
      BaseContext context = request.context;
      RootSnapshot root =
          rootSnapshot(
              metrics.eligibleDisagreements,
              context,
              request,
              candidateAction,
              selectionsByGame.get(context));
      context.eligibleRoots++;
      if (context.reservoir.nextLong(context.eligibleRoots) == 0L) {
        context.selectedRoot = root;
      }
    }
  }

  private static RootSnapshot rootSnapshot(
      long eligibleOrdinal,
      BaseContext context,
      BaseRequest request,
      Action call,
      List<EngineDecisionSelection> parentSelections) {
    GameState state = context.engine.getState();
    int target = request.decision.player();
    int[] scores = snapshotScores(state);
    int[] ranks = ScoreRanking.byScoreThenSeat(scores);
    float[] parentPolicy = request.parentPrediction.policyProbabilities();
    float[] candidatePolicy = request.candidatePrediction.policyProbabilities();
    int passSlot = slotOf(request.decision.legalActions(), Action.Type.PASS);
    int callSlot = slotOf(request.decision.legalActions(), call);
    int sourcePlayer =
        state.getTurnEvent() instanceof TurnEvent.Discard discard ? discard.player() : -1;
    return new RootSnapshot(
        eligibleOrdinal,
        context.gameIndex,
        context.wallSeed,
        context.engine.forkAtDecisionBoundary(),
        request.input,
        List.copyOf(parentSelections),
        request.decision.id(),
        target,
        sourcePlayer,
        request.parentAction,
        call,
        state.getKyokuIndex(),
        state.getTurnNumber(),
        state.getHonba(),
        state.getKyotakuCount(),
        state.getKyokuIndex() >= GameState.HANCHAN_KYOKU_COUNT - 1,
        scores,
        ranks[target] + 1,
        parentPolicy[passSlot],
        parentPolicy[callSlot],
        candidatePolicy[passSlot],
        candidatePolicy[callSlot],
        request.parentPrediction.valueUtility(),
        request.candidatePrediction.valueUtility());
  }

  private static Evaluation evaluateRoots(
      EpsilonDecisionEvaluator parentEvaluator,
      RootCollection collection,
      EpsilonUtilityProfile profile) {
    ArrayList<PairContext> pairs = new ArrayList<>(collection.roots().size());
    ArrayList<BranchContext> active = new ArrayList<>(collection.roots().size() * 2);
    int skipped = 0;
    int controlRoots = Math.min(COMMON_RANDOM_CONTROL_ROOTS, collection.roots().size());
    for (int index = 0; index < collection.roots().size(); index++) {
      RootSnapshot root = collection.roots().get(index);
      PairContext pair = startPair(root, index < controlRoots);
      if (pair == null) {
        skipped++;
        continue;
      }
      pairs.add(pair);
      if (pair.pass.finalScores == null) {
        active.add(pair.pass);
      }
      if (pair.call.finalScores == null) {
        active.add(pair.call);
      }
      if (pair.passReplay != null && pair.passReplay.finalScores == null) {
        active.add(pair.passReplay);
      }
    }

    long inferenceRequests = 0L;
    long inferenceCalls = 0L;
    while (!active.isEmpty()) {
      ArrayList<BranchRequest> requests = new ArrayList<>();
      for (BranchContext context : active) {
        collectBranchRequests(context, requests);
      }
      inferenceRequests += requests.size();
      inferenceCalls +=
          EpsilonDecisionRequestBatcher.evaluateRequests(
                  parentEvaluator,
                  requests,
                  BRANCH_REQUEST_ENCODER,
                  (request, prediction, hostBatch, rowIndex) -> request.prediction = prediction)
              .batchCount();
      IdentityHashMap<BranchContext, List<EngineDecisionSelection>> selections =
          branchSelections(requests);
      for (int index = active.size() - 1; index >= 0; index--) {
        BranchContext context = active.get(index);
        context.step =
            advanceToBoundary(
                context.engine, context.engine.commitDecisions(selections.get(context)));
        if (context.step instanceof GameStepResult.HanchanEnded ended) {
          context.finalScores = ended.snapshotFinalScores();
          active.remove(index);
        }
      }
    }

    ArrayList<RootOutcome> outcomes = new ArrayList<>(pairs.size());
    ArrayList<ProbeSample> probeSamples = new ArrayList<>(pairs.size());
    int controlsChecked = 0;
    for (PairContext pair : pairs) {
      if (pair.pass.finalScores == null || pair.call.finalScores == null) {
        throw new IllegalStateException("Counterfactual branch did not reach terminal state");
      }
      if (pair.passReplay != null) {
        controlsChecked++;
        if (!Arrays.equals(pair.pass.finalScores, pair.passReplay.finalScores)) {
          throw new IllegalStateException(
              "Common-random replay mismatch at root " + pair.root.eligibleOrdinal());
        }
      }
      RootOutcome outcome = toOutcome(pair, profile);
      outcomes.add(outcome);
      probeSamples.add(
          new ProbeSample(
              outcome,
              pair.root.rootInput(),
              pair.pass.afterStateInput,
              pair.call.afterStateInput));
    }
    outcomes.sort(Comparator.comparingInt(RootOutcome::gameIndex));
    BranchMetrics branchMetrics =
        new BranchMetrics(
            pairs.size(),
            skipped,
            Math.multiplyExact(pairs.size(), 2) + controlsChecked,
            inferenceRequests,
            inferenceCalls,
            controlsChecked,
            0);
    log.info(
        "Call counterfactual branches complete: roots={} skipped={} branchGames={} controls={}"
            + " requests={} calls={}",
        pairs.size(),
        skipped,
        branchMetrics.branchGames(),
        controlsChecked,
        inferenceRequests,
        inferenceCalls);
    return new Evaluation(
        List.copyOf(outcomes), List.copyOf(probeSamples), collection.search(), branchMetrics);
  }

  private static PairContext startPair(RootSnapshot root, boolean controlReplay) {
    GameEngine passEngine = root.engine().forkAtDecisionBoundary();
    GameEngine callEngine = root.engine().forkAtDecisionBoundary();
    EngineCommitResult passCommit =
        passEngine.commitDecisionsWithOutcomes(replaceRootSelection(root, root.passAction()));
    EngineCommitResult callCommit =
        callEngine.commitDecisionsWithOutcomes(replaceRootSelection(root, root.callAction()));
    if (!executed(callCommit, root.targetDecisionId())) {
      return null;
    }
    BranchContext pass =
        new BranchContext(
            passEngine, advanceToBoundary(passEngine, passCommit.step()), root.targetSeat());
    BranchContext call =
        new BranchContext(
            callEngine, advanceToBoundary(callEngine, callCommit.step()), root.targetSeat());
    BranchContext replay = null;
    if (controlReplay) {
      GameEngine replayEngine = root.engine().forkAtDecisionBoundary();
      EngineCommitResult replayCommit =
          replayEngine.commitDecisionsWithOutcomes(replaceRootSelection(root, root.passAction()));
      replay =
          new BranchContext(
              replayEngine,
              advanceToBoundary(replayEngine, replayCommit.step()),
              root.targetSeat());
    }
    return new PairContext(root, pass, call, replay);
  }

  private static List<EngineDecisionSelection> replaceRootSelection(
      RootSnapshot root, Action replacement) {
    ArrayList<EngineDecisionSelection> out = new ArrayList<>(root.parentSelections().size());
    for (EngineDecisionSelection selection : root.parentSelections()) {
      out.add(
          selection.decisionId() == root.targetDecisionId()
              ? new EngineDecisionSelection(selection.decisionId(), replacement)
              : selection);
    }
    return out;
  }

  private static boolean executed(EngineCommitResult result, long decisionId) {
    for (EngineDecisionOutcome outcome : result.decisionOutcomes()) {
      if (outcome.decisionId() == decisionId) {
        return outcome.selectedActionWasExecuted();
      }
    }
    throw new IllegalStateException("Missing root decision outcome " + decisionId);
  }

  private static void collectBranchRequests(BranchContext context, List<BranchRequest> requests) {
    if (!(context.step instanceof GameStepResult.AwaitingDecisions awaiting)) {
      throw new IllegalStateException("Expected branch decisions, got " + context.step);
    }
    for (EngineDecisionPoint decision : awaiting.decisions()) {
      requests.add(new BranchRequest(context, decision));
    }
  }

  private static final EpsilonDecisionRequestBatcher.RequestEncoder<BranchRequest>
      BRANCH_REQUEST_ENCODER =
          new EpsilonDecisionRequestBatcher.RequestEncoder<>() {
            @Override
            public DecisionBucket selectBucket(BranchRequest request) {
              return DecisionBatchBuilder.selectInferenceBucket(request.decision.legalActions());
            }

            @Override
            public void encodeRow(BranchRequest request, DecisionBatchBuilder batchBuilder) {
              batchBuilder.addInferenceRow(
                  request.context.engine, request.decision, DecisionBoundaryContext.uniform());
            }
          };

  private static IdentityHashMap<BranchContext, List<EngineDecisionSelection>> branchSelections(
      List<BranchRequest> requests) {
    IdentityHashMap<BranchContext, List<EngineDecisionSelection>> out = new IdentityHashMap<>();
    for (BranchRequest request : requests) {
      out.computeIfAbsent(request.context, ignored -> new ArrayList<>())
          .add(
              new EngineDecisionSelection(
                  request.decision.id(), greedyAction(request.decision, request.prediction)));
    }
    return out;
  }

  private static RootOutcome toOutcome(PairContext pair, EpsilonUtilityProfile profile) {
    RootSnapshot root = pair.root;
    int seat = root.targetSeat();
    int[] passScores = pair.pass.finalScores;
    int[] callScores = pair.call.finalScores;
    int passRank = ScoreRanking.byScoreThenSeat(passScores)[seat] + 1;
    int callRank = ScoreRanking.byScoreThenSeat(callScores)[seat] + 1;
    float passUtility = EpsilonUtilityTargets.fromFinalScores(seat, passScores)[profile.ordinal()];
    float callUtility = EpsilonUtilityTargets.fromFinalScores(seat, callScores)[profile.ordinal()];
    return new RootOutcome(
        root.eligibleOrdinal(),
        root.gameIndex(),
        root.wallSeed(),
        root.targetSeat(),
        root.sourcePlayer(),
        root.kyokuIndex(),
        root.turn(),
        root.honba(),
        root.kyotaku(),
        root.allLast(),
        root.rootScores(),
        root.rootRank(),
        root.callAction().type(),
        root.callAction().toIndex(),
        root.parentPassProbability(),
        root.parentCallProbability(),
        root.candidatePassProbability(),
        root.candidateCallProbability(),
        root.parentExpectedUtility(),
        root.candidateExpectedUtility(),
        passScores,
        callScores,
        passRank,
        callRank,
        callRank - passRank,
        passScores[seat],
        callScores[seat],
        callScores[seat] - passScores[seat],
        passUtility,
        callUtility,
        callUtility - passUtility,
        pair.passReplay != null);
  }

  static PairedMetrics pairedMetrics(List<RootOutcome> outcomes) {
    int n = outcomes.size();
    if (n == 0) {
      return new PairedMetrics(
          0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    }
    double[] rank = new double[n];
    double[] score = new double[n];
    double[] utility = new double[n];
    double topDelta = 0.0;
    double lastDelta = 0.0;
    int worse = 0;
    int better = 0;
    int tied = 0;
    for (int index = 0; index < n; index++) {
      RootOutcome outcome = outcomes.get(index);
      rank[index] = outcome.callMinusPassRank();
      score[index] = outcome.callMinusPassScore();
      utility[index] = outcome.callMinusPassConfiguredUtility();
      if (rank[index] > 0.0) {
        worse++;
      } else if (rank[index] < 0.0) {
        better++;
      } else {
        tied++;
      }
      topDelta += indicator(outcome.callRank() == 1) - indicator(outcome.passRank() == 1);
      lastDelta += indicator(outcome.callRank() == 4) - indicator(outcome.passRank() == 4);
    }
    MeanInterval rankInterval = meanInterval(rank);
    MeanInterval scoreInterval = meanInterval(score);
    MeanInterval utilityInterval = meanInterval(utility);
    return new PairedMetrics(
        n,
        worse,
        better,
        tied,
        rankInterval.mean(),
        rankInterval.standardError(),
        rankInterval.lower95(),
        rankInterval.upper95(),
        scoreInterval.mean(),
        scoreInterval.lower95(),
        scoreInterval.upper95(),
        utilityInterval.mean(),
        utilityInterval.lower95(),
        utilityInterval.upper95(),
        topDelta / n,
        lastDelta / n,
        twoSidedSignP(worse, better));
  }

  private static Map<String, PairedMetrics> grouped(
      List<RootOutcome> outcomes, java.util.function.Function<RootOutcome, String> keyFunction) {
    LinkedHashMap<String, ArrayList<RootOutcome>> groups = new LinkedHashMap<>();
    for (RootOutcome outcome : outcomes) {
      groups.computeIfAbsent(keyFunction.apply(outcome), ignored -> new ArrayList<>()).add(outcome);
    }
    LinkedHashMap<String, PairedMetrics> out = new LinkedHashMap<>();
    for (Map.Entry<String, ArrayList<RootOutcome>> entry : groups.entrySet()) {
      out.put(entry.getKey(), pairedMetrics(entry.getValue()));
    }
    return Map.copyOf(out);
  }

  private static MeanInterval meanInterval(double[] values) {
    if (values.length == 0) {
      return new MeanInterval(0.0, 0.0, 0.0, 0.0);
    }
    double sum = 0.0;
    for (double value : values) {
      sum += value;
    }
    double mean = sum / values.length;
    double standardError = 0.0;
    if (values.length > 1) {
      double squared = 0.0;
      for (double value : values) {
        double deviation = value - mean;
        squared += deviation * deviation;
      }
      standardError = Math.sqrt((squared / (values.length - 1)) / values.length);
    }
    return new MeanInterval(
        mean, standardError, mean - Z_95 * standardError, mean + Z_95 * standardError);
  }

  static double twoSidedSignP(int positive, int negative) {
    if (positive < 0 || negative < 0) {
      throw new IllegalArgumentException("sign counts must be non-negative");
    }
    int n = positive + negative;
    if (n == 0) {
      return 1.0;
    }
    int tail = Math.min(positive, negative);
    double logTerm = -n * Math.log(2.0);
    double sum = 0.0;
    for (int k = 0; k <= tail; k++) {
      sum += Math.exp(logTerm);
      if (k < tail) {
        logTerm += Math.log(n - k) - Math.log(k + 1.0);
      }
    }
    return Math.min(1.0, 2.0 * sum);
  }

  private static Action greedyAction(
      EngineDecisionPoint decision, EpsilonDecisionInferenceServer.Prediction prediction) {
    if (prediction.policySize() != decision.legalActions().size()) {
      throw new IllegalStateException(
          "Policy/legal action mismatch: policy="
              + prediction.policySize()
              + " legal="
              + decision.legalActions().size());
    }
    return decision.legalActions().get(prediction.greedyActionSlot());
  }

  private static int slotOf(List<Action> actions, Action.Type type) {
    for (int slot = 0; slot < actions.size(); slot++) {
      if (actions.get(slot).type() == type) {
        return slot;
      }
    }
    throw new IllegalStateException("Action type not legal: " + type);
  }

  private static int slotOf(List<Action> actions, Action action) {
    int slot = actions.indexOf(action);
    if (slot < 0) {
      throw new IllegalStateException("Action not legal: " + action);
    }
    return slot;
  }

  private static boolean hasChiOrPon(List<Action> actions) {
    return actions.stream().anyMatch(EpsilonDecisionCallCounterfactualAudit::isChiOrPon);
  }

  private static boolean isChiOrPon(Action action) {
    return action.type() == Action.Type.CHI || action.type() == Action.Type.PON;
  }

  private static boolean otherSelectionsArePass(
      List<EngineDecisionSelection> selections, long targetDecisionId) {
    for (EngineDecisionSelection selection : selections) {
      if (selection.decisionId() != targetDecisionId
          && selection.action().type() != Action.Type.PASS) {
        return false;
      }
    }
    return true;
  }

  private static GameStepResult advanceToBoundary(GameEngine engine, GameStepResult step) {
    GameStepResult current = step;
    while (current instanceof GameStepResult.RoundSettled) {
      current = engine.stepHanchan();
    }
    return current;
  }

  private static int[] snapshotScores(GameState state) {
    int[] scores = new int[GameState.NUM_PLAYERS];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = state.getScore(seat);
    }
    return scores;
  }

  private static int indicator(boolean value) {
    return value ? 1 : 0;
  }

  private static String traceLines(List<RootOutcome> outcomes) {
    StringBuilder out = new StringBuilder();
    for (RootOutcome outcome : outcomes) {
      out.append(LINE_GSON.toJson(outcome)).append(System.lineSeparator());
    }
    return out.toString();
  }

  public static Path newOutputFile(Path value, String label) throws IOException {
    Path output = value.toAbsolutePath().normalize();
    if (Files.exists(output)) {
      throw new IOException("Refusing to overwrite call counterfactual " + label + ": " + output);
    }
    Path parent = output.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    return output;
  }

  public static Path requireCheckpoint(Path value, String label) throws IOException {
    Path checkpoint = value.toRealPath();
    EpsilonDecisionCheckpointManager.requireValidCheckpoint(checkpoint);
    return checkpoint;
  }

  public static void requireOutputOutsideCheckpoint(Path output, Path parent, Path candidate)
      throws IOException {
    if (output.startsWith(parent) || output.startsWith(candidate)) {
      throw new IOException("Audit output must be outside immutable checkpoints: " + output);
    }
  }

  public static CheckpointIdentity checkpointIdentity(
      Path path, EpsilonDecisionCheckpointBundle manifest, String devices) {
    return new CheckpointIdentity(
        path.toString(), manifest.globalStep, manifest.iteration, manifest.selfPlayGames, devices);
  }

  /**
   * PASS と実行可能 CALL を同じ乱数列を使う分岐後の対局で比較した監査レポート。
   *
   * @param schema レポートスキーマ ID
   * @param generatedAt 生成時刻
   * @param training 学習を行ったか
   * @param promotionMutation 採用状態を変更したか
   * @param protocol 行動を変えた場合の比較構築規則
   * @param baseGames 比較局面探索に使った基底対局数
   * @param seedBase 基底牌山乱数シード
   * @param gamesInFlight 同時進行対局数
   * @param configuredUtilityProfile 比較に使った効用の定義
   * @param parent 比較元チェックポイント識別情報
   * @param candidate 候補チェックポイント識別情報
   * @param grp 固定 GRP 教師モデル識別情報
   * @param rootSearch 比較局面探索指標
   * @param branchMetrics 行動を変えた場合の比較分岐実行指標
   * @param overall 全対象条件を満たす比較局面の対応をそろえた指標
   * @param byCallType CALL 種類別対応をそろえた指標
   * @param byRootRank 比較局面時点順位別対応をそろえた指標
   * @param byAllLast オーラス別対応をそろえた指標
   * @param traceFile 比較局面単位トレースファイル
   */
  public record AuditReport(
      String schema,
      String generatedAt,
      boolean training,
      boolean promotionMutation,
      Protocol protocol,
      int baseGames,
      long seedBase,
      int gamesInFlight,
      String configuredUtilityProfile,
      CheckpointIdentity parent,
      CheckpointIdentity candidate,
      GrpIdentity grp,
      RootSearch rootSearch,
      BranchMetrics branchMetrics,
      PairedMetrics overall,
      Map<String, PairedMetrics> byCallType,
      Map<String, PairedMetrics> byRootRank,
      Map<String, PairedMetrics> byAllLast,
      String traceFile) {}

  /**
   * 行動を変えた場合の比較監査の再現可能な分岐規則。
   *
   * @param baseTrajectory 比較局面までの基底対局中の行動履歴
   * @param rootEligibility 比較対象比較局面の条件
   * @param withinGameSampling 対局内比較局面無作為抽出規則
   * @param intervention PASS／CALL 分岐の適用方法
   * @param continuationPolicy 分岐後に使う方策
   * @param commonRandomNumbers 両分岐の乱数共有規則
   * @param primaryOutcome 主要評価量
   */
  public record Protocol(
      String baseTrajectory,
      String rootEligibility,
      String withinGameSampling,
      String intervention,
      String continuationPolicy,
      String commonRandomNumbers,
      String primaryOutcome) {}

  /**
   * 監査で固定した Decision チェックポイント識別情報。
   *
   * @param path チェックポイントパス
   * @param globalStep チェックポイント累積更新回数
   * @param iteration チェックポイント反復回数
   * @param selfPlayGames チェックポイントまでの自己対局対局数
   * @param devices 推論に使ったデバイス
   */
  public record CheckpointIdentity(
      String path, int globalStep, int iteration, int selfPlayGames, String devices) {}

  /**
   * 監査で固定した GRP 教師モデル識別情報。
   *
   * @param root チェックポイントのルートディレクトリ
   * @param checkpoint チェックポイントパス
   * @param iteration チェックポイント反復回数
   * @param use 監査内での用途
   */
  public record GrpIdentity(String root, String checkpoint, int iteration, String use) {}

  /**
   * 実行可能な PASS／CALL 行動の不一致比較局面を探した過程の件数。
   *
   * @param requestedBaseGames 要求した基底対局数
   * @param completedBaseGames 完走した基底対局数
   * @param responseDecisions 観測した応答判断数
   * @param parentPassWithCallLegal 比較元が PASS し CALL も合法だった件数
   * @param excludedByOtherParentResponse 他家の優先応答により除外した件数
   * @param candidateCallDisagreements 候補が CALL を選ぶ行動の不一致件数
   * @param eligibleDisagreements 分岐実行条件を満たした行動の不一致件数
   * @param baseGamesWithEligibleRoot 対象条件を満たす比較局面を含んだ基底対局数
   * @param parentInferenceRequests 比較元の論理推論要求数
   * @param parentInferenceCalls 比較元の物理順伝播数
   * @param candidateInferenceRequests 候補の論理推論要求数
   * @param candidateInferenceCalls 候補の物理順伝播数
   */
  public record RootSearch(
      int requestedBaseGames,
      int completedBaseGames,
      long responseDecisions,
      long parentPassWithCallLegal,
      long excludedByOtherParentResponse,
      long candidateCallDisagreements,
      long eligibleDisagreements,
      int baseGamesWithEligibleRoot,
      long parentInferenceRequests,
      long parentInferenceCalls,
      long candidateInferenceRequests,
      long candidateInferenceCalls) {}

  /**
   * 比較局面から二つの分岐後の対局を実行した件数と乱数整合性。
   *
   * @param evaluatedRoots 評価を完了した比較局面数
   * @param skippedUnexecutedCalls 優先度解決で CALL が実行されず除外した数
   * @param branchGames 実行した分岐対局数
   * @param parentInferenceRequests 比較元の論理推論要求数
   * @param parentInferenceCalls 比較元の物理順伝播数
   * @param commonRandomControlsChecked 乱数共有を検査した対照数
   * @param commonRandomControlFailures 乱数共有が一致しなかった対照数
   */
  public record BranchMetrics(
      int evaluatedRoots,
      int skippedUnexecutedCalls,
      int branchGames,
      long parentInferenceRequests,
      long parentInferenceCalls,
      int commonRandomControlsChecked,
      int commonRandomControlFailures) {}

  /**
   * CALL 結果から PASS 結果を引いた対応をそろえた集約値。
   *
   * @param roots 比較局面数
   * @param callWorse CALL の順位が悪化した数
   * @param callBetter CALL の順位が改善した数
   * @param rankTies 順位差ゼロの数
   * @param callMinusPassRankMean 順位効用差の平均
   * @param callMinusPassRankStandardError 順位効用差の標準誤差
   * @param callMinusPassRankLower95 順位効用差の95%下限
   * @param callMinusPassRankUpper95 順位効用差の95%上限
   * @param callMinusPassScoreMean 最終点差の平均
   * @param callMinusPassScoreLower95 最終点差の95%下限
   * @param callMinusPassScoreUpper95 最終点差の95%上限
   * @param callMinusPassConfiguredUtilityMean 設定効用差の平均
   * @param callMinusPassConfiguredUtilityLower95 設定効用差の95%下限
   * @param callMinusPassConfiguredUtilityUpper95 設定効用差の95%上限
   * @param callMinusPassTopRate 1着指示変数差の平均
   * @param callMinusPassLastRate 4着指示変数差の平均
   * @param twoSidedSignP 同順位を除く両側符号検定の p 値
   */
  public record PairedMetrics(
      int roots,
      int callWorse,
      int callBetter,
      int rankTies,
      double callMinusPassRankMean,
      double callMinusPassRankStandardError,
      double callMinusPassRankLower95,
      double callMinusPassRankUpper95,
      double callMinusPassScoreMean,
      double callMinusPassScoreLower95,
      double callMinusPassScoreUpper95,
      double callMinusPassConfiguredUtilityMean,
      double callMinusPassConfiguredUtilityLower95,
      double callMinusPassConfiguredUtilityUpper95,
      double callMinusPassTopRate,
      double callMinusPassLastRate,
      double twoSidedSignP) {}

  /**
   * 一つの対象条件を満たす比較局面と PASS／CALL 分岐後の対局の完全な対応をそろえた結果。
   *
   * @param eligibleOrdinal 対象条件を満たす比較局面の通し番号
   * @param gameIndex 基底対局インデックス
   * @param wallSeed 牌山乱数シード
   * @param targetSeat 行動の変更対象席
   * @param sourcePlayer 捨て牌元の席
   * @param kyokuIndex 局インデックス
   * @param turn 比較局面の巡目
   * @param honba 比較局面の本場
   * @param kyotaku 比較局面の供託本数
   * @param allLast 比較局面がオーラスなら {@code true}
   * @param rootScores 比較局面時点の4人点棒
   * @param rootRank 評価対象の席の比較局面時点順位
   * @param callType 変更後の CALL の種類
   * @param callActionId CALL の安定行動 ID
   * @param parentPassProbability 比較元の PASS 確率
   * @param parentCallProbability 比較元の CALL 確率
   * @param candidatePassProbability 候補の PASS 確率
   * @param candidateCallProbability 候補の CALL 確率
   * @param parentExpectedUtility 比較元価値の期待効用
   * @param candidateExpectedUtility 候補価値の期待効用
   * @param passFinalScores PASS 分岐の終局点棒
   * @param callFinalScores CALL 分岐の終局点棒
   * @param passRank PASS 分岐の評価対象の席順位
   * @param callRank CALL 分岐の評価対象の席順位
   * @param callMinusPassRank CALL から PASS を引いた順位効用差
   * @param passScore PASS 分岐の評価対象の席点棒
   * @param callScore CALL 分岐の評価対象の席点棒
   * @param callMinusPassScore CALL から PASS を引いた点差
   * @param passConfiguredUtility PASS 分岐の設定効用
   * @param callConfiguredUtility CALL 分岐の設定効用
   * @param callMinusPassConfiguredUtility CALL から PASS を引いた設定効用
   * @param commonRandomReplayChecked 両分岐の共通乱数対照を確認したか
   */
  public record RootOutcome(
      long eligibleOrdinal,
      int gameIndex,
      long wallSeed,
      int targetSeat,
      int sourcePlayer,
      int kyokuIndex,
      int turn,
      int honba,
      int kyotaku,
      boolean allLast,
      int[] rootScores,
      int rootRank,
      Action.Type callType,
      int callActionId,
      double parentPassProbability,
      double parentCallProbability,
      double candidatePassProbability,
      double candidateCallProbability,
      double parentExpectedUtility,
      double candidateExpectedUtility,
      int[] passFinalScores,
      int[] callFinalScores,
      int passRank,
      int callRank,
      int callMinusPassRank,
      int passScore,
      int callScore,
      int callMinusPassScore,
      double passConfiguredUtility,
      double callConfiguredUtility,
      double callMinusPassConfiguredUtility,
      boolean commonRandomReplayChecked) {

    /** 点数配列を防御複製し、監査結果を変更不可にする。 */
    public RootOutcome {
      rootScores = rootScores.clone();
      passFinalScores = passFinalScores.clone();
      callFinalScores = callFinalScores.clone();
    }
  }

  public record ProbeDataset(
      List<ProbeSample> samples, RootSearch search, BranchMetrics branchMetrics) {}

  public record ProbeSample(
      RootOutcome outcome,
      DecisionHostBatch rootInput,
      DecisionHostBatch passAfterStateInput,
      DecisionHostBatch callAfterStateInput) {}

  private record Evaluation(
      List<RootOutcome> outcomes,
      List<ProbeSample> probeSamples,
      RootSearch search,
      BranchMetrics branchMetrics) {}

  private record RootCollection(List<RootSnapshot> roots, RootSearch search) {}

  private record RootSnapshot(
      long eligibleOrdinal,
      int gameIndex,
      long wallSeed,
      GameEngine engine,
      DecisionHostBatch rootInput,
      List<EngineDecisionSelection> parentSelections,
      long targetDecisionId,
      int targetSeat,
      int sourcePlayer,
      Action passAction,
      Action callAction,
      int kyokuIndex,
      int turn,
      int honba,
      int kyotaku,
      boolean allLast,
      int[] rootScores,
      int rootRank,
      double parentPassProbability,
      double parentCallProbability,
      double candidatePassProbability,
      double candidateCallProbability,
      double parentExpectedUtility,
      double candidateExpectedUtility) {

    private RootSnapshot {
      rootScores = rootScores.clone();
    }
  }

  private static final class BaseContext {
    private final int gameIndex;
    private final long wallSeed;
    private final SplittableRandom reservoir;
    private final GameEngine engine;
    private GameStepResult step;
    private long eligibleRoots;
    private RootSnapshot selectedRoot;

    private BaseContext(
        int gameIndex,
        long wallSeed,
        SplittableRandom reservoir,
        GameEngine engine,
        GameStepResult step) {
      this.gameIndex = gameIndex;
      this.wallSeed = wallSeed;
      this.reservoir = reservoir;
      this.engine = engine;
      this.step = step;
    }
  }

  private static final class BaseRequest {
    private final BaseContext context;
    private final EngineDecisionPoint decision;
    private EpsilonDecisionInferenceServer.Prediction parentPrediction;
    private EpsilonDecisionInferenceServer.Prediction candidatePrediction;
    private Action parentAction;
    private DecisionHostBatch input;

    private BaseRequest(BaseContext context, EngineDecisionPoint decision) {
      this.context = context;
      this.decision = decision;
    }
  }

  private static final class SearchAccumulator {
    private final int requestedBaseGames;
    private int completedBaseGames;
    private long responseDecisions;
    private long parentPassWithCallLegal;
    private long excludedByOtherParentResponse;
    private long candidateCallDisagreements;
    private long eligibleDisagreements;
    private int baseGamesWithEligibleRoot;
    private long parentInferenceRequests;
    private long parentInferenceCalls;
    private long candidateInferenceRequests;
    private long candidateInferenceCalls;

    private SearchAccumulator(int requestedBaseGames) {
      this.requestedBaseGames = requestedBaseGames;
    }

    private RootSearch snapshot() {
      return new RootSearch(
          requestedBaseGames,
          completedBaseGames,
          responseDecisions,
          parentPassWithCallLegal,
          excludedByOtherParentResponse,
          candidateCallDisagreements,
          eligibleDisagreements,
          baseGamesWithEligibleRoot,
          parentInferenceRequests,
          parentInferenceCalls,
          candidateInferenceRequests,
          candidateInferenceCalls);
    }
  }

  private static final class PairContext {
    private final RootSnapshot root;
    private final BranchContext pass;
    private final BranchContext call;
    private final BranchContext passReplay;

    private PairContext(
        RootSnapshot root, BranchContext pass, BranchContext call, BranchContext passReplay) {
      this.root = root;
      this.pass = pass;
      this.call = call;
      this.passReplay = passReplay;
    }
  }

  private static final class BranchContext {
    private final GameEngine engine;
    private final DecisionHostBatch afterStateInput;
    private GameStepResult step;
    private int[] finalScores;

    private BranchContext(GameEngine engine, GameStepResult step, int targetSeat) {
      this.engine = engine;
      this.step = step;
      this.afterStateInput = DecisionBatchBuilder.stateOnlyBatch(engine.getState(), targetSeat);
      if (step instanceof GameStepResult.HanchanEnded ended) {
        finalScores = ended.snapshotFinalScores();
      }
    }
  }

  private static final class BranchRequest {
    private final BranchContext context;
    private final EngineDecisionPoint decision;
    private EpsilonDecisionInferenceServer.Prediction prediction;

    private BranchRequest(BranchContext context, EngineDecisionPoint decision) {
      this.context = context;
      this.decision = decision;
    }
  }

  private record MeanInterval(double mean, double standardError, double lower95, double upper95) {}
}

package com.epsilon.pico.ai.decision.audit;

import ai.djl.Device;
import com.epsilon.ai.decision.DecisionSelectionMode;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.ai.grp.EpsilonGrpTeacherIdentity;
import com.epsilon.config.settings.DecisionFullSupportSettings;
import com.epsilon.config.settings.DecisionRolloutSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.pico.ai.decision.EpsilonDecisionConstants;
import com.epsilon.pico.ai.decision.EpsilonDecisionReturns;
import com.epsilon.pico.ai.decision.EpsilonUtilityTargets;
import com.epsilon.pico.ai.decision.arena.DecisionSnapshotEvaluatorProvider;
import com.epsilon.pico.ai.decision.arena.EpsilonDecisionArena;
import com.epsilon.pico.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.pico.ai.decision.arena.EpsilonDecisionPopulationCollector;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionGameBoundary;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionSampleRecord;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.ai.decision.policy.EpsilonDecisionBehaviorPolicy;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.pico.ai.decision.training.DecisionSampleBatcher;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointBundle;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.pico.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.pico.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.pico.ai.network.NetworkDevices;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.TreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 新たに生成した同じ局面を更新前モデルと候補モデルで評価し、価値・方策・教師分布の差を比較する。
 *
 * <p>各モデルで対局を生成した2つの対局群を、同じ乱数シードと席順入れ替えで用意する。各群の観測を両モデルへ同じ順序で与え、推論結果の差と対局生成による状態分布の差を分けて調べる。学習、チェックポイントの保存、採用状態の変更は行わない。
 */
public final class EpsilonDecisionPairedTrajectoryAudit {

  private static final Logger log =
      LoggerFactory.getLogger(EpsilonDecisionPairedTrajectoryAudit.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final Gson LINE_GSON = new Gson();
  private static final String SCHEMA = "epsilon-decision-paired-trajectory-audit-v4";
  private static final long PARENT_SNAPSHOT_ID = 1L;
  private static final long CANDIDATE_SNAPSHOT_ID = 2L;
  private static final int TOP_EXAMPLES = 64;
  private static final double PROBABILITY_FLOOR = 1.0e-12;
  private static final float STORED_ADVANTAGE_TOLERANCE = 2.0e-5f;

  private EpsilonDecisionPairedTrajectoryAudit() {}

  /**
   * 比較元・候補駆動で各 {@code gamesPerSource} 半荘を生成し、全サンプルを両チェックポイントで再評価する。
   *
   * @param reportFile 集約監査レポートの新規出力先
   * @param traceFile サンプル単位トレースの新規出力先
   * @param scratchDir 対局中の行動履歴対局単位の学習データを置く未作成の一時ディレクトリ
   * @param grpDecisionCheckpointDir 重みを固定したGRP チェックポイントを含む親ディレクトリ
   * @param parentCheckpointDir 比較元 Decision チェックポイント
   * @param candidateCheckpointDir 候補 Decision チェックポイント
   * @param gamesPerSource 各駆動方策で生成する半荘数。4の正の倍数
   * @param seedBase 二つの対局群で共有する乱数シード列の先頭値
   * @return 入力元別・チェックポイント別の価値、方策、分布による教師値差をまとめたレポート
   * @throws Exception 対局中の行動履歴生成、推論、保存物検証、またはレポート書込に失敗した場合
   */
  public static AuditReport run(
      Path reportFile,
      Path traceFile,
      Path scratchDir,
      Path grpDecisionCheckpointDir,
      Path parentCheckpointDir,
      Path candidateCheckpointDir,
      int gamesPerSource,
      long seedBase)
      throws Exception {
    return run(
        reportFile,
        traceFile,
        scratchDir,
        grpDecisionCheckpointDir,
        parentCheckpointDir,
        candidateCheckpointDir,
        gamesPerSource,
        seedBase,
        EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static AuditReport run(
      Path reportFile,
      Path traceFile,
      Path scratchDir,
      Path grpDecisionCheckpointDir,
      Path parentCheckpointDir,
      Path candidateCheckpointDir,
      int gamesPerSource,
      long seedBase,
      SettingsLoader snapshot)
      throws Exception {
    Objects.requireNonNull(reportFile, "reportFile");
    Objects.requireNonNull(traceFile, "traceFile");
    Objects.requireNonNull(scratchDir, "scratchDir");
    Objects.requireNonNull(grpDecisionCheckpointDir, "grpDecisionCheckpointDir");
    Objects.requireNonNull(parentCheckpointDir, "parentCheckpointDir");
    Objects.requireNonNull(candidateCheckpointDir, "candidateCheckpointDir");
    if (gamesPerSource <= 0 || gamesPerSource % 4 != 0) {
      throw new IllegalArgumentException("gamesPerSource must be positive and divisible by 4");
    }

    Path report = newOutputFile(reportFile, "reportFile");
    Path trace = newOutputFile(traceFile, "traceFile");
    if (report.equals(trace)) {
      throw new IOException("reportFile and traceFile must be different");
    }
    Path scratch = scratchDir.toAbsolutePath().normalize();
    if (Files.exists(scratch)) {
      throw new IOException("Refusing to reuse paired audit scratch directory: " + scratch);
    }

    Path parent = requireCheckpoint(parentCheckpointDir);
    Path candidate = requireCheckpoint(candidateCheckpointDir);
    if (parent.equals(candidate)) {
      throw new IOException("parent and candidate checkpoints must differ");
    }
    requireOutputOutsideCheckpoint(report, parent, candidate);
    requireOutputOutsideCheckpoint(trace, parent, candidate);
    requireOutputOutsideCheckpoint(scratch, parent, candidate);
    Files.createDirectories(scratch);

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

    NetworkDevices inferenceDevices =
        NetworkFactory.getInferenceDevices(snapshot.bind(DeviceSettings.class));
    Device parentDevice = inferenceDevices.primary();
    Device candidateDevice = inferenceDevices.get(Math.min(1, inferenceDevices.size() - 1));
    var campaign = snapshot.bind(DecisionSelectedPgCampaignSettings.class);
    EpsilonDecisionPlayer.RolloutConfig actorRollout =
        new EpsilonDecisionPlayer.RolloutConfig(
            snapshot.bind(DecisionRolloutSettings.class).selectionMode(),
            snapshot.bind(DecisionFullSupportSettings.class),
            campaign.causalTraceLambda(),
            campaign.explorationCreditMix());
    EpsilonDecisionPopulationCollector.requireSelectedPgActor(actorRollout);
    EpsilonDecisionPlayer.RolloutConfig opponentRollout =
        new EpsilonDecisionPlayer.RolloutConfig(
            DecisionSelectionMode.POLICY_GREEDY,
            DecisionFullSupportSettings.disabled(),
            campaign.causalTraceLambda(),
            campaign.explorationCreditMix());

    ArrayList<SourceAudit> sources = new ArrayList<>(2);
    EpsilonGrpTeacherIdentity grpIdentity;
    try (var context = new DecisionExecutionContext();
        EpsilonDecisionEvaluatorFactory.Handle parentHandle =
            EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                parent, NetworkDevices.of(parentDevice), context, snapshot);
        EpsilonDecisionEvaluatorFactory.Handle candidateHandle =
            EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                candidate, NetworkDevices.of(candidateDevice), context, snapshot);
        EpsilonGrpTrainingSession grpSession =
            EpsilonGrpTrainingSession.openInferenceOnly(
                grpRoot,
                NetworkFactory.getGrpDevices(snapshot.bind(DeviceSettings.class)).primary(),
                snapshot);
        BufferedWriter traceWriter =
            Files.newBufferedWriter(
                trace,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE)) {
      grpIdentity = grpSession.identity();
      DecisionSnapshotEvaluatorProvider parentProvider =
          snapshotId -> {
            if (snapshotId != PARENT_SNAPSHOT_ID) {
              throw new IOException("Unexpected paired-audit opponent snapshot: " + snapshotId);
            }
            return parentHandle.evaluator();
          };
      long[][] opponentIds = parentOpponentIds();

      sources.add(
          collectSource(
              "parent-driven",
              parent.toString(),
              parentHandle.evaluator(),
              PARENT_SNAPSHOT_ID,
              parentHandle.evaluator(),
              candidateHandle.evaluator(),
              parentProvider,
              opponentIds,
              actorRollout,
              opponentRollout,
              grpSession,
              gamesPerSource,
              seedBase,
              scratch.resolve("parent-driven"),
              traceWriter,
              true,
              snapshot));
      sources.add(
          collectSource(
              "candidate-driven",
              candidate.toString(),
              candidateHandle.evaluator(),
              CANDIDATE_SNAPSHOT_ID,
              parentHandle.evaluator(),
              candidateHandle.evaluator(),
              parentProvider,
              opponentIds,
              actorRollout,
              opponentRollout,
              grpSession,
              gamesPerSource,
              seedBase,
              scratch.resolve("candidate-driven"),
              traceWriter,
              false,
              snapshot));
    }

    AuditReport audit =
        new AuditReport(
            SCHEMA,
            Instant.now().toString(),
            false,
            false,
            gamesPerSource,
            Math.multiplyExact(gamesPerSource, 2),
            seedBase,
            new RolloutSettings(
                actorRollout.selectionMode(),
                actorRollout.fullSupport().terminalGateExplorationMass(),
                actorRollout.fullSupport().callGateExplorationMass(),
                actorRollout.fullSupport().meldTypeExplorationMass(),
                actorRollout.fullSupport().meldCandidateExplorationMass(),
                actorRollout.fullSupport().kanGateExplorationMass(),
                actorRollout.fullSupport().kanTypeExplorationMass(),
                actorRollout.fullSupport().kanCandidateExplorationMass(),
                actorRollout.fullSupport().discardIdentityExplorationMass(),
                actorRollout.fullSupport().riichiGateExplorationMass(),
                actorRollout.causalTraceLambda(),
                actorRollout.explorationCreditMix(),
                opponentRollout.selectionMode()),
            new CheckpointIdentity(
                parent.toString(),
                parentManifest.globalStep,
                parentManifest.iteration,
                parentManifest.selfPlayGames,
                parentDevice.toString()),
            new CheckpointIdentity(
                candidate.toString(),
                candidateManifest.globalStep,
                candidateManifest.iteration,
                candidateManifest.selfPlayGames,
                candidateDevice.toString()),
            new GrpIdentity(
                grpRoot.toString(),
                grpCheckpoint.toString(),
                grpIdentity.iteration(),
                parentDevice.toString()),
            trace.toString(),
            List.copyOf(sources));
    Files.writeString(
        report,
        GSON.toJson(audit) + System.lineSeparator(),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE_NEW,
        StandardOpenOption.WRITE);
    log.info(
        "Paired trajectory audit written: report={} trace={} sources={}",
        report,
        trace,
        sources.size());
    return audit;
  }

  private static SourceAudit collectSource(
      String sourceLabel,
      String sourceCheckpoint,
      EpsilonDecisionEvaluator sourceEvaluator,
      long sourceSnapshotId,
      EpsilonDecisionEvaluator parentEvaluator,
      EpsilonDecisionEvaluator candidateEvaluator,
      DecisionSnapshotEvaluatorProvider parentProvider,
      long[][] opponentIds,
      EpsilonDecisionPlayer.RolloutConfig actorRollout,
      EpsilonDecisionPlayer.RolloutConfig opponentRollout,
      EpsilonGrpTrainingSession grpSession,
      int games,
      long seedBase,
      Path scratch,
      BufferedWriter traceWriter,
      boolean sourceIsParent,
      SettingsLoader snapshot)
      throws Exception {
    log.info(
        "Paired trajectory source start: source={} checkpoint={} games={} seedBase={}",
        sourceLabel,
        sourceCheckpoint,
        games,
        seedBase);
    SourceAccumulator accumulator = new SourceAccumulator(sourceLabel, traceWriter);
    EpsilonDecisionArena.Metrics arenaMetrics;
    try (EpsilonDecisionTrajectoryPayloadStore payloadStore =
        EpsilonDecisionTrajectoryPayloadStore.asyncFileBacked(scratch, 1024, 2, false)) {
      arenaMetrics =
          EpsilonDecisionPopulationCollector.collect(
              EpsilonDecisionPopulationCollector.Request.standard(
                      games,
                      seedBase,
                      sourceEvaluator,
                      sourceSnapshotId,
                      opponentIds,
                      parentProvider,
                      actorRollout,
                      opponentRollout,
                      (gameIndex, gameSeed, game) ->
                          processGame(
                              sourceLabel,
                              gameIndex,
                              gameSeed,
                              game.samples(),
                              game.boundaries(),
                              parentEvaluator,
                              candidateEvaluator,
                              actorRollout.causalTraceLambda(),
                              actorRollout.explorationCreditMix(),
                              sourceIsParent,
                              accumulator),
                      payloadStore)
                  .withGrpInference(grpSession.inference())
                  .withGameIndexPolicy(EpsilonDecisionArena.GameIndexPolicy.fourSeatRotation(0L)),
              snapshot);
    }
    SourceAudit source = accumulator.toAudit(sourceCheckpoint, games, arenaMetrics);
    if (source.sourceStoredAdvantageMaxAbsError() > STORED_ADVANTAGE_TOLERANCE) {
      throw new IOException(
          "Source scalar advantage reproduction exceeded tolerance: source="
              + sourceLabel
              + " maxAbsError="
              + source.sourceStoredAdvantageMaxAbsError());
    }
    log.info(
        "Paired trajectory source complete: source={} games={} samples={} signDisagreement={}"
            + " greedyDisagreement={} storedAdvantageMaxAbsError={}",
        sourceLabel,
        games,
        source.overall().samples(),
        source.overall().advantageSignDisagreementRate(),
        source.overall().parentCandidateGreedyDisagreementRate(),
        source.sourceStoredAdvantageMaxAbsError());
    return source;
  }

  private static void processGame(
      String sourceLabel,
      int gameIndex,
      long gameSeed,
      List<? extends EpsilonDecisionSampleRecord> records,
      List<EpsilonDecisionGameBoundary> boundaries,
      EpsilonDecisionEvaluator parentEvaluator,
      EpsilonDecisionEvaluator candidateEvaluator,
      float causalTraceLambda,
      float explorationCreditMix,
      boolean sourceIsParent,
      SourceAccumulator accumulator)
      throws Exception {
    ArrayList<EpsilonDecisionSample> samples = new ArrayList<>(records.size());
    for (EpsilonDecisionSampleRecord record : records) {
      EpsilonDecisionSample sample = record.materialize();
      samples.add(sample);
    }
    if (samples.isEmpty()) {
      throw new IOException("Paired trajectory game produced no actor samples: " + gameIndex);
    }
    List<EpsilonDecisionInferenceServer.Prediction> parentPredictions =
        evaluateSamples(parentEvaluator, samples);
    List<EpsilonDecisionInferenceServer.Prediction> candidatePredictions =
        evaluateSamples(candidateEvaluator, samples);
    requirePredictionCount("parent", samples.size(), parentPredictions.size());
    requirePredictionCount("candidate", samples.size(), candidatePredictions.size());

    float[] parentAdvantages =
        recomputeAdvantages(
            samples,
            parentPredictions,
            boundaries,
            causalTraceLambda,
            explorationCreditMix,
            sourceIsParent);
    float[] candidateAdvantages =
        recomputeAdvantages(
            samples,
            candidatePredictions,
            boundaries,
            causalTraceLambda,
            explorationCreditMix,
            !sourceIsParent);
    float[] sourceAdvantages = sourceIsParent ? parentAdvantages : candidateAdvantages;
    for (int index = 0; index < samples.size(); index++) {
      EpsilonDecisionSample sample = samples.get(index);
      float storedError = Math.abs(sourceAdvantages[index] - sample.advantage());
      TraceRecord trace =
          traceRecord(
              sourceLabel,
              gameIndex,
              gameSeed,
              index,
              sample,
              parentPredictions.get(index),
              candidatePredictions.get(index),
              parentAdvantages[index],
              candidateAdvantages[index]);
      accumulator.add(trace, storedError);
    }
    accumulator.finishGame(samples.getFirst().finalRank());
  }

  private static List<EpsilonDecisionInferenceServer.Prediction> evaluateSamples(
      EpsilonDecisionEvaluator evaluator, List<EpsilonDecisionSample> samples) {
    return DecisionSampleBatcher.evaluateInBucketedBatches(evaluator, samples);
  }

  /** 現行方策と同じ局境界GRP endpointとCAUSALに分類された行動の遷移列でスカラーアドバンテージを再現する。 */
  static float[] recomputeAdvantages(
      List<EpsilonDecisionSample> samples,
      List<EpsilonDecisionInferenceServer.Prediction> predictions,
      List<EpsilonDecisionGameBoundary> boundaries,
      float lambda,
      float explorationCreditMix) {
    return recomputeAdvantages(
        samples, predictions, boundaries, lambda, explorationCreditMix, false);
  }

  private static float[] recomputeAdvantages(
      List<EpsilonDecisionSample> samples,
      List<EpsilonDecisionInferenceServer.Prediction> predictions,
      List<EpsilonDecisionGameBoundary> boundaries,
      float lambda,
      float explorationCreditMix,
      boolean useStoredRolloutPolicy) {
    if (samples.size() != predictions.size()) {
      throw new IllegalArgumentException("sample/prediction size mismatch");
    }
    float[] advantages = new float[samples.size()];
    float[] nextPredictionBySeat = new float[EpsilonDecisionConstants.PLAYERS];
    float[] nextTargetBySeat = new float[EpsilonDecisionConstants.PLAYERS];
    boolean[] hasNextBySeat = new boolean[EpsilonDecisionConstants.PLAYERS];
    int[] activeBoundaryBySeat = new int[EpsilonDecisionConstants.PLAYERS];
    float[] nextTraceCoefficientBySeat = new float[EpsilonDecisionConstants.PLAYERS];
    Arrays.fill(activeBoundaryBySeat, -1);
    Arrays.fill(nextTraceCoefficientBySeat, 1.0f);
    for (int index = samples.size() - 1; index >= 0; index--) {
      EpsilonDecisionSample sample = samples.get(index);
      int seat = sample.playerSeat();
      if (activeBoundaryBySeat[seat] != sample.boundaryIndex()) {
        activeBoundaryBySeat[seat] = sample.boundaryIndex();
        hasNextBySeat[seat] = false;
        nextTraceCoefficientBySeat[seat] = 1.0f;
        EpsilonUtilityProfile profile = EpsilonUtilityProfile.values()[sample.ruleProfile()];
        int nextBoundary = sample.boundaryIndex() + 1;
        nextTargetBySeat[seat] =
            nextBoundary < boundaries.size()
                ? EpsilonUtilityTargets.expectedRankUtility(
                    profile,
                    EpsilonGrpRanks.seatMarginal(
                        boundaries.get(nextBoundary).grpRankProbabilities(), seat))
                : profile.utilityForRank(sample.finalRank());
      }
      if (sample.learningRole().advancesActorClock()) {
        float current = predictions.get(index).valueUtility();
        float traceWeight = lambda * nextTraceCoefficientBySeat[seat];
        float target =
            !hasNextBySeat[seat]
                ? nextTargetBySeat[seat]
                : (1.0f - traceWeight) * nextPredictionBySeat[seat]
                    + traceWeight * nextTargetBySeat[seat];
        advantages[index] = target - current;
        nextPredictionBySeat[seat] = current;
        nextTargetBySeat[seat] = target;
        hasNextBySeat[seat] = true;
        EpsilonDecisionInferenceServer.Prediction prediction = predictions.get(index);
        float selectedRolloutProbability =
            useStoredRolloutPolicy
                ? sample.rolloutPolicy()[sample.chosenLegalSlot()]
                : canonicalSelectedRolloutProbability(prediction, sample.chosenLegalSlot());
        nextTraceCoefficientBySeat[seat] =
            EpsilonDecisionReturns.selectedRetraceCoefficient(
                selectedRolloutProbability, sample.behaviorProb(), explorationCreditMix);
      }
    }
    return advantages;
  }

  private static float canonicalSelectedRolloutProbability(
      EpsilonDecisionInferenceServer.Prediction prediction, int selectedSlot) {
    return EpsilonDecisionBehaviorPolicy.directInPlace(prediction.policyProbabilities())
        .rolloutPolicy()[selectedSlot];
  }

  private static TraceRecord traceRecord(
      String source,
      int gameIndex,
      long gameSeed,
      int decisionIndex,
      EpsilonDecisionSample sample,
      EpsilonDecisionInferenceServer.Prediction parentPrediction,
      EpsilonDecisionInferenceServer.Prediction candidatePrediction,
      float parentAdvantage,
      float candidateAdvantage) {
    float[] parentPolicy = parentPrediction.policyProbabilities();
    float[] candidatePolicy = candidatePrediction.policyProbabilities();
    int parentGreedySlot = parentPrediction.greedyActionSlot();
    int candidateGreedySlot = candidatePrediction.greedyActionSlot();
    int parentGreedyAction = sample.input().legalActionId(0, parentGreedySlot);
    int candidateGreedyAction = sample.input().legalActionId(0, candidateGreedySlot);
    int selectedSlot = sample.chosenLegalSlot();
    double parentValueUtility = parentPrediction.valueUtility();
    double candidateValueUtility = candidatePrediction.valueUtility();
    double targetUtility = sample.valueTarget();
    double parentScalarAdvantage = parentAdvantage;
    double candidateScalarAdvantage = candidateAdvantage;
    ScoreContext score = scoreContext(sample.input());
    return new TraceRecord(
        source,
        gameIndex,
        gameSeed,
        decisionIndex,
        sample.gameId(),
        sample.playerSeat(),
        EpsilonGrpFeature.steps(sample.grpFeatureSequenceView()),
        score.kyokuIndex(),
        score.turn(),
        score.honba(),
        score.kyotaku(),
        score.allLast(),
        score.relativeScores(),
        score.currentRank(),
        score.scoreBucket(),
        score.nearestGapBucket(),
        sample.finalRank() + 1,
        sample.chosenActionId(),
        actionType(sample.chosenActionId()),
        sample.learningRole().advancesActorClock(),
        parentGreedyAction,
        actionType(parentGreedyAction),
        candidateGreedyAction,
        actionType(candidateGreedyAction),
        parentPolicy[selectedSlot],
        candidatePolicy[selectedSlot],
        symmetricKl(parentPolicy, candidatePolicy),
        parentValueUtility,
        candidateValueUtility,
        targetUtility,
        sample.advantage(),
        parentScalarAdvantage,
        candidateScalarAdvantage,
        candidateScalarAdvantage - parentScalarAdvantage);
  }

  static ScoreContext scoreContext(DecisionHostBatch input) {
    int[] relativeScores = new int[EpsilonDecisionConstants.PLAYERS];
    for (int rel = 0; rel < relativeScores.length; rel++) {
      relativeScores[rel] =
          Math.round(
              input.playerNumeric(0, rel, DecisionInputSchema.PlayerFloat.SCORE) * 100_000.0f);
    }
    int currentRank = input.playerCategory(0, 0, DecisionInputSchema.PlayerInt.RANK);
    int nearestGap = Integer.MAX_VALUE;
    for (int rel = 1; rel < relativeScores.length; rel++) {
      nearestGap = Math.min(nearestGap, Math.abs(relativeScores[0] - relativeScores[rel]));
    }
    return new ScoreContext(
        input.roundCategory(0, DecisionInputSchema.RoundInt.KYOKU_INDEX) - 1,
        input.roundCategory(0, DecisionInputSchema.RoundInt.TURN_NUMBER),
        input.roundCategory(0, DecisionInputSchema.RoundInt.HONBA),
        input.roundCategory(0, DecisionInputSchema.RoundInt.KYOTAKU),
        input.roundCategory(0, DecisionInputSchema.RoundInt.ALL_LAST) != 0,
        relativeScores,
        currentRank,
        scoreBucket(relativeScores[0]),
        gapBucket(nearestGap));
  }

  private static String scoreBucket(int score) {
    if (score < 15_000) {
      return "lt15000";
    }
    if (score < 25_000) {
      return "15000-24999";
    }
    if (score < 35_000) {
      return "25000-34999";
    }
    return "ge35000";
  }

  private static String gapBucket(int gap) {
    if (gap < 1_000) {
      return "lt1000";
    }
    if (gap < 4_000) {
      return "1000-3999";
    }
    if (gap < 8_000) {
      return "4000-7999";
    }
    return "ge8000";
  }

  private static double symmetricKl(float[] left, float[] right) {
    double total = 0.0;
    for (int index = 0; index < left.length; index++) {
      double p = Math.max(PROBABILITY_FLOOR, left[index]);
      double q = Math.max(PROBABILITY_FLOOR, right[index]);
      total += 0.5 * (p * Math.log(p / q) + q * Math.log(q / p));
    }
    return total;
  }

  private static Action.Type actionType(int actionId) {
    return Action.fromIndex(actionId).type();
  }

  private static long[][] parentOpponentIds() {
    long[][] out = new long[EpsilonDecisionConstants.PLAYERS][];
    for (int seat = 0; seat < out.length; seat++) {
      out[seat] = new long[] {PARENT_SNAPSHOT_ID, PARENT_SNAPSHOT_ID, PARENT_SNAPSHOT_ID};
    }
    return out;
  }

  private static void requirePredictionCount(String label, int expected, int actual)
      throws IOException {
    if (expected != actual) {
      throw new IOException(
          label + " prediction count mismatch: expected=" + expected + " actual=" + actual);
    }
  }

  private static Path newOutputFile(Path value, String label) throws IOException {
    Path output = value.toAbsolutePath().normalize();
    if (Files.exists(output)) {
      throw new IOException("Refusing to overwrite paired audit " + label + ": " + output);
    }
    Path parent = output.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    return output;
  }

  private static Path requireCheckpoint(Path value) throws IOException {
    Path checkpoint = value.toRealPath();
    EpsilonDecisionCheckpointManager.requireValidCheckpoint(checkpoint);
    return checkpoint;
  }

  private static void requireOutputOutsideCheckpoint(Path output, Path parent, Path candidate)
      throws IOException {
    if (output.startsWith(parent) || output.startsWith(candidate)) {
      throw new IOException("Audit output must be outside immutable checkpoints: " + output);
    }
  }

  /**
   * 同じ対局中の行動履歴サンプルを比較元／候補で再評価した対応をそろえた監査レポート。
   *
   * @param schema レポートスキーマ ID
   * @param generatedAt 生成時刻
   * @param training 学習を行ったか
   * @param promotionMutation 採用状態を変更したか
   * @param gamesPerSource 入力元チェックポイントごとの対局数
   * @param totalGames 全入力元の総対局数
   * @param seedBase 基底対局乱数シード
   * @param rollout サンプル収集に使った対局生成規則
   * @param parent 比較元チェックポイント識別情報
   * @param candidate 候補チェックポイント識別情報
   * @param grp 固定 GRP 教師モデル識別情報
   * @param traceFile 行動の不一致トレースファイル
   * @param sources 対局生成入力元ごとの監査結果
   */
  public record AuditReport(
      String schema,
      String generatedAt,
      boolean training,
      boolean promotionMutation,
      int gamesPerSource,
      int totalGames,
      long seedBase,
      RolloutSettings rollout,
      CheckpointIdentity parent,
      CheckpointIdentity candidate,
      GrpIdentity grp,
      String traceFile,
      List<SourceAudit> sources) {}

  /**
   * 対応をそろえた対局中の行動履歴収集時の条件付き節点別無作為抽出設定。
   *
   * @param actorSelectionMode 行動選択プレイヤーの選択方式
   * @param terminalGateExplorationMass 終端二択の判定の探索に割り当てる確率
   * @param callGateExplorationMass CALL 二択の判定の探索に割り当てる確率
   * @param meldTypeExplorationMass 面子種類の探索に割り当てる確率
   * @param meldCandidateExplorationMass 面子候補の探索に割り当てる確率
   * @param kanGateExplorationMass KAN 二択の判定の探索に割り当てる確率
   * @param kanTypeExplorationMass KAN 種類の探索に割り当てる確率
   * @param kanCandidateExplorationMass KAN 候補の探索に割り当てる確率
   * @param discardIdentityExplorationMass 打牌の識別情報の探索に割り当てる確率
   * @param riichiGateExplorationMass RIICHI 二択の判定の探索に割り当てる確率
   * @param causalTraceLambda 因果関係を持つ方策更新用の遷移列のスカラー値のトレース係数
   * @param explorationCreditMix q-retに使う探索による選択の学習への寄与混合率
   * @param opponentSelectionMode 対戦相手の選択方式
   */
  public record RolloutSettings(
      DecisionSelectionMode actorSelectionMode,
      float terminalGateExplorationMass,
      float callGateExplorationMass,
      float meldTypeExplorationMass,
      float meldCandidateExplorationMass,
      float kanGateExplorationMass,
      float kanTypeExplorationMass,
      float kanCandidateExplorationMass,
      float discardIdentityExplorationMass,
      float riichiGateExplorationMass,
      float causalTraceLambda,
      float explorationCreditMix,
      DecisionSelectionMode opponentSelectionMode) {}

  /**
   * 監査で固定した Decision チェックポイント識別情報。
   *
   * @param path チェックポイントパス
   * @param globalStep チェックポイント累積更新回数
   * @param iteration チェックポイント反復回数
   * @param selfPlayGames チェックポイントまでの自己対局対局数
   * @param device 推論デバイス
   */
  public record CheckpointIdentity(
      String path, int globalStep, int iteration, int selfPlayGames, String device) {}

  /**
   * 対局中の行動履歴教師値に使った GRP 教師モデル識別情報。
   *
   * @param root チェックポイントの親ディレクトリ
   * @param checkpoint チェックポイントパス
   * @param iteration チェックポイント反復回数
   * @param device 推論デバイス
   */
  public record GrpIdentity(String root, String checkpoint, int iteration, String device) {}

  /**
   * 一つの対局生成入力元から得たサンプルを比較元／候補で再評価した結果。
   *
   * @param source 入力元ラベル
   * @param sourceCheckpoint 対局生成入力元チェックポイントパス
   * @param games 収集対局数
   * @param samples Decision サンプル数
   * @param inferenceBatches 入力元収集時の物理推論バッチ数
   * @param inferenceRequests 入力元収集時の論理推論要求数
   * @param maxInferenceBatch 最大推論バッチ行数
   * @param sourceStoredAdvantageMaxAbsError 保存アドバンテージ再計算の最大絶対誤差
   * @param sourceAverageFinalRank 入力元行動選択プレイヤーの平均終局順位
   * @param sourceTopRate 入力元行動選択プレイヤーの1着率
   * @param sourceLastRate 入力元行動選択プレイヤーの4着率
   * @param sourceFinalRankCounts 終局順位別件数
   * @param overall 全サンプルの対応をそろえた指標
   * @param bySelectedActionType 選択行動種類別指標
   * @param byCurrentRank 判断時順位別指標
   * @param byScoreBucket 点差容量区分別指標
   * @param byAllLastRank オーラス順位別指標
   * @param byAllLastRankGap オーラス着順差別指標
   * @param parentToCandidateGreedyType 比較元／候補最大確率の行動を選ぶ方式種類遷移件数
   * @param topDisagreements 対称KLダイバージェンス上位のトレース
   */
  public record SourceAudit(
      String source,
      String sourceCheckpoint,
      int games,
      long samples,
      long inferenceBatches,
      long inferenceRequests,
      int maxInferenceBatch,
      float sourceStoredAdvantageMaxAbsError,
      double sourceAverageFinalRank,
      double sourceTopRate,
      double sourceLastRate,
      long[] sourceFinalRankCounts,
      PairMetrics overall,
      Map<String, PairMetrics> bySelectedActionType,
      Map<String, PairMetrics> byCurrentRank,
      Map<String, PairMetrics> byScoreBucket,
      Map<String, PairMetrics> byAllLastRank,
      Map<String, PairMetrics> byAllLastRankGap,
      Map<String, Long> parentToCandidateGreedyType,
      List<TraceRecord> topDisagreements) {}

  /**
   * 同一選択したサンプルを比較元／候補で評価した対応をそろえた集約値。
   *
   * @param samples サンプル数
   * @param causalActorRate 方策損失対象となるCAUSAL サンプルの割合
   * @param parentValueTargetBias 比較元価値効用の教師値バイアス
   * @param candidateValueTargetBias 候補価値効用の教師値バイアス
   * @param parentValueTargetMse 比較元価値効用の教師値 MSE
   * @param candidateValueTargetMse 候補価値効用の教師値 MSE
   * @param candidateMinusParentValueUtilityMean 候補−比較元価値効用平均
   * @param parentAdvantageMean 比較元スカラーアドバンテージ平均
   * @param candidateAdvantageMean 候補スカラーアドバンテージ平均
   * @param candidateMinusParentAdvantageMean 候補−比較元アドバンテージ平均
   * @param parentAdvantageAbsMean 比較元アドバンテージ絶対値平均
   * @param candidateAdvantageAbsMean 候補アドバンテージ絶対値平均
   * @param parentAdvantagePositiveRate 比較元アドバンテージ正値率
   * @param candidateAdvantagePositiveRate 候補アドバンテージ正値率
   * @param advantagePearson 比較元／候補アドバンテージの Pearson 相関
   * @param advantageSignDisagreementRate アドバンテージ符号不一致率
   * @param parentSelectedActionProbabilityMean 比較元の選択行動平均確率
   * @param candidateSelectedActionProbabilityMean 候補の選択行動平均確率
   * @param candidateMinusParentSelectedLogProbabilityMean 選択した対数確率差
   * @param parentGreedySelectedAgreementRate 比較元最大確率の行動を選ぶ方式と実選択の一致率
   * @param candidateGreedySelectedAgreementRate 候補最大確率の行動を選ぶ方式と実選択の一致率
   * @param parentCandidateGreedyDisagreementRate 比較元／候補最大確率の行動を選ぶ方式不一致率
   * @param symmetricKlMean 比較元／候補最終的な行動の確率分布の対称KLダイバージェンス平均
   */
  public record PairMetrics(
      long samples,
      double causalActorRate,
      double parentValueTargetBias,
      double candidateValueTargetBias,
      double parentValueTargetMse,
      double candidateValueTargetMse,
      double candidateMinusParentValueUtilityMean,
      double parentAdvantageMean,
      double candidateAdvantageMean,
      double candidateMinusParentAdvantageMean,
      double parentAdvantageAbsMean,
      double candidateAdvantageAbsMean,
      double parentAdvantagePositiveRate,
      double candidateAdvantagePositiveRate,
      Double advantagePearson,
      double advantageSignDisagreementRate,
      double parentSelectedActionProbabilityMean,
      double candidateSelectedActionProbabilityMean,
      double candidateMinusParentSelectedLogProbabilityMean,
      double parentGreedySelectedAgreementRate,
      double candidateGreedySelectedAgreementRate,
      double parentCandidateGreedyDisagreementRate,
      double symmetricKlMean) {}

  record ScoreContext(
      int kyokuIndex,
      int turn,
      int honba,
      int kyotaku,
      boolean allLast,
      int[] relativeScores,
      int currentRank,
      String scoreBucket,
      String nearestGapBucket) {}

  /**
   * 比較元／候補行動の不一致を一判断単位で再現するトレース。
   *
   * @param source 対局生成入力元ラベル
   * @param gameIndex 入力元内の対局インデックス
   * @param gameSeed 対局乱数シード
   * @param decisionIndex 対局内判断インデックス
   * @param gameId 対局 ID
   * @param playerSeat 学習器席
   * @param grpPrefixSteps 判断時点の GRP の局履歴長
   * @param kyokuIndex 局インデックス
   * @param turn 巡目
   * @param honba 本場
   * @param kyotaku 供託本数
   * @param allLast オーラスなら {@code true}
   * @param relativeScores 学習器席基準の4人点棒
   * @param currentRank 学習器の現在順位
   * @param scoreBucket 点差容量区分
   * @param nearestGapBucket 最寄り順位差容量区分
   * @param finalRank 学習器の終局順位
   * @param selectedActionId 実際に選択した行動 ID
   * @param selectedActionType 実際に選択した行動種類
   * @param selectedActionExecuted 優先度解決後に選択行動が実行されたか
   * @param parentGreedyActionId 比較元最大確率の行動を選ぶ方式行動 ID
   * @param parentGreedyActionType 比較元最大確率の行動を選ぶ方式行動種類
   * @param candidateGreedyActionId 候補最大確率の行動を選ぶ方式行動 ID
   * @param candidateGreedyActionType 候補最大確率の行動を選ぶ方式行動種類
   * @param parentSelectedActionProbability 比較元の選択行動確率
   * @param candidateSelectedActionProbability 候補の選択行動確率
   * @param symmetricKl 比較元／候補最終的な行動の確率分布の対称KLダイバージェンス
   * @param parentValueUtility 比較元の期待順位効用
   * @param candidateValueUtility 候補の期待順位効用
   * @param targetValueUtility 教師値の期待順位効用
   * @param sourceStoredAdvantage 入力元対局単位の学習データに保存されたアドバンテージ
   * @param parentScalarAdvantage 比較元のスカラーアドバンテージ
   * @param candidateScalarAdvantage 候補のスカラーアドバンテージ
   * @param candidateMinusParentScalarAdvantage 候補−比較元スカラーアドバンテージ
   */
  public record TraceRecord(
      String source,
      int gameIndex,
      long gameSeed,
      int decisionIndex,
      long gameId,
      int playerSeat,
      int grpPrefixSteps,
      int kyokuIndex,
      int turn,
      int honba,
      int kyotaku,
      boolean allLast,
      int[] relativeScores,
      int currentRank,
      String scoreBucket,
      String nearestGapBucket,
      int finalRank,
      int selectedActionId,
      Action.Type selectedActionType,
      boolean selectedActionExecuted,
      int parentGreedyActionId,
      Action.Type parentGreedyActionType,
      int candidateGreedyActionId,
      Action.Type candidateGreedyActionType,
      double parentSelectedActionProbability,
      double candidateSelectedActionProbability,
      double symmetricKl,
      double parentValueUtility,
      double candidateValueUtility,
      double targetValueUtility,
      float sourceStoredAdvantage,
      double parentScalarAdvantage,
      double candidateScalarAdvantage,
      double candidateMinusParentScalarAdvantage) {}

  private static final class SourceAccumulator {
    private final String source;
    private final BufferedWriter traceWriter;
    private final PairTotals overall = new PairTotals();
    private final Map<String, PairTotals> byAction = new LinkedHashMap<>();
    private final Map<String, PairTotals> byRank = new LinkedHashMap<>();
    private final Map<String, PairTotals> byScore = new LinkedHashMap<>();
    private final Map<String, PairTotals> byAllLastRank = new LinkedHashMap<>();
    private final Map<String, PairTotals> byAllLastRankGap = new LinkedHashMap<>();
    private final Map<String, Long> greedyTransitions = new LinkedHashMap<>();
    private final PriorityQueue<RankedTrace> top =
        new PriorityQueue<>(Comparator.comparingDouble(RankedTrace::severity));
    private float sourceStoredAdvantageMaxAbsError;
    private long samples;
    private long games;
    private final long[] finalRankCounts = new long[EpsilonDecisionConstants.PLAYERS];

    private SourceAccumulator(String source, BufferedWriter traceWriter) {
      this.source = source;
      this.traceWriter = traceWriter;
    }

    private void add(TraceRecord trace, float storedError) throws IOException {
      if (!source.equals(trace.source())) {
        throw new IOException("Trace source mismatch");
      }
      samples++;
      sourceStoredAdvantageMaxAbsError = Math.max(sourceStoredAdvantageMaxAbsError, storedError);
      overall.add(trace);
      add(byAction, trace.selectedActionType().name(), trace);
      add(byRank, "rank" + trace.currentRank(), trace);
      add(byScore, trace.scoreBucket(), trace);
      if (trace.allLast()) {
        String rank = "rank" + trace.currentRank();
        add(byAllLastRank, rank, trace);
        add(byAllLastRankGap, rank + "-gap-" + trace.nearestGapBucket(), trace);
      }
      greedyTransitions.merge(
          trace.parentGreedyActionType() + "->" + trace.candidateGreedyActionType(),
          1L,
          Math::addExact);
      double severity =
          Math.abs(trace.candidateMinusParentScalarAdvantage())
              + Math.sqrt(Math.max(0.0, trace.symmetricKl()));
      RankedTrace ranked = new RankedTrace(severity, trace);
      if (top.size() < TOP_EXAMPLES) {
        top.add(ranked);
      } else if (severity > top.peek().severity()) {
        top.remove();
        top.add(ranked);
      }
      traceWriter.write(LINE_GSON.toJson(trace));
      traceWriter.newLine();
    }

    private void finishGame(int finalRank) throws IOException {
      if (finalRank < 0 || finalRank >= finalRankCounts.length) {
        throw new IOException("Invalid paired-audit final rank: " + finalRank);
      }
      games++;
      finalRankCounts[finalRank]++;
      traceWriter.flush();
    }

    private SourceAudit toAudit(
        String sourceCheckpoint, int games, EpsilonDecisionArena.Metrics arenaMetrics) {
      if (this.games != games) {
        throw new IllegalStateException(
            "Paired-audit completed game mismatch: expected=" + games + " actual=" + this.games);
      }
      ArrayList<RankedTrace> ranked = new ArrayList<>(top);
      ranked.sort(Comparator.comparingDouble(RankedTrace::severity).reversed());
      double averageRank = 0.0;
      for (int rank = 0; rank < finalRankCounts.length; rank++) {
        averageRank += (rank + 1.0) * finalRankCounts[rank];
      }
      averageRank /= games;
      return new SourceAudit(
          source,
          sourceCheckpoint,
          games,
          samples,
          arenaMetrics.inferenceBatchCount(),
          arenaMetrics.inferenceRequestCount(),
          arenaMetrics.maxInferenceBatchSize(),
          sourceStoredAdvantageMaxAbsError,
          averageRank,
          finalRankCounts[0] / (double) games,
          finalRankCounts[finalRankCounts.length - 1] / (double) games,
          finalRankCounts.clone(),
          overall.toMetrics(),
          metrics(byAction),
          metrics(byRank),
          metrics(byScore),
          metrics(byAllLastRank),
          metrics(byAllLastRankGap),
          sortedCounts(greedyTransitions),
          ranked.stream().map(RankedTrace::trace).toList());
    }

    private static void add(Map<String, PairTotals> groups, String key, TraceRecord trace) {
      groups.computeIfAbsent(key, ignored -> new PairTotals()).add(trace);
    }

    private static Map<String, PairMetrics> metrics(Map<String, PairTotals> groups) {
      TreeMap<String, PairMetrics> sorted = new TreeMap<>();
      groups.forEach((key, value) -> sorted.put(key, value.toMetrics()));
      return sorted;
    }

    private static Map<String, Long> sortedCounts(Map<String, Long> counts) {
      return new TreeMap<>(counts);
    }
  }

  private record RankedTrace(double severity, TraceRecord trace) {}

  private static final class PairTotals {
    private long samples;
    private long eligible;
    private long parentPositive;
    private long candidatePositive;
    private long signDisagreement;
    private long parentGreedySelected;
    private long candidateGreedySelected;
    private long greedyDisagreement;
    private double parentValueError;
    private double candidateValueError;
    private double parentValueSquaredError;
    private double candidateValueSquaredError;
    private double valueDelta;
    private double parentAdvantage;
    private double candidateAdvantage;
    private double parentAdvantageSquared;
    private double candidateAdvantageSquared;
    private double advantageProduct;
    private double parentAdvantageAbs;
    private double candidateAdvantageAbs;
    private double parentSelectedProbability;
    private double candidateSelectedProbability;
    private double selectedLogProbabilityDelta;
    private double symmetricKl;

    private void add(TraceRecord trace) {
      samples++;
      if (trace.selectedActionExecuted()) {
        eligible++;
      }
      double parentError = trace.parentValueUtility() - trace.targetValueUtility();
      double candidateError = trace.candidateValueUtility() - trace.targetValueUtility();
      parentValueError += parentError;
      candidateValueError += candidateError;
      parentValueSquaredError += parentError * parentError;
      candidateValueSquaredError += candidateError * candidateError;
      valueDelta += trace.candidateValueUtility() - trace.parentValueUtility();
      double parentAdv = trace.parentScalarAdvantage();
      double candidateAdv = trace.candidateScalarAdvantage();
      parentAdvantage += parentAdv;
      candidateAdvantage += candidateAdv;
      parentAdvantageSquared += parentAdv * parentAdv;
      candidateAdvantageSquared += candidateAdv * candidateAdv;
      advantageProduct += parentAdv * candidateAdv;
      parentAdvantageAbs += Math.abs(parentAdv);
      candidateAdvantageAbs += Math.abs(candidateAdv);
      if (parentAdv > 0.0) {
        parentPositive++;
      }
      if (candidateAdv > 0.0) {
        candidatePositive++;
      }
      if ((parentAdv > 0.0) != (candidateAdv > 0.0)) {
        signDisagreement++;
      }
      parentSelectedProbability += trace.parentSelectedActionProbability();
      candidateSelectedProbability += trace.candidateSelectedActionProbability();
      selectedLogProbabilityDelta +=
          Math.log(Math.max(PROBABILITY_FLOOR, trace.candidateSelectedActionProbability()))
              - Math.log(Math.max(PROBABILITY_FLOOR, trace.parentSelectedActionProbability()));
      if (trace.parentGreedyActionId() == trace.selectedActionId()) {
        parentGreedySelected++;
      }
      if (trace.candidateGreedyActionId() == trace.selectedActionId()) {
        candidateGreedySelected++;
      }
      if (trace.parentGreedyActionId() != trace.candidateGreedyActionId()) {
        greedyDisagreement++;
      }
      symmetricKl += trace.symmetricKl();
    }

    private PairMetrics toMetrics() {
      if (samples <= 0L) {
        throw new IllegalStateException("Pair metrics require at least one sample");
      }
      double n = samples;
      double parentMean = parentAdvantage / n;
      double candidateMean = candidateAdvantage / n;
      Double pearson =
          pearson(
              samples,
              parentAdvantage,
              candidateAdvantage,
              parentAdvantageSquared,
              candidateAdvantageSquared,
              advantageProduct);
      return new PairMetrics(
          samples,
          eligible / n,
          parentValueError / n,
          candidateValueError / n,
          parentValueSquaredError / n,
          candidateValueSquaredError / n,
          valueDelta / n,
          parentMean,
          candidateMean,
          candidateMean - parentMean,
          parentAdvantageAbs / n,
          candidateAdvantageAbs / n,
          parentPositive / n,
          candidatePositive / n,
          pearson,
          signDisagreement / n,
          parentSelectedProbability / n,
          candidateSelectedProbability / n,
          selectedLogProbabilityDelta / n,
          parentGreedySelected / n,
          candidateGreedySelected / n,
          greedyDisagreement / n,
          symmetricKl / n);
    }

    private static Double pearson(
        long count, double sumX, double sumY, double sumXX, double sumYY, double sumXY) {
      if (count <= 1L) {
        return null;
      }
      double centeredXX = sumXX - sumX * sumX / count;
      double centeredYY = sumYY - sumY * sumY / count;
      if (!(centeredXX > 0.0 && centeredYY > 0.0)) {
        return null;
      }
      double centeredXY = sumXY - sumX * sumY / count;
      return centeredXY / Math.sqrt(centeredXX * centeredYY);
    }
  }
}

package com.epsilon.pico.ai.decision.benchmark;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtMemoryStats;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.DecisionTrainInFlightSpoolSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.GrpSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.GameState;
import com.epsilon.pico.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.pico.ai.decision.arena.EpsilonDecisionPopulationCollector;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.runtime.DecisionInferenceResult;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.pico.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.DecisionInferenceSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionDevicePipeline;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 自己対局から保存した入力を繰り返し使い、演算、転送、推論全体の処理速度を測る。
 *
 * <p>CREATE では学習側と対戦相手の入力を容量区分ごとに保存し、LOAD
 * では保存済みデータを検証して再利用する。計測中は対局進行と特徴量生成を省き、演算だけ、転送を含む計算、推論全体を別々に測定する。入力の採取用モデルは計測前に閉じ、実行計画や作業領域を計測用モデルへ引き継がない。
 */
public final class EpsilonDecisionInferenceReplayBenchmark {

  private static final Logger log =
      LoggerFactory.getLogger(EpsilonDecisionInferenceReplayBenchmark.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final int SCHEMA_VERSION = 6;
  private static final int CAPTURE_GAMES = 512;
  private static final int MINIMUM_WARMUP_BATCHES = 50;
  private static final int MINIMUM_MEASURED_BATCHES = 200;
  private static final int MAXIMUM_MEASURED_BATCHES = 10_000;
  private static final long TARGET_MEASUREMENT_NANOS = 5_000_000_000L;
  private static final int[] DEFAULT_BATCH_ROWS = {512, 1024, 1536, 2048, 3072, 4096};
  private static final DecisionBucket DISCARD_BUCKET = new DecisionBucket(16, 1);
  private static final DecisionBucket CALL_BUCKET = new DecisionBucket(4, 16);

  private EpsilonDecisionInferenceReplayBenchmark() {}

  /** 固定再生固定入力データの取得方法。 */
  public enum CorpusMode {
    /** 実自己対局から一度だけ採取し、新規固定入力データファイルとして公開する。 */
    CREATE,
    /** 既存固定入力データファイルを検証して読み込み、自己対局記録を行わない。 */
    LOAD;

    /** CLI文字列を大文字小文字を区別せず解析する。 */
    public static CorpusMode parse(String value) {
      try {
        return valueOf(value.toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException | NullPointerException e) {
        throw new IllegalArgumentException("corpus mode must be CREATE or LOAD: " + value, e);
      }
    }
  }

  /**
   * チェックポイントを変更せず、固定入力データの作成または読込と再生を実行してJSON レポートを書く。
   *
   * @param checkpointRoot Decision チェックポイントまたはその起点
   * @param reportFile 新規作成または置換するJSON レポート
   * @param corpusFile 新規作成または読込する固定ホスト固定入力データ
   * @param corpusMode 固定入力データを作成するか読み込むか
   * @param seedBase 記録自己対局の先頭乱数シード
   * @param repetitions 各測定点を繰り返す回数
   * @param targetRowsPerSecond 300 対局/s判定に必要な2 GPU合計行処理速度
   * @return レポートへ書いた構造化結果
   * @throws Exception チェックポイント、収集、GPU推論、レポート書込に失敗した場合
   */
  public static Report run(
      Path checkpointRoot,
      Path reportFile,
      Path corpusFile,
      CorpusMode corpusMode,
      long seedBase,
      int repetitions,
      double targetRowsPerSecond)
      throws Exception {
    return run(
        checkpointRoot,
        reportFile,
        corpusFile,
        corpusMode,
        seedBase,
        repetitions,
        targetRowsPerSecond,
        EpsilonSettings.defaults());
  }

  /** 起動時の設定スナップショットで固定入力データ記録と再生を実行する。 */
  public static Report run(
      Path checkpointRoot,
      Path reportFile,
      Path corpusFile,
      CorpusMode corpusMode,
      long seedBase,
      int repetitions,
      double targetRowsPerSecond,
      SettingsLoader snapshot)
      throws Exception {
    return run(
        checkpointRoot,
        reportFile,
        corpusFile,
        corpusMode,
        seedBase,
        repetitions,
        targetRowsPerSecond,
        DEFAULT_BATCH_ROWS,
        snapshot);
  }

  /**
   * チェックポイントを変更せず、指定した物理的なバッチ行列で固定再生を実行する。
   *
   * @param checkpointRoot Decision チェックポイントまたはその起点
   * @param reportFile 新規作成または置換するJSON レポート
   * @param corpusFile 新規作成または読込する固定ホスト固定入力データ
   * @param corpusMode 固定入力データを作成するか読み込むか
   * @param seedBase 記録自己対局の先頭乱数シード
   * @param repetitions 各測定点を繰り返す回数
   * @param targetRowsPerSecond 300 対局/s判定に必要な2 GPU合計行処理速度
   * @param batchRows 昇順かつ重複のないバッチ行数の列
   * @return レポートへ書いた構造化結果
   * @throws Exception チェックポイント、収集、GPU推論、レポート書込に失敗した場合
   */
  public static Report run(
      Path checkpointRoot,
      Path reportFile,
      Path corpusFile,
      CorpusMode corpusMode,
      long seedBase,
      int repetitions,
      double targetRowsPerSecond,
      int[] batchRows)
      throws Exception {
    return run(
        checkpointRoot,
        reportFile,
        corpusFile,
        corpusMode,
        seedBase,
        repetitions,
        targetRowsPerSecond,
        batchRows,
        EpsilonSettings.defaults());
  }

  /** 起動時の設定スナップショットで固定入力データ記録と再生を実行する。 */
  public static Report run(
      Path checkpointRoot,
      Path reportFile,
      Path corpusFile,
      CorpusMode corpusMode,
      long seedBase,
      int repetitions,
      double targetRowsPerSecond,
      int[] batchRows,
      SettingsLoader snapshot)
      throws Exception {
    int[] measuredBatchRows = validateBatchRows(batchRows);
    if ((repetitions != 1 && repetitions != 3) || !(targetRowsPerSecond > 0.0)) {
      throw new IllegalArgumentException(
          "repetitions must be 1 (preliminary) or 3 (family-wise inference), and target rows/s"
              + " must be positive");
    }
    Path checkpoint = EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpointRoot);
    if (checkpoint == null) {
      throw new IOException("Decision replay benchmark checkpoint not found: " + checkpointRoot);
    }
    Path normalizedReport = reportFile.toAbsolutePath().normalize();
    Path reportParent = normalizedReport.getParent();
    if (reportParent != null) {
      Files.createDirectories(reportParent);
    }
    try (var context = new DecisionExecutionContext()) {
      ArrayList<Measurement> measurements = new ArrayList<>();
      PreparedCorpora prepared =
          prepareCorpora(
              checkpointRoot,
              checkpoint,
              normalizedReport,
              corpusFile,
              corpusMode,
              seedBase,
              context,
              snapshot);
      EpsilonDecisionInferenceReplayCorpusStore.Bundle corpusBundle = prepared.bundle();
      EpsilonDecisionInferenceReplayCorpus.Corpus actorCorpus = corpusBundle.actor();
      EpsilonDecisionInferenceReplayCorpus.Corpus opponentCorpus = corpusBundle.opponent();
      String actorDevices = prepared.actorDevices();
      String opponentDevices = prepared.opponentDevices();
      Set<Device> replayDeviceSet = prepared.replayDevices();
      requireMajorBuckets(actorCorpus, "actor");
      requireMajorBuckets(opponentCorpus, "opponent");

      // requested 行ごとに新しいモデルコンテキストを開き、Fusion 収容上限キャッシュを測定点間で共有しない。
      for (int repetition = 0; repetition < repetitions; repetition++) {
        List<Mode> modeOrder = rotatedModes(repetition);
        for (int requestedBatchRows : rotatedBatchRows(measuredBatchRows, repetition)) {
          try (EpsilonDecisionEvaluatorFactory.Handle actor =
                  EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                      checkpoint, context, snapshot);
              EpsilonDecisionEvaluatorFactory.Handle opponent =
                  EpsilonDecisionEvaluatorFactory.openPolicyCheckpointEvaluator(
                      checkpoint, context, snapshot)) {
            EpsilonDecisionInferenceServer[] actorServers = actor.serversForReplayBenchmark();
            EpsilonDecisionInferenceServer[] opponentServers = opponent.serversForReplayBenchmark();
            requireTwoDeviceReplay(actorServers, opponentServers);
            requireDeviceSet("actor", replayDeviceSet, replayDevices(actorServers));
            requireDeviceSet("opponent", replayDeviceSet, replayDevices(opponentServers));
            measureScope(
                measurements,
                repetition,
                Suite.PURE,
                DISCARD_BUCKET,
                requestedBatchRows,
                modeOrder,
                actorCorpus,
                opponentCorpus,
                actorServers,
                opponentServers);
            if (requestedBatchRows <= preferredRows(actorServers, opponentServers, CALL_BUCKET)) {
              measureScope(
                  measurements,
                  repetition,
                  Suite.PURE,
                  CALL_BUCKET,
                  requestedBatchRows,
                  modeOrder,
                  actorCorpus,
                  opponentCorpus,
                  actorServers,
                  opponentServers);
            } else {
              for (Mode mode : modeOrder) {
                measurements.add(
                    Measurement.skipped(
                        repetition,
                        Suite.PURE,
                        mode,
                        bucketName(CALL_BUCKET),
                        requestedBatchRows,
                        "exceeds production multi-transition batch limit"));
              }
            }
            measureScope(
                measurements,
                repetition,
                Suite.MIXED,
                null,
                requestedBatchRows,
                modeOrder,
                actorCorpus,
                opponentCorpus,
                actorServers,
                opponentServers);
          }
        }
      }

      List<ModeSummary> summaries = summarize(measurements, repetitions, measuredBatchRows);
      Verdict verdict = verdict(summaries, repetitions, targetRowsPerSecond);
      DecisionInferenceSettings settings = snapshot.bind(DecisionInferenceSettings.class);
      DecisionInferenceFusionSettings fusionSettings =
          snapshot.bind(DecisionInferenceFusionSettings.class);
      Report report =
          new Report(
              SCHEMA_VERSION,
              checkpoint.toAbsolutePath().normalize().toString(),
              corpusBundle.seedBase(),
              corpusBundle.captureGames(),
              repetitions,
              targetRowsPerSecond,
              java.util.Arrays.stream(measuredBatchRows).boxed().toList(),
              new EffectiveSettings(
                  settings.computePrecision(),
                  settings.maxBatch(),
                  settings.multiTransitionMaxBatch(),
                  settings.slotsPerDevice(),
                  settings.readyBatchesPerDevice(),
                  fusionSettings,
                  actorDevices,
                  opponentDevices),
              new CorpusFileReport(
                  corpusMode,
                  corpusFile.toAbsolutePath().normalize().toString(),
                  corpusBundle.formatVersion(),
                  corpusBundle.schemaFingerprint(),
                  corpusBundle.seedBase(),
                  corpusBundle.captureGames()),
              corpusReport(actorCorpus.report()),
              corpusReport(opponentCorpus.report()),
              List.copyOf(measurements),
              summaries,
              modeDefinitions(),
              verdict);
      writeAtomically(normalizedReport, report);
      return report;
    }
  }

  private static PreparedCorpora prepareCorpora(
      Path checkpointRoot,
      Path checkpoint,
      Path reportFile,
      Path corpusFile,
      CorpusMode mode,
      long seedBase,
      DecisionExecutionContext context,
      SettingsLoader snapshot)
      throws Exception {
    Path normalizedCorpus = corpusFile.toAbsolutePath().normalize();
    if (mode == CorpusMode.LOAD) {
      EpsilonDecisionInferenceReplayCorpusStore.Bundle loaded =
          EpsilonDecisionInferenceReplayCorpusStore.load(normalizedCorpus);
      if (loaded.seedBase() != seedBase) {
        throw new IllegalArgumentException(
            "Decision replay corpus seed mismatch: corpus="
                + loaded.seedBase()
                + " requested="
                + seedBase);
      }
      var inferenceDevices =
          NetworkFactory.getInferenceDevices(snapshot.bind(DeviceSettings.class));
      Set<Device> devices = Set.copyOf(inferenceDevices.asList());
      String deviceDescription = inferenceDevices.toString();
      log.info(
          "Loaded fixed Decision inference corpus: file={} actorRows={} opponentRows={}",
          normalizedCorpus,
          loaded.actor().report().retainedRows(),
          loaded.opponent().report().retainedRows());
      return new PreparedCorpora(loaded, deviceDescription, deviceDescription, devices);
    }
    if (Files.exists(normalizedCorpus)) {
      throw new java.nio.file.FileAlreadyExistsException(normalizedCorpus.toString());
    }
    Path reportParent = reportFile.getParent();
    Path scratchRoot =
        (reportParent == null ? Path.of(".") : reportParent)
            .resolve(reportFile.getFileName() + ".capture-inflight");
    Files.createDirectories(scratchRoot);
    EpsilonDecisionInferenceReplayCorpus.Recorder actorRecorder =
        new EpsilonDecisionInferenceReplayCorpus.Recorder(
            EpsilonDecisionInferenceReplayCorpus.Role.ACTOR_POLICY_AND_VALUE);
    EpsilonDecisionInferenceReplayCorpus.Recorder opponentRecorder =
        new EpsilonDecisionInferenceReplayCorpus.Recorder(
            EpsilonDecisionInferenceReplayCorpus.Role.OPPONENT_POLICY_ONLY);
    String actorDevices;
    String opponentDevices;
    Set<Device> replayDevices;
    try (EpsilonDecisionEvaluatorFactory.Handle captureActor =
            EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(checkpoint, context, snapshot);
        EpsilonDecisionEvaluatorFactory.Handle captureOpponent =
            EpsilonDecisionEvaluatorFactory.openPolicyCheckpointEvaluator(
                checkpoint, context, snapshot);
        EpsilonGrpTrainingSession grp =
            snapshot.bind(GrpSettings.class).enabled()
                ? EpsilonGrpTrainingSession.openInferenceOnly(
                    checkpointRoot,
                    NetworkFactory.getGrpDevices(snapshot.bind(DeviceSettings.class)).primary(),
                    snapshot)
                : EpsilonGrpTrainingSession.disabled(checkpointRoot.resolve("grp"));
        EpsilonDecisionTrajectoryPayloadStore payloadStore =
            EpsilonDecisionTrajectoryPayloadStore.fromSettings(
                scratchRoot, false, snapshot.bind(DecisionTrainInFlightSpoolSettings.class))) {
      EpsilonDecisionInferenceServer[] actorServers = captureActor.serversForReplayBenchmark();
      EpsilonDecisionInferenceServer[] opponentServers =
          captureOpponent.serversForReplayBenchmark();
      requireTwoDeviceReplay(actorServers, opponentServers);
      actorDevices = captureActor.devices();
      opponentDevices = captureOpponent.devices();
      replayDevices = Set.copyOf(replayDevices(actorServers));
      setObserver(actorServers, actorRecorder);
      setObserver(opponentServers, opponentRecorder);
      try {
        capture(
            checkpointRoot,
            seedBase,
            captureActor.evaluator(),
            captureOpponent.evaluator(),
            grp,
            payloadStore,
            snapshot);
      } finally {
        clearObserver(actorServers);
        clearObserver(opponentServers);
      }
    }
    EpsilonDecisionInferenceReplayCorpus.Corpus actorCorpus = actorRecorder.freeze();
    EpsilonDecisionInferenceReplayCorpus.Corpus opponentCorpus = opponentRecorder.freeze();
    requireMajorBuckets(actorCorpus, "actor");
    requireMajorBuckets(opponentCorpus, "opponent");
    EpsilonDecisionInferenceReplayCorpusStore.Bundle created =
        EpsilonDecisionInferenceReplayCorpusStore.create(
            normalizedCorpus, actorCorpus, opponentCorpus, seedBase, CAPTURE_GAMES);
    log.info(
        "Created fixed Decision inference corpus: file={} actorRows={} opponentRows={}",
        normalizedCorpus,
        actorCorpus.report().retainedRows(),
        opponentCorpus.report().retainedRows());
    return new PreparedCorpora(created, actorDevices, opponentDevices, replayDevices);
  }

  private static void capture(
      Path checkpointRoot,
      long seedBase,
      EpsilonDecisionEvaluator actor,
      EpsilonDecisionEvaluator opponent,
      EpsilonGrpTrainingSession grp,
      EpsilonDecisionTrajectoryPayloadStore payloadStore,
      SettingsLoader snapshot)
      throws Exception {
    EpsilonDecisionPlayer.RolloutConfig actorRollout =
        EpsilonDecisionPlayer.RolloutConfig.fromSettings(snapshot);
    EpsilonDecisionPlayer.RolloutConfig opponentRollout =
        EpsilonDecisionPlayer.RolloutConfig.opponentFromSettings(snapshot);
    EpsilonDecisionPopulationCollector.requireSelectedPgActor(actorRollout);
    log.info(
        "Capturing fixed Decision inference corpus: games={} seedBase={} checkpoint={}",
        CAPTURE_GAMES,
        seedBase,
        checkpointRoot);
    EpsilonDecisionPopulationCollector.collect(
        EpsilonDecisionPopulationCollector.Request.standard(
                CAPTURE_GAMES,
                seedBase,
                actor,
                2L,
                opponentIds(),
                ignored -> opponent,
                actorRollout,
                opponentRollout,
                (gameIndex, gameSeed, game) -> {},
                payloadStore)
            .withGrpInference(grp.inference()),
        snapshot);
  }

  private static long[][] opponentIds() {
    long[][] result = new long[GameState.NUM_PLAYERS][GameState.NUM_PLAYERS - 1];
    for (long[] row : result) {
      java.util.Arrays.fill(row, 1L);
    }
    return result;
  }

  private static void setObserver(
      EpsilonDecisionInferenceServer[] servers,
      EpsilonDecisionInferenceReplayCorpus.Recorder recorder) {
    for (EpsilonDecisionInferenceServer server : servers) {
      server.setPhysicalBatchObserver(recorder);
    }
  }

  private static void clearObserver(EpsilonDecisionInferenceServer[] servers) {
    for (EpsilonDecisionInferenceServer server : servers) {
      server.clearPhysicalBatchObserver();
    }
  }

  private static void requireMajorBuckets(
      EpsilonDecisionInferenceReplayCorpus.Corpus corpus, String role) {
    if (!corpus.hasBucket(DISCARD_BUCKET) || !corpus.hasBucket(CALL_BUCKET)) {
      throw new IllegalStateException(
          "fixed replay capture did not observe both major buckets for " + role);
    }
  }

  private static void requireTwoDeviceReplay(
      EpsilonDecisionInferenceServer[] actorServers,
      EpsilonDecisionInferenceServer[] opponentServers) {
    LinkedHashSet<Device> actorDevices = replayDevices(actorServers);
    LinkedHashSet<Device> opponentDevices = replayDevices(opponentServers);
    boolean allGpu =
        actorDevices.stream().allMatch(Device::isGpu)
            && opponentDevices.stream().allMatch(Device::isGpu);
    if (actorServers.length != 2
        || opponentServers.length != 2
        || actorDevices.size() != 2
        || !actorDevices.equals(opponentDevices)
        || !allGpu) {
      throw new IllegalStateException(
          "181k rows/s replay protocol requires matching replicas on exactly two distinct GPUs:"
              + " actor="
              + actorDevices
              + " opponent="
              + opponentDevices);
    }
  }

  private static LinkedHashSet<Device> replayDevices(EpsilonDecisionInferenceServer[] servers) {
    LinkedHashSet<Device> devices = new LinkedHashSet<>();
    for (EpsilonDecisionInferenceServer server : servers) {
      devices.add(server.device());
    }
    return devices;
  }

  private static void requireDeviceSet(String role, Set<Device> expected, Set<Device> actual) {
    if (!expected.equals(actual)) {
      throw new IllegalStateException(
          "fixed replay device set changed for "
              + role
              + ": expected="
              + expected
              + " actual="
              + actual);
    }
  }

  private static CorpusReport corpusReport(EpsilonDecisionInferenceReplayCorpus.Report source) {
    LinkedHashMap<String, CorpusBucketReport> buckets = new LinkedHashMap<>();
    source
        .buckets()
        .forEach(
            (bucket, statistics) ->
                buckets.put(
                    bucketName(bucket),
                    new CorpusBucketReport(
                        statistics.observedRows(),
                        statistics.retainedRows(),
                        statistics.targetRows())));
    return new CorpusReport(
        source.role(),
        source.observedRows(),
        source.retainedRows(),
        source.sampling(),
        Map.copyOf(buckets));
  }

  private static String bucketName(DecisionBucket bucket) {
    return bucket.legalActionCapacity() + "x" + bucket.actionTransitionCapacity();
  }

  private static List<Mode> rotatedModes(int repetition) {
    Mode[] values = Mode.values();
    ArrayList<Mode> order = new ArrayList<>(values.length);
    for (int offset = 0; offset < values.length; offset++) {
      order.add(values[(repetition + offset) % values.length]);
    }
    return order;
  }

  /** CLIのcomma-separated 行列を解析する。 */
  public static int[] parseBatchRowsCsv(String csv) {
    if (csv == null || csv.isBlank()) {
      throw new IllegalArgumentException("batch rows CSV must not be blank");
    }
    String[] tokens = csv.split(",", -1);
    int[] rows = new int[tokens.length];
    for (int index = 0; index < tokens.length; index++) {
      String token = tokens[index].trim();
      if (token.isEmpty()) {
        throw new IllegalArgumentException("batch rows CSV contains an empty value: " + csv);
      }
      try {
        rows[index] = Integer.parseInt(token);
      } catch (NumberFormatException e) {
        throw new IllegalArgumentException("invalid batch row value: " + token, e);
      }
    }
    return validateBatchRows(rows);
  }

  /** 既定行列を3反復で均等に前後させる従来席順を入れ替えた対局。 */
  static int[] rotatedBatchRows(int repetition) {
    return rotatedBatchRows(DEFAULT_BATCH_ROWS, repetition);
  }

  /** 指定行列を3反復の測定位置へ分散する。 */
  static int[] rotatedBatchRows(int[] batchRows, int repetition) {
    int[] validated = validateBatchRows(batchRows);
    if (repetition < 0) {
      throw new IllegalArgumentException("repetition must not be negative");
    }
    int[] rows = new int[validated.length];
    int offset = Math.floorMod((validated.length * repetition) / 3, validated.length);
    for (int index = 0; index < rows.length; index++) {
      rows[index] = validated[(index + offset) % validated.length];
    }
    return rows;
  }

  private static int[] validateBatchRows(int[] batchRows) {
    if (batchRows == null || batchRows.length == 0) {
      throw new IllegalArgumentException("batch rows must not be empty");
    }
    int[] validated = batchRows.clone();
    int previous = 0;
    for (int row : validated) {
      if (row <= previous) {
        throw new IllegalArgumentException(
            "batch rows must be positive, strictly increasing, and unique");
      }
      previous = row;
    }
    return validated;
  }

  private static void measureScope(
      List<Measurement> destination,
      int repetition,
      Suite suite,
      DecisionBucket pureBucket,
      int requestedBatchRows,
      List<Mode> modes,
      EpsilonDecisionInferenceReplayCorpus.Corpus actorCorpus,
      EpsilonDecisionInferenceReplayCorpus.Corpus opponentCorpus,
      EpsilonDecisionInferenceServer[] actorServers,
      EpsilonDecisionInferenceServer[] opponentServers) {
    List<Variant> variants =
        variants(
            suite,
            pureBucket,
            requestedBatchRows,
            actorCorpus,
            opponentCorpus,
            actorServers,
            opponentServers);
    for (Mode mode : modes) {
      Measurement measurement =
          measure(
              repetition,
              suite,
              pureBucket,
              requestedBatchRows,
              mode,
              variants,
              actorServers,
              opponentServers);
      destination.add(measurement);
      log.info(
          "Decision fixed replay: repetition={} suite={} bucket={} mode={} requestedRows={} "
              + "physicalRows={} batches={} rows={} elapsedMs={} rowsPerSecond={} "
              + "allocatorMemory={}",
          repetition,
          suite,
          pureBucket == null ? "mixed" : bucketName(pureBucket),
          mode,
          requestedBatchRows,
          measurement.physicalBatchHistogram(),
          measurement.measuredBatches(),
          measurement.rows(),
          measurement.elapsedNanos() / 1_000_000.0,
          String.format(Locale.ROOT, "%.3f", measurement.rowsPerSecond()),
          measurement.allocatorMemory());
    }
  }

  private static List<Variant> variants(
      Suite suite,
      DecisionBucket pureBucket,
      int requestedBatchRows,
      EpsilonDecisionInferenceReplayCorpus.Corpus actorCorpus,
      EpsilonDecisionInferenceReplayCorpus.Corpus opponentCorpus,
      EpsilonDecisionInferenceServer[] actorServers,
      EpsilonDecisionInferenceServer[] opponentServers) {
    ArrayList<Variant> variants = new ArrayList<>();
    if (suite == Suite.PURE) {
      addVariant(
          variants,
          EpsilonDecisionInferenceReplayCorpus.Role.ACTOR_POLICY_AND_VALUE,
          actorCorpus,
          pureBucket,
          requestedBatchRows,
          actorServers,
          actorCorpus.observedRows(pureBucket));
      addVariant(
          variants,
          EpsilonDecisionInferenceReplayCorpus.Role.OPPONENT_POLICY_ONLY,
          opponentCorpus,
          pureBucket,
          requestedBatchRows,
          opponentServers,
          opponentCorpus.observedRows(pureBucket));
    } else {
      Set<DecisionBucket> buckets = new LinkedHashSet<>();
      buckets.addAll(actorCorpus.buckets());
      buckets.addAll(opponentCorpus.buckets());
      ArrayList<DecisionBucket> ordered = new ArrayList<>(buckets);
      ordered.sort(
          Comparator.comparingInt(DecisionBucket::actionTransitionCapacity)
              .thenComparingInt(DecisionBucket::legalActionCapacity));
      for (DecisionBucket bucket : ordered) {
        if (actorCorpus.hasBucket(bucket)) {
          addVariant(
              variants,
              EpsilonDecisionInferenceReplayCorpus.Role.ACTOR_POLICY_AND_VALUE,
              actorCorpus,
              bucket,
              requestedBatchRows,
              actorServers,
              actorCorpus.observedRows(bucket));
        }
        if (opponentCorpus.hasBucket(bucket)) {
          addVariant(
              variants,
              EpsilonDecisionInferenceReplayCorpus.Role.OPPONENT_POLICY_ONLY,
              opponentCorpus,
              bucket,
              requestedBatchRows,
              opponentServers,
              opponentCorpus.observedRows(bucket));
        }
      }
    }
    if (variants.isEmpty()) {
      throw new IllegalStateException("fixed replay workload has no captured variants");
    }
    return List.copyOf(variants);
  }

  private static void addVariant(
      List<Variant> destination,
      EpsilonDecisionInferenceReplayCorpus.Role role,
      EpsilonDecisionInferenceReplayCorpus.Corpus corpus,
      DecisionBucket bucket,
      int requestedBatchRows,
      EpsilonDecisionInferenceServer[] servers,
      long observedRows) {
    if (observedRows <= 0L) {
      return;
    }
    int rows = Math.min(requestedBatchRows, preferredRows(servers, bucket));
    destination.add(new Variant(role, bucket, rows, observedRows, corpus.batch(bucket, rows)));
  }

  private static int preferredRows(
      EpsilonDecisionInferenceServer[] actorServers,
      EpsilonDecisionInferenceServer[] opponentServers,
      DecisionBucket bucket) {
    return Math.min(preferredRows(actorServers, bucket), preferredRows(opponentServers, bucket));
  }

  private static int preferredRows(
      EpsilonDecisionInferenceServer[] servers, DecisionBucket bucket) {
    int rows = Integer.MAX_VALUE;
    for (EpsilonDecisionInferenceServer server : servers) {
      rows = Math.min(rows, server.preferredBatchSize(bucket));
    }
    return rows;
  }

  private static Measurement measure(
      int repetition,
      Suite suite,
      DecisionBucket pureBucket,
      int requestedBatchRows,
      Mode mode,
      List<Variant> variants,
      EpsilonDecisionInferenceServer[] actorServers,
      EpsilonDecisionInferenceServer[] opponentServers) {
    try (ReplayExecution execution =
        new ReplayExecution(mode, variants, actorServers, opponentServers)) {
      WeightedSequence warmup = new WeightedSequence(variants);
      long warmupNanos = execution.run(warmup, MINIMUM_WARMUP_BATCHES).elapsedNanos();
      int measuredBatches = measuredBatchCount(warmupNanos);
      WeightedSequence measured = new WeightedSequence(variants);
      settleHeapBeforeMeasurement();
      List<Device> devices = List.copyOf(replayDevices(actorServers));
      PtEngine engine = (PtEngine) Engine.getInstance();
      Map<Device, PtMemoryStats> memoryBefore = memoryStats(engine, devices);
      devices.forEach(engine::resetPeakMemoryStats);
      GcSnapshot beforeGc = GcSnapshot.current();
      RunResult run = execution.run(measured, measuredBatches);
      GcSnapshot afterGc = GcSnapshot.current();
      Map<String, AllocatorMemoryMeasurement> allocatorMemory =
          allocatorMemory(memoryBefore, memoryStats(engine, devices));
      double maximumMixError = maximumMixError(variants, run);
      double mixTolerance = maximumMixTolerance(variants, run.rows());
      if (maximumMixError > mixTolerance) {
        throw new IllegalStateException(
            "fixed replay variant mix drifted from capture: error="
                + maximumMixError
                + " tolerance="
                + mixTolerance
                + " actual="
                + run.variantRows());
      }
      return new Measurement(
          repetition,
          suite,
          mode,
          pureBucket == null ? "mixed" : bucketName(pureBucket),
          requestedBatchRows,
          run.batchHistogram(),
          run.roleRows(),
          run.bucketRows(),
          run.variantRows(),
          run.deviceRows(),
          maximumMixError,
          mixTolerance,
          MINIMUM_WARMUP_BATCHES,
          measuredBatches,
          run.rows(),
          run.elapsedNanos(),
          1_000_000_000.0 * run.rows() / run.elapsedNanos(),
          allocatorMemory,
          Math.max(0L, afterGc.collections() - beforeGc.collections()),
          Math.max(0L, afterGc.millis() - beforeGc.millis()),
          MeasurementStatus.MEASURED,
          null);
    }
  }

  private static void settleHeapBeforeMeasurement() {
    System.gc();
    System.runFinalization();
    System.gc();
  }

  private static Map<Device, PtMemoryStats> memoryStats(PtEngine engine, List<Device> devices) {
    LinkedHashMap<Device, PtMemoryStats> result = new LinkedHashMap<>();
    for (Device device : devices) {
      result.put(device, engine.getMemoryStats(device));
    }
    return result;
  }

  private static Map<String, AllocatorMemoryMeasurement> allocatorMemory(
      Map<Device, PtMemoryStats> before, Map<Device, PtMemoryStats> after) {
    LinkedHashMap<String, AllocatorMemoryMeasurement> result = new LinkedHashMap<>();
    before.forEach(
        (device, baseline) -> {
          PtMemoryStats measured = after.get(device);
          result.put(
              device.toString(),
              new AllocatorMemoryMeasurement(
                  baseline.getAllocatedBytes(),
                  measured.getAllocatedBytes(),
                  measured.getPeakAllocatedBytes(),
                  baseline.getActiveBytes(),
                  measured.getActiveBytes(),
                  measured.getPeakActiveBytes(),
                  baseline.getReservedBytes(),
                  measured.getReservedBytes(),
                  measured.getPeakReservedBytes()));
        });
    return Collections.unmodifiableMap(result);
  }

  static double maximumMixError(List<Variant> variants, RunResult run) {
    double totalWeight = variants.stream().mapToDouble(Variant::weight).sum();
    double maximum = 0.0;
    for (Variant variant : variants) {
      double expected = variant.weight() / totalWeight;
      double actual = (double) run.variantRows().getOrDefault(variant.key(), 0L) / run.rows();
      maximum = Math.max(maximum, Math.abs(actual - expected));
    }
    return maximum;
  }

  static double maximumMixTolerance(List<Variant> variants, long measuredRows) {
    int maximumQuantum = variants.stream().mapToInt(Variant::rows).max().orElseThrow();
    return Math.max(0.01, 2.0 * maximumQuantum / measuredRows);
  }

  private static int measuredBatchCount(long warmupNanos) {
    long estimated =
        warmupNanos <= 0L
            ? MINIMUM_MEASURED_BATCHES
            : Math.ceilDiv(
                Math.multiplyExact(TARGET_MEASUREMENT_NANOS, MINIMUM_WARMUP_BATCHES), warmupNanos);
    return (int) Math.max(MINIMUM_MEASURED_BATCHES, Math.min(MAXIMUM_MEASURED_BATCHES, estimated));
  }

  private static final class ReplayExecution implements AutoCloseable {
    private final Mode mode;
    private final EpsilonDecisionInferenceServer[] actorServers;
    private final EpsilonDecisionInferenceServer[] opponentServers;
    private final IdentityHashMap<
            EpsilonDecisionInferenceServer,
            Map<Variant, EpsilonDecisionInferenceServer.FixedReplay>>
        computeReplays = new IdentityHashMap<>();
    private final AtomicInteger nextActor = new AtomicInteger();
    private final AtomicInteger nextOpponent = new AtomicInteger();

    private ReplayExecution(
        Mode mode,
        List<Variant> variants,
        EpsilonDecisionInferenceServer[] actorServers,
        EpsilonDecisionInferenceServer[] opponentServers) {
      this.mode = mode;
      this.actorServers = actorServers;
      this.opponentServers = opponentServers;
      if (mode == Mode.COMPUTE_ONLY) {
        prepareComputeReplays(
            variants,
            EpsilonDecisionInferenceReplayCorpus.Role.ACTOR_POLICY_AND_VALUE,
            actorServers);
        prepareComputeReplays(
            variants,
            EpsilonDecisionInferenceReplayCorpus.Role.OPPONENT_POLICY_ONLY,
            opponentServers);
      }
    }

    private void prepareComputeReplays(
        List<Variant> variants,
        EpsilonDecisionInferenceReplayCorpus.Role role,
        EpsilonDecisionInferenceServer[] servers) {
      for (EpsilonDecisionInferenceServer server : servers) {
        HashMap<Variant, EpsilonDecisionInferenceServer.FixedReplay> byVariant = new HashMap<>();
        for (Variant variant : variants) {
          if (variant.role() != role) {
            continue;
          }
          byVariant.put(variant, server.openFixedReplay(variant.batch()));
        }
        computeReplays.put(server, byVariant);
      }
    }

    private RunResult run(WeightedSequence sequence, int batches) {
      ArrayList<CompletableFuture<Void>> futures = new ArrayList<>(batches);
      LinkedHashMap<String, Long> histogram = new LinkedHashMap<>();
      EnumMap<EpsilonDecisionInferenceReplayCorpus.Role, Long> roleRows =
          new EnumMap<>(EpsilonDecisionInferenceReplayCorpus.Role.class);
      LinkedHashMap<String, Long> bucketRows = new LinkedHashMap<>();
      LinkedHashMap<String, Long> variantRows = new LinkedHashMap<>();
      LinkedHashMap<String, Long> deviceRows = new LinkedHashMap<>();
      long rows = 0L;
      long started = System.nanoTime();
      for (int index = 0; index < batches; index++) {
        Variant variant = sequence.next();
        ReplayTarget target = acquireTarget(variant.role());
        CompletableFuture<Void> future = submit(target, variant);
        futures.add(future);
        rows += variant.rows();
        histogram.merge(bucketName(variant.bucket()) + ":" + variant.rows(), 1L, Long::sum);
        roleRows.merge(variant.role(), (long) variant.rows(), Long::sum);
        bucketRows.merge(bucketName(variant.bucket()), (long) variant.rows(), Long::sum);
        variantRows.merge(variant.key(), (long) variant.rows(), Long::sum);
        deviceRows.merge(
            variant.role() + "@" + target.server().device(), (long) variant.rows(), Long::sum);
      }
      CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).join();
      return new RunResult(
          rows,
          System.nanoTime() - started,
          Map.copyOf(histogram),
          Map.copyOf(roleRows),
          Map.copyOf(bucketRows),
          Map.copyOf(variantRows),
          Map.copyOf(deviceRows));
    }

    private ReplayTarget acquireTarget(EpsilonDecisionInferenceReplayCorpus.Role role) {
      EpsilonDecisionInferenceServer[] servers = isActor(role) ? actorServers : opponentServers;
      AtomicInteger cursor = isActor(role) ? nextActor : nextOpponent;
      while (true) {
        int first = Math.floorMod(cursor.getAndIncrement(), servers.length);
        CompletableFuture<?>[] available = new CompletableFuture<?>[servers.length];
        int preferred = first;
        long preferredWork = servers[first].pendingEstimatedWork();
        for (int offset = 0; offset < servers.length; offset++) {
          int index = (first + offset) % servers.length;
          available[index] = servers[index].hostReadyCellAvailable();
          long work = servers[index].pendingEstimatedWork();
          if (work < preferredWork) {
            preferred = index;
            preferredWork = work;
          }
        }
        for (int offset = 0; offset < servers.length; offset++) {
          int index = (preferred + offset) % servers.length;
          DecisionDevicePipeline.HostReadyCell cell = servers[index].acquireHostReadyCell();
          if (cell != null) {
            return new ReplayTarget(servers[index], cell);
          }
        }
        CompletableFuture.anyOf(available).join();
      }
    }

    private CompletableFuture<Void> submit(ReplayTarget target, Variant variant) {
      boolean consumed = false;
      try {
        CompletableFuture<Void> result =
            switch (mode) {
              case COMPUTE_ONLY ->
                  computeReplays
                      .get(target.server())
                      .get(variant)
                      .submitCompute(target.hostReadyCell());
              case GPU_PIPELINE ->
                  target
                      .server()
                      .submitPipelineOnly(
                          DecisionHostBatch.RowBatch.of(
                              variant.batch().sliceRows(0, variant.batch().size())),
                          target.hostReadyCell());
              case FULL_INFERENCE -> submitFull(target, variant.batch());
            };
        consumed = true;
        return result;
      } finally {
        if (!consumed) {
          target.hostReadyCell().abortIfActive();
        }
      }
    }

    private static CompletableFuture<Void> submitFull(
        ReplayTarget target, DecisionHostBatch batch) {
      return target
          .server()
          .submitToRing(
              DecisionHostBatch.RowBatch.of(batch.sliceRows(0, batch.size())),
              DecisionInferenceResult.Kind.PREDICTIONS,
              target.hostReadyCell())
          .thenAccept(ignored -> {});
    }

    @Override
    public void close() {
      Throwable failure = null;
      for (Map<Variant, EpsilonDecisionInferenceServer.FixedReplay> byVariant :
          computeReplays.values()) {
        for (EpsilonDecisionInferenceServer.FixedReplay replay : byVariant.values()) {
          try {
            replay.close();
          } catch (Throwable closeFailure) {
            if (failure == null) {
              failure = closeFailure;
            } else {
              failure.addSuppressed(closeFailure);
            }
          }
        }
      }
      computeReplays.clear();
      if (failure instanceof RuntimeException runtimeException) {
        throw runtimeException;
      }
      if (failure instanceof Error error) {
        throw error;
      }
      if (failure != null) {
        throw new IllegalStateException(failure);
      }
    }
  }

  private static boolean isActor(EpsilonDecisionInferenceReplayCorpus.Role role) {
    return role == EpsilonDecisionInferenceReplayCorpus.Role.ACTOR_POLICY_AND_VALUE;
  }

  /** 行比率に応じた仮想完了時刻が最小の種類を選ぶ決定論的スケジューラー。 */
  static final class WeightedSequence {
    private final List<Variant> variants;
    private final double[] nextFinish;
    private final double[] quanta;

    WeightedSequence(List<Variant> variants) {
      this.variants = variants;
      nextFinish = new double[variants.size()];
      quanta = new double[variants.size()];
      for (int index = 0; index < variants.size(); index++) {
        Variant variant = variants.get(index);
        quanta[index] = (double) variant.rows() / variant.weight();
        nextFinish[index] = 0.5 * quanta[index];
      }
    }

    Variant next() {
      int selected = 0;
      for (int index = 1; index < variants.size(); index++) {
        if (nextFinish[index] < nextFinish[selected]) {
          selected = index;
        }
      }
      Variant result = variants.get(selected);
      nextFinish[selected] += quanta[selected];
      return result;
    }
  }

  private static List<ModeSummary> summarize(
      List<Measurement> measurements, int repetitions, int[] batchRows) {
    ArrayList<ModeSummary> result = new ArrayList<>();
    int familyComparisons = Math.multiplyExact(batchRows.length, Mode.values().length);
    for (Mode mode : Mode.values()) {
      for (int requestedBatchRows : batchRows) {
        List<Double> samples =
            measurements.stream()
                .filter(value -> value.suite() == Suite.MIXED)
                .filter(value -> value.mode() == mode)
                .filter(value -> value.requestedBatchRows() == requestedBatchRows)
                .filter(value -> value.status() == MeasurementStatus.MEASURED)
                .map(Measurement::rowsPerSecond)
                .toList();
        if (samples.isEmpty()) {
          continue;
        }
        Confidence confidence = logConfidence(samples, familyComparisons);
        result.add(
            new ModeSummary(
                mode,
                requestedBatchRows,
                samples.size(),
                confidence.geometricMean(),
                confidence.lower95(),
                confidence.upper95(),
                repetitions == 1
                    ? ConfidenceKind.PRELIMINARY
                    : ConfidenceKind.BONFERRONI_GLOBAL_95));
      }
    }
    return List.copyOf(result);
  }

  static Confidence logConfidence(List<Double> samples, int familyComparisons) {
    if (samples.isEmpty() || familyComparisons < 1) {
      throw new IllegalArgumentException("samples and family comparisons must be positive");
    }
    double mean = 0.0;
    for (double sample : samples) {
      mean += Math.log(sample);
    }
    mean /= samples.size();
    if (samples.size() == 1) {
      double value = Math.exp(mean);
      return new Confidence(value, value, value);
    }
    double variance = 0.0;
    for (double sample : samples) {
      double delta = Math.log(sample) - mean;
      variance += delta * delta;
    }
    variance /= samples.size() - 1;
    if (samples.size() != 3) {
      throw new IllegalArgumentException("confirmed replay confidence requires exactly 3 samples");
    }
    double critical = studentTCriticalDf2(0.05 / familyComparisons);
    double halfWidth = critical * Math.sqrt(variance / samples.size());
    return new Confidence(Math.exp(mean), Math.exp(mean - halfWidth), Math.exp(mean + halfWidth));
  }

  private static double studentTCriticalDf2(double twoSidedAlpha) {
    double probability = 1.0 - twoSidedAlpha / 2.0;
    double centered = 2.0 * probability - 1.0;
    return Math.sqrt(2.0) * centered / Math.sqrt(1.0 - centered * centered);
  }

  static Verdict verdict(List<ModeSummary> summaries, int repetitions, double targetRowsPerSecond) {
    EnumMap<Mode, ModeSummary> best = new EnumMap<>(Mode.class);
    EnumMap<Mode, Double> maximumLower95 = new EnumMap<>(Mode.class);
    EnumMap<Mode, Double> maximumUpper95 = new EnumMap<>(Mode.class);
    for (ModeSummary summary : summaries) {
      ModeSummary current = best.get(summary.mode());
      if (current == null
          || summary.geometricMeanRowsPerSecond() > current.geometricMeanRowsPerSecond()) {
        best.put(summary.mode(), summary);
      }
      maximumLower95.merge(summary.mode(), summary.lower95RowsPerSecond(), Math::max);
      maximumUpper95.merge(summary.mode(), summary.upper95RowsPerSecond(), Math::max);
    }
    if (best.size() != Mode.values().length) {
      return new Verdict(
          Classification.INCOMPLETE,
          "missing mixed replay measurements",
          Map.copyOf(best),
          Map.copyOf(maximumLower95),
          Map.copyOf(maximumUpper95));
    }
    boolean preliminary = repetitions < 2;
    if (maximumUpper95.get(Mode.COMPUTE_ONLY) < targetRowsPerSecond) {
      return new Verdict(
          preliminary
              ? Classification.PRELIMINARY_MODEL_OR_KERNEL_LIMIT
              : Classification.MODEL_OR_KERNEL_LIMIT,
          "all compute-only upper confidence bounds are below target",
          Map.copyOf(best),
          Map.copyOf(maximumLower95),
          Map.copyOf(maximumUpper95));
    }
    if (maximumLower95.get(Mode.COMPUTE_ONLY) >= targetRowsPerSecond
        && maximumUpper95.get(Mode.GPU_PIPELINE) < targetRowsPerSecond) {
      return new Verdict(
          preliminary
              ? Classification.PRELIMINARY_TRANSFER_STREAM_OR_DJL_LIMIT
              : Classification.TRANSFER_STREAM_OR_DJL_LIMIT,
          "a compute-only lower bound clears target, but every GPU-pipeline upper bound misses it",
          Map.copyOf(best),
          Map.copyOf(maximumLower95),
          Map.copyOf(maximumUpper95));
    }
    if (maximumLower95.get(Mode.GPU_PIPELINE) >= targetRowsPerSecond
        && maximumUpper95.get(Mode.FULL_INFERENCE) < targetRowsPerSecond) {
      return new Verdict(
          preliminary
              ? Classification.PRELIMINARY_JAVA_COMPOSE_OR_SLOT_LIFETIME_LIMIT
              : Classification.JAVA_COMPOSE_OR_SLOT_LIFETIME_LIMIT,
          "a GPU-pipeline lower bound clears target, but every full-inference upper bound misses"
              + " it",
          Map.copyOf(best),
          Map.copyOf(maximumLower95),
          Map.copyOf(maximumUpper95));
    }
    if (maximumLower95.get(Mode.FULL_INFERENCE) >= targetRowsPerSecond) {
      return new Verdict(
          preliminary
              ? Classification.PRELIMINARY_INFERENCE_CAPACITY_SUFFICIENT
              : Classification.INFERENCE_CAPACITY_SUFFICIENT,
          "at least one full-inference lower confidence bound clears target",
          Map.copyOf(best),
          Map.copyOf(maximumLower95),
          Map.copyOf(maximumUpper95));
    }
    return new Verdict(
        preliminary ? Classification.PRELIMINARY_INCONCLUSIVE : Classification.INCONCLUSIVE,
        "confidence interval crosses target or multiple boundaries overlap",
        Map.copyOf(best),
        Map.copyOf(maximumLower95),
        Map.copyOf(maximumUpper95));
  }

  private static Map<Mode, String> modeDefinitions() {
    return Map.of(
        Mode.COMPUTE_ONLY,
        "fixed device input; production forward and output packing; no H2D, D2H, or host compose",
        Mode.GPU_PIPELINE,
        "heap rows to reusable pinned input, H2D, forward, output packing, and D2H; no pinned read"
            + " or host compose",
        Mode.FULL_INFERENCE,
        "production heap-to-pinned staging, H2D, forward, output packing, D2H, pinned read,"
            + " compose, and decode");
  }

  private static void writeAtomically(Path target, Report report) throws IOException {
    Path temporary = target.resolveSibling(target.getFileName() + ".tmp");
    Files.writeString(temporary, GSON.toJson(report) + System.lineSeparator());
    try {
      Files.move(
          temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } catch (AtomicMoveNotSupportedException ignored) {
      Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  record Variant(
      EpsilonDecisionInferenceReplayCorpus.Role role,
      DecisionBucket bucket,
      int rows,
      long weight,
      DecisionHostBatch batch) {

    private String key() {
      return role + "/" + bucketName(bucket);
    }
  }

  private record ReplayTarget(
      EpsilonDecisionInferenceServer server, DecisionDevicePipeline.HostReadyCell hostReadyCell) {}

  record RunResult(
      long rows,
      long elapsedNanos,
      Map<String, Long> batchHistogram,
      Map<EpsilonDecisionInferenceReplayCorpus.Role, Long> roleRows,
      Map<String, Long> bucketRows,
      Map<String, Long> variantRows,
      Map<String, Long> deviceRows) {}

  private record GcSnapshot(long collections, long millis) {
    private static GcSnapshot current() {
      long collections = 0L;
      long millis = 0L;
      for (GarbageCollectorMXBean collector : ManagementFactory.getGarbageCollectorMXBeans()) {
        collections += Math.max(0L, collector.getCollectionCount());
        millis += Math.max(0L, collector.getCollectionTime());
      }
      return new GcSnapshot(collections, millis);
    }
  }

  record Confidence(double geometricMean, double lower95, double upper95) {}

  /** 性能測定が分離する三つの推論境界。 */
  public enum Mode {
    COMPUTE_ONLY,
    GPU_PIPELINE,
    FULL_INFERENCE
  }

  /** 単一容量区分曲線と本番容量区分混合の区別。 */
  public enum Suite {
    PURE,
    MIXED
  }

  /** 測定点を実行したか、対象外として省略したか。 */
  public enum MeasurementStatus {
    MEASURED,
    SKIPPED
  }

  /** 反復数に応じた信頼区間の算出方法。 */
  public enum ConfidenceKind {
    PRELIMINARY,
    BONFERRONI_GLOBAL_95
  }

  /** 推論境界の容量と目標処理速度を比較した分類。 */
  public enum Classification {
    INCOMPLETE,
    MODEL_OR_KERNEL_LIMIT,
    PRELIMINARY_MODEL_OR_KERNEL_LIMIT,
    TRANSFER_STREAM_OR_DJL_LIMIT,
    PRELIMINARY_TRANSFER_STREAM_OR_DJL_LIMIT,
    JAVA_COMPOSE_OR_SLOT_LIFETIME_LIMIT,
    PRELIMINARY_JAVA_COMPOSE_OR_SLOT_LIFETIME_LIMIT,
    INFERENCE_CAPACITY_SUFFICIENT,
    PRELIMINARY_INFERENCE_CAPACITY_SUFFICIENT,
    INCONCLUSIVE,
    PRELIMINARY_INCONCLUSIVE
  }

  /** 一測定点の実経過時間結果。 */
  public record Measurement(
      int repetition,
      Suite suite,
      Mode mode,
      String bucket,
      int requestedBatchRows,
      Map<String, Long> physicalBatchHistogram,
      Map<EpsilonDecisionInferenceReplayCorpus.Role, Long> roleRows,
      Map<String, Long> bucketRows,
      Map<String, Long> variantRows,
      Map<String, Long> deviceRows,
      double maximumAbsoluteMixError,
      double mixErrorTolerance,
      int warmupBatches,
      int measuredBatches,
      long rows,
      long elapsedNanos,
      double rowsPerSecond,
      Map<String, AllocatorMemoryMeasurement> allocatorMemory,
      long gcCount,
      long gcMillis,
      MeasurementStatus status,
      String note) {

    private static Measurement skipped(
        int repetition,
        Suite suite,
        Mode mode,
        String bucket,
        int requestedBatchRows,
        String note) {
      return new Measurement(
          repetition,
          suite,
          mode,
          bucket,
          requestedBatchRows,
          Map.of(),
          Map.of(),
          Map.of(),
          Map.of(),
          Map.of(),
          0.0,
          0.0,
          0,
          0,
          0L,
          0L,
          0.0,
          Map.of(),
          0L,
          0L,
          MeasurementStatus.SKIPPED,
          note);
    }
  }

  /** 一つのGPUについて、ウォームアップ後の基準値と測定中peakを分離したPyTorch allocator統計。 */
  public record AllocatorMemoryMeasurement(
      long baselineAllocatedBytes,
      long finalAllocatedBytes,
      long peakAllocatedBytes,
      long baselineActiveBytes,
      long finalActiveBytes,
      long peakActiveBytes,
      long baselineReservedBytes,
      long finalReservedBytes,
      long peakReservedBytes) {}

  /** 実効推論設定。 */
  public record EffectiveSettings(
      DecisionComputePrecision computePrecision,
      int maximumBatchRows,
      int multiTransitionMaximumBatchRows,
      int slotsPerDevice,
      int readyBatchesPerDevice,
      DecisionInferenceFusionSettings fusion,
      String actorDevices,
      String opponentDevices) {}

  /** 性能測定が作成または読込した固定ホスト固定入力データファイルの再現条件。 */
  public record CorpusFileReport(
      CorpusMode mode,
      String file,
      int formatVersion,
      String inputSchemaFingerprint,
      long seedBase,
      int captureGames) {}

  /** 記録した一容量区分の観測量と保持量。 */
  public record CorpusBucketReport(long observedRows, int retainedRows, int targetRows) {}

  /** 一モデル役割から記録した固定ホスト入力固定入力データ。 */
  public record CorpusReport(
      EpsilonDecisionInferenceReplayCorpus.Role role,
      long observedRows,
      int retainedRows,
      String sampling,
      Map<String, CorpusBucketReport> buckets) {}

  /** 同一方式・混合バッチ sizeの反復要約。 */
  public record ModeSummary(
      Mode mode,
      int requestedBatchRows,
      int samples,
      double geometricMeanRowsPerSecond,
      double lower95RowsPerSecond,
      double upper95RowsPerSecond,
      ConfidenceKind confidenceKind) {}

  /** 181k 行/s境界に対する最終分類。 */
  public record Verdict(
      Classification classification,
      String reason,
      Map<Mode, ModeSummary> bestMixedByMode,
      Map<Mode, Double> maximumLower95RowsPerSecondByMode,
      Map<Mode, Double> maximumUpper95RowsPerSecondByMode) {}

  /** 再現条件、全測定点、判定をまとめたJSON 起点。 */
  public record Report(
      int schemaVersion,
      String checkpoint,
      long seedBase,
      int captureGames,
      int repetitions,
      double targetRowsPerSecond,
      List<Integer> batchRows,
      EffectiveSettings settings,
      CorpusFileReport corpusFile,
      CorpusReport actorCorpus,
      CorpusReport opponentCorpus,
      List<Measurement> measurements,
      List<ModeSummary> summaries,
      Map<Mode, String> modeDefinitions,
      Verdict verdict) {}

  private record PreparedCorpora(
      EpsilonDecisionInferenceReplayCorpusStore.Bundle bundle,
      String actorDevices,
      String opponentDevices,
      Set<Device> replayDevices) {}
}

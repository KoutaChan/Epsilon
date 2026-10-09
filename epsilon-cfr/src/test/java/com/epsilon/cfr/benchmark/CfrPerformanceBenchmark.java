package com.epsilon.cfr.benchmark;

import com.epsilon.cfr.algorithm.LcfrTargets;
import com.epsilon.cfr.collection.CfrCollector;
import com.epsilon.cfr.collection.CfrEpisodeSink;
import com.epsilon.cfr.collection.FrozenGrpBoundary;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.config.settings.EpsilonSettings;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrEpisodeFrame;
import com.epsilon.cfr.data.CfrEpisodeWriter;
import com.epsilon.cfr.data.CfrMatchFile;
import com.epsilon.cfr.data.CfrReplayStore;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.LogWeightedReplay;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.cfr.runtime.CfrPolicyFleet;
import com.epsilon.nano.ai.decision.data.DecisionInputRowCodec;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.network.NetworkDeviceResolver;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.random.RandomGenerator;
import java.util.stream.LongStream;

/**
 * 本番の収集・圧縮保存・重み付きリプレイ・学習処理を使い、明示的に起動して性能を測定する。
 *
 * <pre>
 * collect SETTINGS CHECKPOINT GRP_CHECKPOINT|- OUTPUT [WARMUP_GAMES=256] [PERSIST=true]
 * learn SETTINGS CHECKPOINT MATCHES_DIR OUTPUT [WARMUP_UPDATES=8] [MEASURED_UPDATES=32] [SAMPLING_MODE=weighted]
 * inspect MATCHES_DIR OUTPUT [MAX_MATCHES=32]
 * </pre>
 *
 * <p>{@code SETTINGS}には現行形式のTOMLを指定し、{@code -}なら同梱設定を使う。{@code OUTPUT}には
 * 未作成のディレクトリを指定する。収集件数・並列度・バッチサイズ・保存間隔は設定に従う。
 *
 * <p>{@code learn}は元モデルを変更せず、現行形式で保存された完了済み半荘の全経路から測定用のリプレイを作る。
 * 後悔学習と平均方策学習のどちらを行うかは、チェックポイントの役割（{@code REGRET}/{@code AVERAGE}）に従う。
 *
 * <p>{@code uniform-distinct}は重複の少ない負荷を測る診断専用の方式で、保持行を重みと無関係に非復元抽出する。
 * 教師データと重みは変更しないが、抽出分布が本番と異なるため、速度や損失を本番の学習結果として扱わない。
 *
 * <p>{@code inspect}は完成済み半荘を名前順に指定件数まで読み、保存された主経路の候補数とパディングを調べる。
 * モデルやGPUは使わず、一度に復号するのは一半荘だけとする。この部分集合を全対局の代表標本とはみなさない。
 */
public final class CfrPerformanceBenchmark {
  private static final String SOLVE = "performance-benchmark";

  private CfrPerformanceBenchmark() {}

  public static void main(String[] args) throws Exception {
    if (args.length == 0) throw new IllegalArgumentException("Expected collect, learn or inspect");
    switch (args[0]) {
      case "collect" -> collect(args);
      case "learn" -> learn(args);
      case "inspect" -> inspect(args);
      default ->
          throw new IllegalArgumentException("Expected collect, learn or inspect: " + args[0]);
    }
  }

  private static void inspect(String[] args) throws IOException {
    if (args.length < 3 || args.length > 4)
      throw new IllegalArgumentException("inspect MATCHES_DIR OUTPUT [MAX_MATCHES=32]");
    int limit = nonNegative(args, 3, 32);
    if (limit == 0) throw new IllegalArgumentException("Match limit must be positive");
    Path output = Path.of(args[2]).toAbsolutePath();
    Files.createDirectories(output.getParent());
    Files.createDirectory(output);
    write(output, inspectMatches(Path.of(args[1]), limit));
  }

  /** 保存元を変更せず、小さな部分集合から入力形状と保存領域の内訳を得る。 */
  static Map<String, Object> inspectMatches(Path directory, int limit) throws IOException {
    if (limit <= 0) throw new IllegalArgumentException("Match limit must be positive");
    long started = System.nanoTime();
    List<Path> files = matchFiles(directory);
    var totals = new ObservationStats();
    var buckets = new TreeMap<String, ObservationStats>();
    var matches = new ArrayList<Map<String, Object>>();
    for (Path file : files.subList(0, Math.min(limit, files.size()))) {
      List<CfrEpisode> trajectories = CfrMatchFile.read(file);
      long decisions = 0, branches = 0;
      for (CfrEpisode trajectory : trajectories) {
        branches += trajectory.branchOutcomes().size();
        for (CfrDecision decision : trajectory.decisions()) {
          ObservationShape shape = ObservationShape.read(decision);
          totals.add(shape);
          buckets
              .computeIfAbsent(shape.storedBucket(), ignored -> new ObservationStats())
              .add(shape);
          decisions++;
        }
      }
      matches.add(
          Map.of(
              "file", file.getFileName().toString(),
              "gameId", trajectories.getFirst().episodeId(),
              "trajectories", trajectories.size(),
              "decisions", decisions,
              "branchOutcomes", branches,
              "compressedFileBytes", Files.size(file)));
    }
    var byBucket = new TreeMap<String, Object>();
    buckets.forEach((bucket, stats) -> byBucket.put(bucket, stats.summary()));
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("benchmarkVersion", 2);
    report.put("operation", "inspect");
    report.put("matchesDirectory", directory.toAbsolutePath().normalize().toString());
    report.put("availableMatches", files.size());
    report.put("requestedMatches", limit);
    report.put("inspectedMatches", matches.size());
    report.put(
        "selection",
        "First completed .bin files in lexicographic filename order; not a random sample");
    report.put(
        "scope",
        "Saved main-path decisions from every trajectory and seat; branch outcome metadata is"
            + " counted but branch inference inputs are not retained");
    report.put(
        "byteEstimateScope",
        "Canonical uncompressed observation bytes including the CFR header; compact estimate"
            + " removes only transition padding and excludes any new metadata, object overhead or"
            + " compression effects; natural bucket sizes are hypothetical regrouping, not measured"
            + " GPU work");
    report.put(
        "zeroRunScope",
        "Contiguous structural padding within each transition tensor, merging each action suffix"
            + " with the following unused action slots; zero-valued real features are retained");
    report.put("bytesPerTransition", ObservationShape.TRANSITION_BYTES);
    report.put("transitionTensorCount", ObservationShape.TRANSITION_TENSORS);
    report.put("matches", matches);
    report.put("observations", totals.summary());
    report.put("byStoredBucket", byBucket);
    report.put("diagnosticSeconds", seconds(started));
    return report;
  }

  private record ObservationShape(
      int actions,
      int maximumTransitions,
      int transitions,
      int[] transitionsPerAction,
      String storedBucket,
      String naturalBucket,
      long actionSlots,
      long transitionSlots,
      long naturalActionSlots,
      long naturalTransitionSlots,
      long denseBytes,
      long naturalDenseBytes,
      long paddingBytes,
      long zeroRuns) {
    private static final int TRANSITION_BYTES = transitionBytes();
    private static final int TRANSITION_TENSORS =
        (int)
            Arrays.stream(DecisionInputLayout.Tensor.values())
                .filter(DecisionInputLayout.Tensor::isTransition)
                .count();

    static ObservationShape read(CfrDecision decision) throws IOException {
      var host = CfrObservationCodec.decode(decision.observation()).host();
      int actions = host.legalActionCount(0);
      if (actions != decision.actionCount())
        throw new IOException("Observation legal-action count differs from decision metadata");
      var bucket = host.bucket();
      int[] counts = new int[actions];
      int maximum = 0, total = 0, runs = 0;
      for (int action = 0; action < actions; action++) {
        int count = counts[action] = host.actionTransitionCount(0, action);
        maximum = Math.max(maximum, count);
        total += count;
        if (count < bucket.actionTransitionCapacity()) runs++;
      }
      if (actions < bucket.legalActionCapacity()
          && counts[actions - 1] == bucket.actionTransitionCapacity()) runs++;
      DecisionBucket natural = DecisionBucket.forRequiredCounts(actions, maximum);
      long slots = (long) bucket.legalActionCapacity() * bucket.actionTransitionCapacity();
      long denseBytes = Integer.BYTES + DecisionInputRowCodec.encodedSize(host.sliceRows(0, 1));
      return new ObservationShape(
          actions,
          maximum,
          total,
          counts,
          bucketName(bucket),
          bucketName(natural),
          bucket.legalActionCapacity(),
          slots,
          natural.legalActionCapacity(),
          (long) natural.legalActionCapacity() * natural.actionTransitionCapacity(),
          denseBytes,
          denseBytes - tensorBytes(bucket) + tensorBytes(natural),
          (slots - total) * TRANSITION_BYTES,
          (long) runs * TRANSITION_TENSORS);
    }

    private static String bucketName(DecisionBucket bucket) {
      return bucket.legalActionCapacity() + "x" + bucket.actionTransitionCapacity();
    }

    private static long tensorBytes(DecisionBucket bucket) {
      long bytes = 0;
      for (var tensor : DecisionInputLayout.Tensor.values())
        bytes +=
            (long) tensor.elementsPerRow(bucket)
                * (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL
                    ? Short.BYTES
                    : Float.BYTES);
      return bytes;
    }

    private static int transitionBytes() {
      int bytes = 0;
      for (var tensor : DecisionInputLayout.Tensor.values())
        if (tensor.isTransition())
          bytes +=
              tensor.elementsPerTransition()
                  * (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL
                      ? Short.BYTES
                      : Float.BYTES);
      return bytes;
    }
  }

  private static final class ObservationStats {
    private long rows, actions, transitions, actionSlots, transitionSlots;
    private long naturalActionSlots, naturalTransitionSlots, denseBytes, naturalDenseBytes;
    private long paddingBytes, zeroRuns, singleTransitionRows;
    private final TreeMap<Integer, Long> actionCounts = new TreeMap<>();
    private final TreeMap<Integer, Long> maximumTransitionCounts = new TreeMap<>();
    private final TreeMap<Integer, Long> rowTransitionCounts = new TreeMap<>();
    private final TreeMap<Integer, Long> actionTransitionCounts = new TreeMap<>();
    private final TreeMap<String, Long> naturalBuckets = new TreeMap<>();

    void add(ObservationShape shape) {
      rows++;
      actions += shape.actions();
      transitions += shape.transitions();
      actionSlots += shape.actionSlots();
      transitionSlots += shape.transitionSlots();
      naturalActionSlots += shape.naturalActionSlots();
      naturalTransitionSlots += shape.naturalTransitionSlots();
      denseBytes += shape.denseBytes();
      naturalDenseBytes += shape.naturalDenseBytes();
      paddingBytes += shape.paddingBytes();
      zeroRuns += shape.zeroRuns();
      if (shape.maximumTransitions() == 1) singleTransitionRows++;
      actionCounts.merge(shape.actions(), 1L, Long::sum);
      maximumTransitionCounts.merge(shape.maximumTransitions(), 1L, Long::sum);
      rowTransitionCounts.merge(shape.transitions(), 1L, Long::sum);
      for (int count : shape.transitionsPerAction())
        actionTransitionCounts.merge(count, 1L, Long::sum);
      naturalBuckets.merge(shape.naturalBucket(), 1L, Long::sum);
    }

    Map<String, Object> summary() {
      Map<String, Object> result = new LinkedHashMap<>();
      result.put("rows", rows);
      result.put("legalActions", actions);
      result.put("actualTransitions", transitions);
      result.put("oneTransitionPerActionRows", singleTransitionRows);
      result.put("meanLegalActions", ratio(actions, rows));
      result.put("meanActualTransitions", ratio(transitions, rows));
      result.put("legalActionCountRows", actionCounts);
      result.put("maximumTransitionsPerActionRows", maximumTransitionCounts);
      result.put("actualTransitionCountRows", rowTransitionCounts);
      result.put("transitionCountActions", actionTransitionCounts);
      result.put("naturalBucketRows", naturalBuckets);
      result.put("storedActionSlots", actionSlots);
      result.put("storedTransitionSlots", transitionSlots);
      result.put("naturalActionSlots", naturalActionSlots);
      result.put("naturalTransitionSlots", naturalTransitionSlots);
      result.put("storedActionPaddingFraction", ratio(actionSlots - actions, actionSlots));
      result.put(
          "storedTransitionPaddingFraction", ratio(transitionSlots - transitions, transitionSlots));
      result.put("densePayloadBytes", denseBytes);
      result.put("naturalBucketDensePayloadBytes", naturalDenseBytes);
      result.put("transitionPaddingBytes", paddingBytes);
      result.put("compactPayloadBytesExcludingMetadata", denseBytes - paddingBytes);
      result.put("transitionPaddingByteFraction", ratio(paddingBytes, denseBytes));
      result.put("structuralZeroRuns", zeroRuns);
      return result;
    }
  }

  private static void collect(String[] args) throws Exception {
    if (args.length < 5 || args.length > 7)
      throw new IllegalArgumentException(
          "collect SETTINGS CHECKPOINT GRP_CHECKPOINT|- OUTPUT"
              + " [WARMUP_GAMES=256] [PERSIST=true]");
    CfrConfig config = settings(args[1]);
    Path checkpoint = Path.of(args[2]).toAbsolutePath();
    Path output = output(args[4], config);
    int warmup = nonNegative(args, 5, 256);
    boolean persist = args.length < 7 || booleanValue(args[6]);
    boolean hand = config.collection().mode().equals("HAND_GRP");
    if (hand && args[3].equals("-"))
      throw new IllegalArgumentException("HAND_GRP requires a frozen GRP path");
    long started = System.nanoTime();
    var devices = NetworkDeviceResolver.inference(config.network().devices());
    Map<String, Object> report = report("collect", config, checkpoint);
    report.put("inferenceDevices", devices.toString());
    report.put("grpCheckpoint", args[3]);
    report.put("persistMatches", persist);
    try (CfrModel model = CfrModel.loadModel(checkpoint, devices.primary());
        CfrPolicyFleet fleet = new CfrPolicyFleet(model, devices, config.collection());
        EpsilonGrpTrainingSession grp =
            hand
                ? EpsilonGrpTrainingSession.openInferenceOnly(
                    Path.of(args[3]),
                    NetworkDeviceResolver.grp(config.network().devices()).primary())
                : null) {
      report.put("setupSeconds", seconds(started));
      report.put("modelConfig", model.config());
      FrozenGrpBoundary boundary =
          grp == null ? null : new FrozenGrpBoundary(grp.inference(), config.utilityProfile());
      if (warmup > 0)
        report.put(
            "warmup",
            collectionWindow(
                config,
                fleet,
                boundary,
                output.resolve("warmup-matches"),
                warmup,
                SeedMixer.derive(config.seed(), 0x5741524dL),
                persist));
      report.put(
          "measured",
          collectionWindow(
              config,
              fleet,
              boundary,
              output.resolve("matches"),
              config.collection().games(),
              config.seed(),
              persist));
    }
    if (persist) {
      long diagnostics = System.nanoTime();
      Shapes shapes = new Shapes();
      for (Path file : matchFiles(output.resolve("matches"))) {
        for (var trajectory : CfrMatchFile.read(file))
          for (var row : trajectory.decisions())
            shapes.add(shape(row.observation(), row.actionCount()));
      }
      report.put("completedObservationShapes", shapes.summary());
      report.put("postMeasurementDecodeDiagnosticSeconds", seconds(diagnostics));
    }
    report.put(
        "timingScope",
        "Measured collection includes feature encoding, all inference and engine work, GRP, branch"
            + " completion, and optional compressed match persistence including writer drain. Games"
            + " count actual hanchan completions; hands and LCFR trajectories are separate."
            + " Completed decisions count all retained main-game choices, including forced choices;"
            + " branch work is reported separately. Model load, warmup and post-run observation"
            + " shape decoding are excluded. The game budget uses the same hanchan unit as Nano,"
            + " but CFR retains all four seats instead of Nano's learner seat. CollectorStats"
            + " elapsedSeconds excludes the final writer drain; headline elapsedSeconds includes"
            + " it.");
    write(output, report);
  }

  private static Map<String, Object> collectionWindow(
      CfrConfig config,
      CfrPolicyFleet fleet,
      FrozenGrpBoundary boundary,
      Path matches,
      int count,
      long seed,
      boolean persist)
      throws IOException {
    if (persist) Files.createDirectory(matches);
    var settings = config.collection();
    var mode = CfrCollector.Mode.valueOf(settings.mode());
    var collector =
        new CfrCollector(
            new CfrCollector.Config(
                mode,
                settings.exploration(),
                settings.inferenceBatch(),
                settings.concurrentGames(),
                seed,
                settings.centerUtility(),
                true,
                settings.multiTransitionMaxBatch(),
                settings.arena(),
                settings.batching(),
                config.utilityProfile()),
            settings.branchComparison());
    long[] completed = new long[4];
    long started = System.nanoTime();
    CfrCollector.CollectionStats stats;
    try (var writer = persist ? new CfrEpisodeWriter(matches) : null) {
      stats =
          collector.collect(
              LongStream.range(0, count).boxed().toList(),
              fleet,
              boundary,
              new CfrEpisodeSink() {
                @Override
                public boolean tryAccept(CfrCollector.CollectedEpisode item) {
                  CfrEpisode episode = item.episode();
                  if (writer != null
                      && !writer.tryAccept(
                          new CfrEpisodeFrame(episode, item.trajectoryIndex(), item.gameEnded())))
                    return false;
                  if (item.gameEnded()) completed[0]++;
                  completed[1]++;
                  completed[2] += episode.decisions().size();
                  for (var decision : episode.decisions())
                    if (decision.actionCount() == 1) completed[3]++;
                  return true;
                }

                @Override
                public CompletionStage<Void> whenAvailable() {
                  return writer == null
                      ? CompletableFuture.completedFuture(null)
                      : writer.whenAvailable();
                }
              });
    } catch (UncheckedIOException error) {
      throw error.getCause();
    }
    double elapsed = seconds(started);
    Map<String, Object> result = new LinkedHashMap<>();
    result.put("elapsedSeconds", elapsed);
    result.put("completedGames", completed[0]);
    result.put("completedHands", stats.hands());
    result.put("completedTrajectories", completed[1]);
    result.put("completedDecisions", completed[2]);
    result.put("forcedDecisions", completed[3]);
    result.put("forcedDecisionFraction", ratio(completed[3], completed[2]));
    result.put("gamesPerSecond", completed[0] / elapsed);
    result.put("handsPerSecond", stats.hands() / elapsed);
    result.put("trajectoriesPerSecond", completed[1] / elapsed);
    result.put("completedDecisionsPerSecond", completed[2] / elapsed);
    result.put("nonForcedDecisionsPerSecond", (completed[2] - completed[3]) / elapsed);
    result.put("mainInferenceRowsPerSecond", stats.decisions() / elapsed);
    result.put("branchInferenceRows", stats.branchInferenceRows());
    result.put("branchInferenceRowsPerSecond", stats.branchInferenceRows() / elapsed);
    result.put(
        "averageInferenceBatchRows",
        ratio(stats.decisions() + stats.branchInferenceRows(), stats.inferenceBatches()));
    result.put("collectorStats", stats);
    return result;
  }

  private static void learn(String[] args) throws Exception {
    if (args.length < 5 || args.length > 8)
      throw new IllegalArgumentException(
          "learn SETTINGS CHECKPOINT MATCHES_DIR OUTPUT [WARMUP_UPDATES=8] [MEASURED_UPDATES=32]"
              + " [SAMPLING_MODE=weighted|uniform-distinct]");
    SamplingMode samplingMode = SamplingMode.parse(args.length > 7 ? args[7] : "weighted");
    CfrConfig config = settings(args[1]);
    Path checkpoint = Path.of(args[2]).toAbsolutePath();
    List<Path> matches = matchFiles(Path.of(args[3]));
    Path output = output(args[4], config);
    int warmup = nonNegative(args, 5, 8);
    int measured = nonNegative(args, 6, 32);
    if (measured == 0) throw new IllegalArgumentException("Measured updates must be positive");
    Map<String, Object> report = report("learn", config, checkpoint);
    report.put("samplingMode", samplingMode.label);
    report.put("diagnosticOnly", samplingMode == SamplingMode.UNIFORM_DISTINCT);
    report.put("samplingScope", samplingMode.description);
    Map<CfrSampleId, Shape> shapesById = new HashMap<>();
    Shapes corpusShapes = new Shapes();
    Path replayPath = output.resolve("replay");
    long started = System.nanoTime();
    var summaries = new ArrayList<CfrMatchFile.Summary>(matches.size());
    var gameIds = new HashSet<Long>();
    long sourceTrajectories = 0, sourceDecisions = 0;
    for (Path file : matches) {
      var summary = CfrMatchFile.summary(file);
      if (!gameIds.add(summary.gameId()))
        throw new IOException("Duplicate benchmark game identity");
      summaries.add(summary);
      sourceTrajectories = Math.addExact(sourceTrajectories, summary.trajectories());
      sourceDecisions = Math.addExact(sourceDecisions, summary.decisions());
    }
    double decodeSeconds = 0;
    try (CfrReplayStore replay =
        new CfrReplayStore(
            replayPath,
            SOLVE,
            config.replay().regretCapacity(),
            config.replay().averageCapacity(),
            SeedMixer.derive(config.seed(), 0x52524553L),
            SeedMixer.derive(config.seed(), 0x41524553L),
            config.replay().maxRowBytes(),
            Math.min(256L << 20, config.replay().cacheRows() * config.replay().maxRowBytes()),
            config.replay().maxStoredBytes())) {
      for (int game = 0; game < matches.size(); game++) {
        var rows = new ArrayList<CfrTrainingRow>();
        for (CfrEpisode trajectory : CfrMatchFile.read(matches.get(game)))
          rows.addAll(LcfrTargets.generate(trajectory, 1, sourceTrajectories));
        replay.appendEpisode(1, summaries.get(game).gameId(), rows);
        long decodeStart = System.nanoTime();
        for (CfrTrainingRow row : rows) {
          Shape shape = shape(row.observation(), row.actionCount());
          shapesById.put(row.id(), shape);
          corpusShapes.add(shape);
        }
        decodeSeconds += seconds(decodeStart);
      }
      replay.save(1);
      report.put("replayStorage", replay.stats());
    }
    report.put("corpusPreparationSeconds", seconds(started));
    report.put("corpusDecodeDiagnosticSeconds", decodeSeconds);
    report.put("corpusShapes", corpusShapes.summary());
    report.put("sourceGames", matches.size());
    report.put("sourceTrajectories", sourceTrajectories);
    report.put("sourceDecisions", sourceDecisions);
    report.put(
        "trajectoryNormalization",
        "Each hand or match target uses the total trajectory count in the source corpus.");
    var devices = NetworkDeviceResolver.learner(config.network().devices());
    report.put("learnerDevices", devices.toString());
    started = System.nanoTime();
    try (CfrReplayStore replay = CfrReplayStore.loadGeneration(replayPath, SOLVE, 1);
        CfrModel model = CfrModel.loadTraining(checkpoint, devices)) {
      report.put("loadSeconds", seconds(started));
      report.put("modelConfig", model.config());
      report.put("role", model.role());
      LogWeightedReplay sampler =
          model.role() == CfrModel.Role.REGRET ? replay.regretSampler() : replay.averageSampler();
      report.put("replayWeights", sampler.metrics());
      samplingMode.validate(sampler, config.learning().batchSize());
      if (warmup > 0)
        report.put(
            "warmup",
            learningWindow(
                config,
                model,
                sampler,
                samplingMode,
                shapesById,
                output.resolve("warmup-checkpoints"),
                warmup,
                0));
      report.put(
          "measured",
          learningWindow(
              config,
              model,
              sampler,
              samplingMode,
              shapesById,
              output.resolve("measured-checkpoints"),
              measured,
              warmup));
    }
    report.put(
        "timingScope",
        "Headline samples/s includes replay sampling and payload decode, CfrModel.train"
            + " including observation decode, bucket assignment, CPU preparation, transfer,"
            + " forward/backward/optimizer, and periodic saveTraining including deferred GPU"
            + " synchronization. Checkpoints are saved every configured interval and at window"
            + " end. Internal decode cannot be separated through the public model API; the"
            + " reported corpusDecodeDiagnosticSeconds is separate setup work, not a subtraction"
            + " from learner time. Unique processed rows sum within-update distinct identities."
            + " Sampling mode is reported separately; uniform-distinct is a compute diagnostic,"
            + " not production weighted training. Stored teachers and weights are unchanged."
            + " Warmup, model/replay load, and corpus preparation are reported separately.");
    write(output, report);
  }

  private static Map<String, Object> learningWindow(
      CfrConfig config,
      CfrModel model,
      LogWeightedReplay sampler,
      SamplingMode samplingMode,
      Map<CfrSampleId, Shape> shapesById,
      Path checkpoints,
      int updates,
      int offset)
      throws IOException {
    Files.createDirectory(checkpoints);
    Shapes logicalShapes = new Shapes(), processedShapes = new Shapes();
    HashSet<CfrSampleId> windowUnique = new HashSet<>();
    long logical = 0, processed = 0;
    double sampling = 0, training = 0, saving = 0, totalLoss = 0;
    int saves = 0;
    long started = System.nanoTime();
    for (int update = 0; update < updates; update++) {
      long part = System.nanoTime();
      List<CfrTrainingRow> rows =
          samplingMode.sample(
              sampler,
              config.learning().batchSize(),
              new SplittableRandom(SeedMixer.derive(config.seed(), 0x44524157L, offset + update)));
      sampling += seconds(part);
      part = System.nanoTime();
      CfrModel.TrainingMetrics metrics = model.train(rows);
      training += seconds(part);
      totalLoss += metrics.loss();
      logical += metrics.samples();
      HashSet<CfrSampleId> batchUnique = new HashSet<>();
      for (CfrTrainingRow row : rows) {
        Shape shape = shapesById.get(row.id());
        logicalShapes.add(shape);
        if (batchUnique.add(row.id())) processedShapes.add(shape);
      }
      processed += batchUnique.size();
      windowUnique.addAll(batchUnique);
      if ((update + 1) % config.learning().checkpointEveryUpdates() == 0 || update + 1 == updates) {
        part = System.nanoTime();
        model.saveTraining(checkpoints.resolve(String.format(Locale.ROOT, "%06d", update + 1)));
        saving += seconds(part);
        saves++;
      }
    }
    double elapsed = seconds(started);
    Map<String, Object> result = new LinkedHashMap<>();
    result.put("samplingMode", samplingMode.label);
    result.put("diagnosticOnly", samplingMode == SamplingMode.UNIFORM_DISTINCT);
    result.put("elapsedSeconds", elapsed);
    result.put("samplingAndPayloadDecodeSeconds", sampling);
    result.put("trainCallIncludingObservationDecodeSeconds", training);
    result.put("checkpointSaveAndSynchronizationSeconds", saving);
    result.put(
        "instrumentationAndLoopSeconds", Math.max(0, elapsed - sampling - training - saving));
    result.put("updates", updates);
    result.put("checkpointSaves", saves);
    result.put("logicalSamples", logical);
    result.put("uniqueProcessedRows", processed);
    result.put("distinctRowsAcrossWindow", windowUnique.size());
    result.put("logicalSamplesPerSecond", logical / elapsed);
    result.put("uniqueProcessedRowsPerSecond", processed / elapsed);
    result.put("updatesPerSecond", updates / elapsed);
    result.put("meanLoss", totalLoss / updates);
    result.put("logicalSampleShapes", logicalShapes.summary());
    result.put("uniqueProcessedShapes", processedShapes.summary());
    return result;
  }

  enum SamplingMode {
    WEIGHTED("weighted", "Production log-weight sampling with replacement."),
    UNIFORM_DISTINCT(
        "uniform-distinct",
        "Diagnostic only: uniform sampling without replacement within each update over all"
            + " retained rows in the role reservoir, including zero-weight rows. The same row"
            + " may recur across updates. Stored targets and weights are unchanged; this"
            + " changes the sampling distribution and is not a training-algorithm setting.");

    final String label;
    final String description;

    SamplingMode(String label, String description) {
      this.label = label;
      this.description = description;
    }

    static SamplingMode parse(String value) {
      for (SamplingMode mode : values()) if (mode.label.equals(value)) return mode;
      throw new IllegalArgumentException("Expected weighted or uniform-distinct: " + value);
    }

    void validate(LogWeightedReplay sampler, int count) {
      if (count < 0) throw new IllegalArgumentException("Negative benchmark batch size");
      if (this == WEIGHTED && sampler.metrics().positiveWeightRows() == 0)
        throw new IllegalArgumentException("Replay has no positive-weight rows");
      if (this == UNIFORM_DISTINCT && count > sampler.size())
        throw new IllegalArgumentException(
            "uniform-distinct needs at least " + count + " retained rows, found " + sampler.size());
    }

    List<CfrTrainingRow> sample(LogWeightedReplay sampler, int count, RandomGenerator random) {
      validate(sampler, count);
      if (this == WEIGHTED) return sampler.sample(count, random);
      // Fisher-Yates法の必要な部分だけを実行する。変更位置だけを保持し、抽出した行だけを既存リプレイから復号する。
      Map<Integer, Integer> positions = new HashMap<>();
      List<CfrTrainingRow> rows = new ArrayList<>(count);
      for (int picked = 0; picked < count; picked++) {
        int last = sampler.size() - picked - 1;
        int selected = random.nextInt(last + 1);
        int index = positions.getOrDefault(selected, selected);
        if (selected != last) positions.put(selected, positions.getOrDefault(last, last));
        positions.remove(last);
        rows.add(sampler.rowAt(index));
      }
      return List.copyOf(rows);
    }
  }

  private static CfrConfig settings(String path) throws IOException {
    return CfrConfig.load(
        path.equals("-")
            ? EpsilonSettings.defaults()
            : EpsilonSettings.load(Path.of(path), Map.of()));
  }

  private static Path output(String value, CfrConfig config) throws IOException {
    Path path = Path.of(value).toAbsolutePath();
    Files.createDirectories(path.getParent());
    Files.createDirectory(path);
    config.write(path.resolve("effective-config.json"));
    return path;
  }

  private static List<Path> matchFiles(Path directory) throws IOException {
    try (var files = Files.list(directory)) {
      List<Path> matches =
          files
              .filter(Files::isRegularFile)
              .filter(file -> file.getFileName().toString().endsWith(".bin"))
              .sorted()
              .toList();
      if (matches.isEmpty()) throw new IOException("No completed match files in " + directory);
      return matches;
    }
  }

  private static Map<String, Object> report(String operation, CfrConfig config, Path checkpoint) {
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("benchmarkVersion", 2);
    report.put("operation", operation);
    report.put("checkpoint", checkpoint.toString());
    report.put("effectiveConfig", config);
    report.put("javaVersion", System.getProperty("java.version"));
    report.put("availableProcessors", Runtime.getRuntime().availableProcessors());
    return report;
  }

  private static void write(Path output, Map<String, Object> report) throws IOException {
    String json = CfrConfig.JSON.toJson(report);
    Files.writeString(output.resolve("report.json"), json + System.lineSeparator());
    System.out.println(json);
  }

  private static int nonNegative(String[] args, int index, int fallback) {
    int value = args.length > index ? Integer.parseInt(args[index]) : fallback;
    if (value < 0) throw new IllegalArgumentException("Negative benchmark count");
    return value;
  }

  private static boolean booleanValue(String value) {
    if (!value.equals("true") && !value.equals("false"))
      throw new IllegalArgumentException("Expected true or false: " + value);
    return Boolean.parseBoolean(value);
  }

  private static double seconds(long start) {
    return (System.nanoTime() - start) / 1_000_000_000.0;
  }

  private static double ratio(long numerator, long denominator) {
    return denominator == 0 ? 0 : (double) numerator / denominator;
  }

  private static Shape shape(ObservationPayload observation, int actions) {
    var decoded = CfrObservationCodec.decode(observation);
    var bucket = decoded.host().bucket();
    return new Shape(
        actions, bucket.legalActionCapacity() + "x" + bucket.actionTransitionCapacity());
  }

  private record Shape(int actions, String bucket) {}

  private static final class Shapes {
    private long rows, forced;
    private final TreeMap<String, Long> buckets = new TreeMap<>();

    void add(Shape shape) {
      rows++;
      if (shape.actions() == 1) forced++;
      buckets.merge(shape.bucket(), 1L, Long::sum);
    }

    Map<String, Object> summary() {
      return Map.of(
          "rows", rows, "forcedDecisionFraction", ratio(forced, rows), "bucketRows", buckets);
    }
  }
}

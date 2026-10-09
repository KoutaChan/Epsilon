package com.epsilon.cfr.config;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.config.settings.EpsilonSettings;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.Default;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.config.settings.MultipleOf;
import com.epsilon.config.settings.NonNegative;
import com.epsilon.config.settings.Positive;
import com.epsilon.config.settings.Range;
import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.config.settings.SettingsPrefix;
import com.epsilon.io.AtomicFiles;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.lang.reflect.RecordComponent;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

/** 共通ローダーから直接生成し、実行ディレクトリへ保存する確定済みの学習設定。 */
@SettingsPrefix("epsilon.cfr")
public record CfrConfig(
    @Setting("seed") @Default("20261007") long seed,
    @Setting("solves") @Default("1") @Positive int solves,
    @Setting("iterations") @Default("128") @Positive int iterations,
    @Setting("utilityProfile") @Default("TENHOU") EpsilonUtilityProfile utilityProfile,
    @Setting("collection") Collection collection,
    @Setting("learning") Learning learning,
    @Setting("replay") Replay replay,
    @Setting("network") Network network,
    @Setting("boundary") Boundary boundary,
    @Setting("evaluation") Evaluation evaluation,
    @Setting("initialization") Initialization initialization) {
  public static final Gson JSON =
      new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();

  /** Nanoからの初期化。空のcheckpointは初期化なしを表し、保存後はrun内の相対パスとなる。 */
  @SettingsPrefix("epsilon.cfr.initialization")
  public record Initialization(
      @Setting("nanoCheckpoint") @Default("") String nanoCheckpoint,
      @Setting("games") @Default("256") @Range(min = 2) int games,
      @Setting("updates") @Default("2048") @Positive int updates,
      @Setting("capacity") @Default("65536") @Positive int capacity) {
    public Initialization(String nanoCheckpoint, int games, int updates, int capacity) {
      if (nanoCheckpoint == null)
        throw new IllegalArgumentException("Nano checkpoint setting is required");
      this.nanoCheckpoint = nanoCheckpoint.isBlank() ? "" : nanoCheckpoint;
      this.games = games;
      this.updates = updates;
      this.capacity = capacity;
      SettingsLoader.validate(this);
    }

    public boolean enabled() {
      return !nanoCheckpoint.isEmpty();
    }

    public Initialization withCheckpoint(String checkpoint) {
      return new Initialization(checkpoint, games, updates, capacity);
    }
  }

  public CfrConfig withInitialization(Initialization value) {
    return new CfrConfig(
        seed,
        solves,
        iterations,
        utilityProfile,
        collection,
        learning,
        replay,
        network,
        boundary,
        evaluation,
        value);
  }

  @SettingsPrefix("epsilon.cfr.collection")
  public record Collection(
      @Setting("mode") @Default("HAND_GRP") String mode,
      @Setting("games") @Default("32768") @Positive int games,
      @Setting("concurrentGames") @Default("4096") @Positive int concurrentGames,
      @Setting("inferenceBatch") @Default("4096") @Positive int inferenceBatch,
      @Setting("exploration") @Default("0.10") @Range(min = 0, minInclusive = false, max = 1)
          double exploration,
      @Setting("centerUtility") @Default("true") boolean centerUtility,
      @Setting("multiTransitionMaxBatch") @Default("2048") @Positive int multiTransitionMaxBatch,
      @Setting("slotsPerDevice") @Default("2") @Range(min = 1, max = 4) int slotsPerDevice,
      @Setting("readyBatchesPerDevice") @Default("4") @Positive int readyBatchesPerDevice,
      @Setting("arena") DecisionDuelArenaSettings arena,
      @Setting("batching") InferenceBatchingSettings batching,
      @Setting("branchComparison") DecisionBranchComparisonSettings branchComparison) {
    public Collection {
      if (!Set.of("HAND_GRP", "MATCH_MC").contains(mode))
        throw new IllegalArgumentException("Unknown mode: " + mode);
      if (multiTransitionMaxBatch > inferenceBatch
          || readyBatchesPerDevice < slotsPerDevice
          || arena == null
          || batching == null
          || branchComparison == null)
        throw new IllegalArgumentException("Invalid collection execution settings");
    }
  }

  @SettingsPrefix("epsilon.cfr.learning")
  public record Learning(
      @Setting("batchSize") @Default("1024") @Positive int batchSize,
      @Setting("microBatchSize") @Default("1024") @Positive int microBatchSize,
      @Setting("regretUpdates") @Default("256") @Positive int regretUpdates,
      @Setting("averageUpdates") @Default("256") @Positive int averageUpdates,
      @Setting("checkpointEveryUpdates") @Default("32") @Positive int checkpointEveryUpdates,
      @Setting("learningRate") @Default("2e-4") @Positive float learningRate,
      @Setting("weightDecay") @Default("1e-4") @NonNegative float weightDecay,
      @Setting("gradientClip") @Default("0.5") @NonNegative float gradientClip,
      @Setting("computePrecision") @Default("BFLOAT16") DecisionComputePrecision computePrecision,
      @Setting("tensorTransfer") @Default("PINNED_BUFFER") DecisionTensorTransfer tensorTransfer) {
    public Learning {
      if (microBatchSize > batchSize)
        throw new IllegalArgumentException("microBatchSize exceeds batchSize");
      if (computePrecision == null || tensorTransfer == null)
        throw new IllegalArgumentException("Learning execution settings are required");
    }
  }

  @SettingsPrefix("epsilon.cfr.replay")
  public record Replay(
      @Setting("regretCapacity") @Default("1048576") @Positive int regretCapacity,
      @Setting("averageCapacity") @Default("1048576") @Positive int averageCapacity,
      @Setting("maxStoredBytes") @Default("137438953472") @Positive long maxStoredBytes,
      @Setting("maxRowBytes") @Default("1048576") @Range(min = 1024) long maxRowBytes,
      @Setting("cacheRows") @Default("1024") @Positive int cacheRows) {
    public Replay {
      if (maxStoredBytes < maxRowBytes)
        throw new IllegalArgumentException("Invalid replay byte limits");
    }
  }

  @SettingsPrefix("epsilon.cfr.network")
  public record Network(
      @Setting("hidden") @Default("256") @Range(min = 16) @MultipleOf(4) int hidden,
      @Setting("devices") DeviceSettings devices) {
    public Network {
      if (devices == null) throw new IllegalArgumentException("devices are required");
    }
  }

  @SettingsPrefix("epsilon.cfr.boundary")
  public record Boundary(
      @Setting("checkpoint") @Default("") String checkpoint,
      @Setting("bootstrapGames") @Default("4096") @Range(min = 2) int bootstrapGames,
      @Setting("calibrationGames") @Default("4096") @Range(min = 2) int calibrationGames) {
    public Boundary {
      if (checkpoint == null)
        throw new IllegalArgumentException("Boundary checkpoint setting is required");
      if (checkpoint.isBlank()) checkpoint = "";
    }
  }

  @SettingsPrefix("epsilon.cfr.evaluation")
  public record Evaluation(
      @Setting("every") @Default("10") @Positive int every,
      @Setting("championDuel") DecisionChampionDuelSettings championDuel,
      @Setting("evalVs") DecisionEvalVsSettings evalVs,
      @Setting("arena") DecisionDuelArenaSettings arena,
      @Setting("batching") InferenceBatchingSettings batching) {}

  /** TOML・保存JSON・手動構築のすべてで、同じ共通設定検証を通す。 */
  public CfrConfig(
      long seed,
      int solves,
      int iterations,
      EpsilonUtilityProfile utilityProfile,
      Collection collection,
      Learning learning,
      Replay replay,
      Network network,
      Boundary boundary,
      Evaluation evaluation,
      Initialization initialization) {
    this.seed = seed;
    this.solves = solves;
    this.iterations = iterations;
    this.utilityProfile = utilityProfile;
    this.collection = collection;
    this.learning = learning;
    this.replay = replay;
    this.network = network;
    this.boundary = boundary;
    this.evaluation = evaluation;
    this.initialization = initialization;
    if (utilityProfile == null || !utilityProfile.rankBased())
      throw new IllegalArgumentException(
          "CFR requires a rank-based utility profile: " + utilityProfile);
    SettingsLoader.validate(this);
  }

  public static CfrConfig defaults() {
    return load(EpsilonSettings.defaults());
  }

  /** 平均方策の学習と評価を行う反復かどうかを判定する。各学習区間の最終反復も対象に含める。 */
  public boolean evaluatesAt(long iteration) {
    return iteration % evaluation.every() == 0 || iteration == iterations;
  }

  public static CfrConfig load(SettingsLoader settings) {
    return settings.bind(CfrConfig.class);
  }

  /** 実行ディレクトリ内に固定した設定JSONを読み込む。ユーザー設定ファイルの読み込みには{@code load}を使う。 */
  public static CfrConfig read(Path file) throws IOException {
    try {
      JsonObject json = JsonParser.parseString(Files.readString(file)).getAsJsonObject();
      requireFields(json, CfrConfig.class);
      return JSON.fromJson(json, CfrConfig.class);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid LCFR configuration " + file, invalid);
    }
  }

  public void write(Path file) throws IOException {
    Path parent = file.toAbsolutePath().getParent();
    if (parent != null) Files.createDirectories(parent);
    AtomicFiles.write(
        file, (JSON.toJson(this) + System.lineSeparator()).getBytes(StandardCharsets.UTF_8));
  }

  private static void requireFields(JsonObject json, Class<?> type) {
    if (json == null) throw new IllegalArgumentException("Missing section " + type.getSimpleName());
    Set<String> names =
        Arrays.stream(type.getRecordComponents())
            .map(RecordComponent::getName)
            .collect(Collectors.toSet());
    for (String name : json.keySet())
      if (!names.contains(name))
        throw new IllegalArgumentException("Unknown " + type.getSimpleName() + " setting: " + name);
    for (String name : names)
      if (!json.has(name) || json.get(name).isJsonNull())
        throw new IllegalArgumentException("Missing " + type.getSimpleName() + " setting: " + name);
    for (RecordComponent component : type.getRecordComponents()) {
      if (component.getType().isRecord()
          && json.has(component.getName())
          && !json.get(component.getName()).isJsonNull()) {
        requireFields(json.getAsJsonObject(component.getName()), component.getType());
      }
    }
  }
}

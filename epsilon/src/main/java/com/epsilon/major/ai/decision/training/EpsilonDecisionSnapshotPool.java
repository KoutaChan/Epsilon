package com.epsilon.major.ai.decision.training;

import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.config.settings.DecisionSnapshotPoolSettings;
import com.epsilon.core.GameState;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/** 自己対局の対戦相手として使う、過去の保存モデルのスナップショットを管理する。 */
public final class EpsilonDecisionSnapshotPool {

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final String REGISTRY_FILE = "decision-snapshot-pool.json";
  static final String CURRENT_REGISTRY_FORMAT = "epsilon-decision-arena-snapshot-pool";
  static final int CURRENT_REGISTRY_VERSION = 3;
  private static final Set<String> REGISTRY_FIELDS = Set.of("format", "version", "snapshots");
  private static final Set<String> SNAPSHOT_FIELDS =
      Set.of("path", "id", "iteration", "createdStep", "currentArenaChampion", "evalStats");
  private static final Set<String> EVAL_FIELDS = Set.of("averageRank", "topRate", "lastRate");

  private final int maxSnapshots;
  private final ArrayList<SnapshotEntry> snapshots = new ArrayList<>();

  /** 設定の保持上限を使って空の対局実行処理スナップショット集約を作る。 */
  public EpsilonDecisionSnapshotPool() {
    this(EpsilonSettings.defaults().bind(DecisionSnapshotPoolSettings.class).max());
  }

  /**
   * 指定した保持上限で空の対局実行処理スナップショット集約を作る。
   *
   * @param maxSnapshots 登録先に保持する最大チェックポイント数
   */
  public EpsilonDecisionSnapshotPool(int maxSnapshots) {
    if (maxSnapshots <= 0) {
      throw new IllegalArgumentException("maxSnapshots must be positive");
    }
    this.maxSnapshots = maxSnapshots;
  }

  /**
   * 対戦相手のチェックポイント項目をIDで置換または追加する。
   *
   * <p>上限を超えた場合は現在の対局収集用の採用モデルを残し、最古の非現在の項目を削除する。
   *
   * @param snapshot 検証済み変更不可チェックポイント項目
   */
  private synchronized void addSnapshot(SnapshotEntry snapshot) {
    if (snapshot.currentArenaChampion()) {
      for (int i = 0; i < snapshots.size(); i++) {
        SnapshotEntry existing = snapshots.get(i);
        if (existing.id() != snapshot.id() && existing.currentArenaChampion()) {
          snapshots.set(i, withCurrentArenaChampion(existing, false));
        }
      }
    }
    long snapshotId = snapshot.id();
    snapshots.removeIf(existing -> existing.id() == snapshotId);
    snapshots.add(snapshot);
    snapshots.sort(Comparator.comparingLong(SnapshotEntry::id));
    while (snapshots.size() > maxSnapshots) {
      removeOldestEvictableSnapshot();
    }
  }

  /** 採用モデルを登録する。反復回数とは独立したIDを割り当てる。 */
  public synchronized void registerArenaChampion(Path checkpoint) throws IOException {
    Path normalized = checkpoint.toAbsolutePath().normalize();
    for (SnapshotEntry entry : snapshots) {
      if (entry.currentArenaChampion() && Path.of(entry.path()).equals(normalized)) return;
    }
    registerArenaChampion(checkpoint, EvalStats.empty());
  }

  void registerArenaChampion(Path checkpoint, EvalStats stats) throws IOException {
    registerCheckpoint(checkpoint, true, stats);
  }

  /** 保存済みmacroをコピーせず、対戦相手の履歴へ登録する。 */
  void registerMacroCheckpoint(Path checkpoint) throws IOException {
    registerCheckpoint(checkpoint, false, EvalStats.empty());
  }

  private synchronized void registerCheckpoint(Path checkpoint, boolean champion, EvalStats stats)
      throws IOException {
    Path resolved = checkpoint.toAbsolutePath().normalize();
    EpsilonDecisionCheckpointBundle manifest =
        EpsilonDecisionCheckpointManager.loadManifest(resolved);
    long id = snapshots.isEmpty() ? 1L : Math.addExact(snapshots.getLast().id(), 1L);
    for (SnapshotEntry entry : snapshots) {
      if (Path.of(entry.path()).equals(resolved)
          || (champion
              && entry.iteration() == manifest.iteration
              && entry.createdStep() == manifest.globalStep)) {
        id = entry.id();
        break;
      }
    }
    SnapshotEntry snapshot =
        new SnapshotEntry(
            resolved.toString(), id, manifest.iteration, manifest.globalStep, champion, stats);
    requireSnapshotCheckpoint(snapshot);
    addSnapshot(snapshot);
  }

  /** 現在の対局収集用の採用モデルだけを固定し、それ以外の対局実行処理スナップショットを古い順に削除する。 */
  private void removeOldestEvictableSnapshot() {
    for (int i = 0; i < snapshots.size(); i++) {
      if (!snapshots.get(i).currentArenaChampion()) {
        snapshots.remove(i);
        return;
      }
    }
  }

  private static SnapshotEntry withCurrentArenaChampion(
      SnapshotEntry snapshot, boolean currentArenaChampion) {
    return new SnapshotEntry(
        snapshot.path(),
        snapshot.id(),
        snapshot.iteration(),
        snapshot.createdStep(),
        currentArenaChampion,
        snapshot.evalStats());
  }

  /** macro全体で一つの相手を使う。履歴がなければ必ず現在の採用モデルを使う。 */
  public synchronized long[][] sampleOpponentIdsForMacro(long seed, double championProbability) {
    SnapshotEntry champion = null;
    ArrayList<SnapshotEntry> history = new ArrayList<>();
    for (SnapshotEntry snapshot : snapshots) {
      if (snapshot.currentArenaChampion()) champion = snapshot;
      else history.add(snapshot);
    }
    if (champion == null) throw new IllegalStateException("Decision snapshot pool has no champion");
    SplittableRandom random = new SplittableRandom(seed);
    long id =
        history.isEmpty() || random.nextDouble() < championProbability
            ? champion.id()
            : history.get(random.nextInt(history.size())).id();
    long[][] ids = new long[GameState.NUM_PLAYERS][GameState.NUM_PLAYERS - 1];
    for (long[] seats : ids) Arrays.fill(seats, id);
    return ids;
  }

  /**
   * 登録先内の対局実行処理スナップショット数を返す。
   *
   * @return 保持中項目数
   */
  public synchronized int size() {
    return snapshots.size();
  }

  /**
   * 指定IDのスナップショット項目を検索する。
   *
   * @param id 検索するスナップショットID
   * @return 対応項目。存在しない場合は {@code null}
   */
  public synchronized SnapshotEntry snapshot(long id) {
    for (SnapshotEntry snapshot : snapshots) {
      if (snapshot.id() == id) {
        return snapshot;
      }
    }
    return null;
  }

  /**
   * 登録先をチェックポイントの親ディレクトリへ原子的に保存する。
   *
   * @param checkpointDir 登録先ファイルを置くチェックポイントの親ディレクトリ
   * @throws IOException 現在の対局収集用の採用モデル、チェックポイント識別情報、または書込の検証に失敗した場合
   */
  public synchronized void save(Path checkpointDir) throws IOException {
    requireSingleCurrentArenaChampion(snapshots);
    for (SnapshotEntry snapshot : snapshots) {
      requireSnapshotCheckpoint(snapshot);
    }
    Files.createDirectories(checkpointDir);
    Path target = checkpointDir.resolve(REGISTRY_FILE);
    Path tmp = Files.createTempFile(checkpointDir, "." + REGISTRY_FILE + ".tmp-", ".json");
    boolean installed = false;
    try {
      Files.writeString(
          tmp,
          GSON.toJson(new Registry(CURRENT_REGISTRY_FORMAT, CURRENT_REGISTRY_VERSION, snapshots)));
      try {
        Files.move(
            tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
      } catch (AtomicMoveNotSupportedException e) {
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
      }
      installed = true;
    } finally {
      if (!installed) {
        Files.deleteIfExists(tmp);
      }
    }
  }

  /**
   * チェックポイントの親ディレクトリの現行形式登録先を厳密に読み込む。
   *
   * @param checkpointDir 登録先ファイルを持つチェックポイントの親ディレクトリ
   * @return 検証済み集約。ファイルが無い場合は空の集約
   * @throws IOException 形式、フィールド、チェックポイント識別情報のいずれかが不正な場合
   */
  public static EpsilonDecisionSnapshotPool load(Path checkpointDir) throws IOException {
    return load(
        checkpointDir, EpsilonSettings.defaults().bind(DecisionSnapshotPoolSettings.class).max());
  }

  public static EpsilonDecisionSnapshotPool load(Path checkpointDir, int maxSnapshots)
      throws IOException {
    EpsilonDecisionSnapshotPool pool = new EpsilonDecisionSnapshotPool(maxSnapshots);
    Path file = checkpointDir.resolve(REGISTRY_FILE);
    if (!Files.exists(file)) {
      return pool;
    }
    String json = Files.readString(file);
    Registry registry;
    try {
      JsonElement parsed = JsonParser.parseString(json);
      if (!parsed.isJsonObject()) {
        throw new IOException("Decision snapshot registry root must be an object");
      }
      JsonObject root = parsed.getAsJsonObject();
      if (!root.has("format") || !root.has("version")) {
        throw new IOException(
            "Unsupported Decision snapshot registry format/version; use a fresh current-format"
                + " checkpoint root");
      }
      requireExactFields(root, REGISTRY_FIELDS, "registry");
      if (CURRENT_REGISTRY_FORMAT.equals(root.get("format").getAsString())
          && root.get("version").getAsInt() == 2) {
        for (JsonElement element : root.getAsJsonArray("snapshots")) {
          JsonObject entry = element.getAsJsonObject();
          JsonElement version = entry.remove("version");
          entry.add("id", version);
          entry.add("iteration", version);
          entry.remove("candidateId");
        }
        root.addProperty("version", CURRENT_REGISTRY_VERSION);
      }
      if (!root.get("format").isJsonPrimitive()
          || !CURRENT_REGISTRY_FORMAT.equals(root.get("format").getAsString())
          || !root.get("version").isJsonPrimitive()
          || root.get("version").getAsInt() != CURRENT_REGISTRY_VERSION) {
        throw new IOException(
            "Unsupported Decision snapshot registry format/version; use a fresh current-format"
                + " checkpoint root");
      }
      if (!root.get("snapshots").isJsonArray()) {
        throw new IOException("Decision snapshot registry snapshots must be an array");
      }
      for (JsonElement entry : root.getAsJsonArray("snapshots")) {
        if (!entry.isJsonObject()) {
          throw new IOException("Decision snapshot registry entry must be an object");
        }
        JsonObject snapshot = entry.getAsJsonObject();
        requireExactFields(snapshot, SNAPSHOT_FIELDS, "snapshot entry");
        JsonElement evalStats = snapshot.get("evalStats");
        if (evalStats == null || !evalStats.isJsonObject()) {
          throw new IOException("Decision snapshot registry evalStats must be an object");
        }
        requireExactFields(evalStats.getAsJsonObject(), EVAL_FIELDS, "snapshot evalStats");
      }
      registry = GSON.fromJson(root, Registry.class);
    } catch (IOException e) {
      throw e;
    } catch (RuntimeException e) {
      throw new IOException("Invalid Decision snapshot registry: " + file, e);
    }
    if (registry == null
        || !CURRENT_REGISTRY_FORMAT.equals(registry.format())
        || registry.version() != CURRENT_REGISTRY_VERSION
        || registry.snapshots() == null) {
      throw new IOException("Invalid Decision snapshot registry schema: " + file);
    }
    HashSet<Long> ids = new HashSet<>();
    ArrayList<SnapshotEntry> canonicalSnapshots = new ArrayList<>();
    int currentArenaChampionCount = 0;
    for (SnapshotEntry snapshot : registry.snapshots()) {
      SnapshotEntry canonical = requireCanonicalSnapshot(snapshot);
      if (!ids.add(canonical.id())) {
        throw new IOException(
            "Decision snapshot registry contains duplicate id: " + canonical.id());
      }
      if (canonical.currentArenaChampion()) {
        currentArenaChampionCount++;
      }
      canonicalSnapshots.add(canonical);
    }
    if (currentArenaChampionCount > 1) {
      throw new IOException(
          "Decision snapshot registry contains multiple current arena champion entries");
    }
    if (!canonicalSnapshots.isEmpty() && currentArenaChampionCount == 0) {
      throw new IOException("Decision snapshot registry has no current arena champion entry");
    }
    for (SnapshotEntry canonical : canonicalSnapshots) {
      requireSnapshotCheckpoint(canonical);
      pool.addSnapshot(canonical);
    }
    return pool;
  }

  private static void requireSingleCurrentArenaChampion(List<SnapshotEntry> entries)
      throws IOException {
    long count = entries.stream().filter(SnapshotEntry::currentArenaChampion).count();
    if (count > 1) {
      throw new IOException(
          "Decision snapshot registry contains multiple current arena champion entries");
    }
    if (!entries.isEmpty() && count == 0) {
      throw new IOException("Decision snapshot registry has no current arena champion entry");
    }
  }

  /** 登録情報が実在する変更不可チェックポイントの反復回数・更新回数に一致することを検証する。 */
  static Path requireSnapshotCheckpoint(SnapshotEntry snapshot) throws IOException {
    Path checkpoint = Path.of(snapshot.path()).toAbsolutePath().normalize();
    Path resolved = EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpoint);
    if (resolved == null || !checkpoint.equals(resolved.toAbsolutePath().normalize())) {
      throw new IOException(
          "Decision snapshot checkpoint does not resolve to the exact checkpoint: id="
              + snapshot.id()
              + " path="
              + checkpoint
              + " resolved="
              + resolved);
    }
    EpsilonDecisionCheckpointBundle checkpointManifest =
        EpsilonDecisionCheckpointManager.loadManifest(checkpoint);
    if (checkpointManifest.iteration != snapshot.iteration()
        || checkpointManifest.globalStep != snapshot.createdStep()) {
      throw new IOException(
          "Decision snapshot checkpoint generation mismatch: snapshotId="
              + snapshot.id()
              + " checkpoint="
              + checkpoint
              + " manifestIteration="
              + checkpointManifest.iteration
              + " manifestStep="
              + checkpointManifest.globalStep
              + " expectedIteration="
              + snapshot.iteration()
              + " expectedStep="
              + snapshot.createdStep());
    }
    return checkpoint;
  }

  private static void requireExactFields(JsonObject object, Set<String> expected, String label)
      throws IOException {
    if (!object.keySet().equals(expected)) {
      throw new IOException(
          "Decision snapshot "
              + label
              + " fields mismatch: expected="
              + expected
              + " actual="
              + object.keySet());
    }
  }

  private static SnapshotEntry requireCanonicalSnapshot(SnapshotEntry snapshot) throws IOException {
    if (snapshot == null) {
      throw new IOException("Decision snapshot registry contains a null entry");
    }
    if (snapshot.id() < 0L) {
      throw new IOException("Decision snapshot registry contains a negative id: " + snapshot.id());
    }
    try {
      return new SnapshotEntry(
          snapshot.path(),
          snapshot.id(),
          snapshot.iteration(),
          snapshot.createdStep(),
          snapshot.currentArenaChampion(),
          snapshot.evalStats());
    } catch (IllegalArgumentException e) {
      throw new IOException(
          "Decision snapshot registry contains an invalid entry: id=" + snapshot.id(), e);
    }
  }

  /**
   * スナップショット登録先に保存する変更不可チェックポイント項目。
   *
   * @param path チェックポイントの正規化済み絶対パス
   * @param id 登録先内のスナップショットID
   * @param iteration チェックポイントの反復回数
   * @param createdStep チェックポイント作成時の累積更新回数
   * @param currentArenaChampion 現在の対局収集用の採用モデルへの参照なら {@code true}
   * @param evalStats チェックポイントの保存済み評価指標
   */
  public record SnapshotEntry(
      String path,
      long id,
      int iteration,
      int createdStep,
      boolean currentArenaChampion,
      EvalStats evalStats) {

    /** ID、反復回数、絶対パス、評価値を検証して項目を構築する。 */
    public SnapshotEntry {
      if (id < 0L || iteration < 0) {
        throw new IllegalArgumentException(
            "Decision snapshot id and iteration must be non-negative: " + id + "/" + iteration);
      }
      if (createdStep < 0) {
        throw new IllegalArgumentException(
            "Decision snapshot createdStep must be non-negative: " + createdStep);
      }
      if (path == null || path.isBlank()) {
        throw new IllegalArgumentException("Decision snapshot checkpoint path must not be blank");
      }
      Path checkpoint;
      try {
        checkpoint = Path.of(path);
      } catch (RuntimeException e) {
        throw new IllegalArgumentException(
            "Decision snapshot checkpoint path is invalid: " + path, e);
      }
      if (!checkpoint.isAbsolute()) {
        throw new IllegalArgumentException(
            "Decision snapshot checkpoint path must be absolute: " + path);
      }
      path = checkpoint.normalize().toString();
      if (evalStats == null) {
        throw new IllegalArgumentException("Decision snapshot evalStats must not be null");
      }
    }
  }

  /**
   * スナップショットに付随する最小対戦評価指標。
   *
   * @param averageRank 候補の平均順位
   * @param topRate 候補の1着率
   * @param lastRate 候補の4着率
   */
  public record EvalStats(float averageRank, float topRate, float lastRate) {

    /** すべての評価値が有限であることを検証する。 */
    public EvalStats {
      if (!Float.isFinite(averageRank) || !Float.isFinite(topRate) || !Float.isFinite(lastRate)) {
        throw new IllegalArgumentException("Decision snapshot evalStats must be finite");
      }
    }

    /**
     * 対戦評価未実施を表す全ゼロ指標を返す。
     *
     * @return 全項目0の評価指標
     */
    public static EvalStats empty() {
      return new EvalStats(0.0f, 0.0f, 0.0f);
    }

    /**
     * 対戦評価結果から候補の最小登録先指標を抽出する。
     *
     * @param result 候補と対戦相手の対戦評価結果
     * @return 候補の平均順位、1着率、4着率
     */
    public static EvalStats fromDuelResult(DuelEvaluation.Result result) {
      return new EvalStats(
          (float) result.candidateAverageRank(),
          (float) result.candidateTopRate(),
          (float) result.candidateLastRate());
    }
  }

  private record Registry(String format, int version, List<SnapshotEntry> snapshots) {}
}

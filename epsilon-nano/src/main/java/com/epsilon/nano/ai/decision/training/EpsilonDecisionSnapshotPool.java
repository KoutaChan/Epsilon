package com.epsilon.nano.ai.decision.training;

import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.config.settings.DecisionSnapshotPoolSettings;
import com.epsilon.core.GameState;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.util.SeedMixer;
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
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/** 自己対局の対戦相手に使う、過去に採用されたモデルのスナップショットを管理する。 */
public final class EpsilonDecisionSnapshotPool {

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final String REGISTRY_FILE = "decision-snapshot-pool.json";
  static final String CURRENT_REGISTRY_FORMAT = "epsilon-decision-arena-snapshot-pool";
  static final int CURRENT_REGISTRY_VERSION = 2;
  private static final Set<String> REGISTRY_FIELDS = Set.of("format", "version", "snapshots");
  private static final Set<String> SNAPSHOT_FIELDS =
      Set.of("path", "version", "createdStep", "currentArenaChampion", "evalStats", "candidateId");
  private static final Set<String> EVAL_FIELDS = Set.of("averageRank", "topRate", "lastRate");
  private static final long OPPONENT_SEAT_SALT = 0x9FB21C651E98DF25L;

  private final int maxSnapshots;
  private final ArrayList<SnapshotEntry> snapshots = new ArrayList<>();

  /** 設定の保持上限を使って空の対局スナップショット集約を作る。 */
  public EpsilonDecisionSnapshotPool() {
    this(EpsilonSettings.defaults().bind(DecisionSnapshotPoolSettings.class).max());
  }

  /**
   * 指定した保持上限で空の対局スナップショット集約を作る。
   *
   * @param maxSnapshots 登録一覧に保持する最大チェックポイント数
   */
  public EpsilonDecisionSnapshotPool(int maxSnapshots) {
    if (maxSnapshots <= 0) {
      throw new IllegalArgumentException("maxSnapshots must be positive");
    }
    this.maxSnapshots = maxSnapshots;
  }

  /**
   * 対局収集に使う採用モデルチェックポイント登録情報をバージョンで置換または追加する。
   *
   * <p>上限を超えた場合は現在の対局収集に使う採用モデルを残し、最古の非現在の登録情報を削除する。
   *
   * @param snapshot 検証済み変更不可チェックポイント登録情報
   */
  public synchronized void addArenaSnapshot(SnapshotEntry snapshot) {
    if (snapshot.currentArenaChampion()) {
      for (int i = 0; i < snapshots.size(); i++) {
        SnapshotEntry existing = snapshots.get(i);
        if (existing.version() != snapshot.version() && existing.currentArenaChampion()) {
          snapshots.set(i, withCurrentArenaChampion(existing, false));
        }
      }
    }
    long snapshotVersion = snapshot.version();
    snapshots.removeIf(existing -> existing.version() == snapshotVersion);
    snapshots.add(snapshot);
    snapshots.sort(Comparator.comparingLong(SnapshotEntry::version));
    while (snapshots.size() > maxSnapshots) {
      removeOldestEvictableSnapshot();
    }
  }

  /**
   * 指定バージョンを現在の対局収集に使う採用モデルとして固定する。既存登録情報があっても、検証済み変更不可チェックポイントの {@code fallbackPath} と候補 IDへ差し替える。
   *
   * @param version 現在の対局収集に使う採用モデルにするスナップショットバージョン
   * @param fallbackPath 検証済みチェックポイントの絶対パス
   * @param fallbackCreatedStep チェックポイント作成時の累積更新回数
   * @param fallbackCandidateId 保存先から決まる候補の不変ID
   */
  public synchronized void markArenaChampion(
      long version, String fallbackPath, int fallbackCreatedStep, String fallbackCandidateId) {
    boolean found = false;
    for (int i = 0; i < snapshots.size(); i++) {
      SnapshotEntry entry = snapshots.get(i);
      if (entry.version() == version) {
        snapshots.set(
            i,
            new SnapshotEntry(
                fallbackPath,
                version,
                fallbackCreatedStep,
                true,
                entry.evalStats(),
                fallbackCandidateId));
        found = true;
      } else if (entry.currentArenaChampion()) {
        snapshots.set(i, withCurrentArenaChampion(entry, false));
      }
    }
    if (!found) {
      addArenaSnapshot(
          new SnapshotEntry(
              fallbackPath,
              version,
              fallbackCreatedStep,
              true,
              EvalStats.empty(),
              fallbackCandidateId));
    }
  }

  /** 対局収集に使う採用モデルを現在のスナップショットとして登録する。 */
  public void registerArenaChampion(Path champion) throws IOException {
    Path resolved = EpsilonDecisionCheckpointManager.resolveExistingStrict(champion);
    if (resolved == null) {
      throw new IOException("Arena champion checkpoint not found: " + champion);
    }
    EpsilonDecisionCheckpointBundle manifest =
        EpsilonDecisionCheckpointManager.loadManifest(resolved);
    String candidateId = EpsilonDecisionCheckpointManager.candidateId(resolved);
    markArenaChampion(
        manifest.iteration,
        resolved.toAbsolutePath().normalize().toString(),
        manifest.globalStep,
        candidateId);
  }

  /** 現在の対局収集に使う採用モデルだけを固定し、それ以外の対局スナップショットを古い順に削除する。 */
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
        snapshot.version(),
        snapshot.createdStep(),
        currentArenaChampion,
        snapshot.evalStats(),
        snapshot.candidateId());
  }

  /**
   * 方策モデル席ごとの3対戦相手を選ぶ。現在の対局収集に使う採用モデルが候補自身でない限り、各対戦相手の組合せへ必ず1席以上含める。
   *
   * @param candidateVersion 対戦相手から除外する現在の候補バージョン
   * @param seed 対戦相手の組合せを再現する乱数シード
   * @param maxDistinctSnapshots 一つの収集で利用するスナップショット種類の上限
   * @return {@code [actorSeat][opponentSeatSlot]} のスナップショットバージョン
   */
  public synchronized long[][] sampleOpponentIdsForSeats(
      long candidateVersion, long seed, int maxDistinctSnapshots) {
    ArrayList<SnapshotEntry> eligible = eligibleArenaSnapshots(candidateVersion);
    if (eligible.isEmpty()) {
      throw new IllegalStateException(
          "Decision snapshot pool has no eligible arena opponent: candidateVersion="
              + candidateVersion);
    }
    int distinctCount = Math.min(maxDistinctSnapshots, eligible.size());
    long[] distinctIds = sampleDistinctSnapshotIds(eligible, distinctCount, seed);
    long currentArenaChampionId = currentArenaChampionId(eligible);
    long[][] ids = new long[GameState.NUM_PLAYERS][];
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      SplittableRandom rng =
          new SplittableRandom(SeedMixer.indexed(seed, OPPONENT_SEAT_SALT, seat));
      ids[seat] = new long[GameState.NUM_PLAYERS - 1];
      int currentArenaChampionSlot =
          currentArenaChampionId >= 0L ? rng.nextInt(ids[seat].length) : -1;
      for (int i = 0; i < ids[seat].length; i++) {
        ids[seat][i] =
            i == currentArenaChampionSlot
                ? currentArenaChampionId
                : distinctIds[rng.nextInt(distinctIds.length)];
      }
    }
    return ids;
  }

  private ArrayList<SnapshotEntry> eligibleArenaSnapshots(long candidateVersion) {
    ArrayList<SnapshotEntry> eligible = new ArrayList<>();
    for (SnapshotEntry snapshot : snapshots) {
      if (snapshot.version() != candidateVersion) {
        eligible.add(snapshot);
      }
    }
    return eligible;
  }

  private static long[] sampleDistinctSnapshotIds(
      ArrayList<SnapshotEntry> eligible, int distinctCount, long seed) {
    ArrayList<SnapshotEntry> remaining = new ArrayList<>(eligible);
    SplittableRandom rng = new SplittableRandom(seed ^ 0x5EED5EEDL);
    long[] ids = new long[distinctCount];
    int next = 0;
    for (SnapshotEntry snapshot : eligible) {
      if (snapshot.currentArenaChampion() && next < ids.length) {
        ids[next++] = snapshot.version();
        remaining.remove(snapshot);
        break;
      }
    }
    for (int i = next; i < ids.length; i++) {
      int index = rng.nextInt(remaining.size());
      ids[i] = remaining.remove(index).version();
    }
    return ids;
  }

  private static long currentArenaChampionId(ArrayList<SnapshotEntry> eligible) {
    for (SnapshotEntry snapshot : eligible) {
      if (snapshot.currentArenaChampion()) {
        return snapshot.version();
      }
    }
    return -1L;
  }

  /**
   * 登録一覧内の対局スナップショット数を返す。
   *
   * @return 保持中登録情報数
   */
  public synchronized int size() {
    return snapshots.size();
  }

  /**
   * 指定バージョンのスナップショット登録情報を検索する。
   *
   * @param version 検索するスナップショットバージョン
   * @return 対応登録情報。存在しない場合は {@code null}
   */
  public synchronized SnapshotEntry snapshot(long version) {
    for (SnapshotEntry snapshot : snapshots) {
      if (snapshot.version() == version) {
        return snapshot;
      }
    }
    return null;
  }

  /**
   * 登録一覧をチェックポイントのルートディレクトリへ不可分な操作で保存する。
   *
   * @param checkpointDir 登録一覧ファイルを置くチェックポイントのルートディレクトリ
   * @throws IOException 現在の対局収集に使う採用モデル、チェックポイント識別情報、または書込の検証に失敗した場合
   */
  public synchronized void save(Path checkpointDir) throws IOException {
    requireSingleCurrentArenaChampion(snapshots);
    for (SnapshotEntry snapshot : snapshots) {
      requireArenaChampionCheckpoint(snapshot);
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
   * チェックポイントのルートディレクトリの現在の-形式登録一覧を厳密に読み込む。
   *
   * @param checkpointDir 登録一覧ファイルを持つチェックポイントのルートディレクトリ
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
    HashSet<Long> versions = new HashSet<>();
    ArrayList<SnapshotEntry> canonicalSnapshots = new ArrayList<>();
    int currentArenaChampionCount = 0;
    for (SnapshotEntry snapshot : registry.snapshots()) {
      SnapshotEntry canonical = requireCanonicalSnapshot(snapshot);
      if (!versions.add(canonical.version())) {
        throw new IOException(
            "Decision snapshot registry contains duplicate version: " + canonical.version());
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
      requireArenaChampionCheckpoint(canonical);
      pool.addArenaSnapshot(canonical);
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

  /** 登録情報が実在する変更不可チェックポイントの反復回数と候補 ID に一致することを検証する。 */
  static Path requireArenaChampionCheckpoint(SnapshotEntry snapshot) throws IOException {
    Path checkpoint = Path.of(snapshot.path()).toAbsolutePath().normalize();
    Path resolved = EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpoint);
    if (resolved == null || !checkpoint.equals(resolved.toAbsolutePath().normalize())) {
      throw new IOException(
          "Decision snapshot checkpoint does not resolve to the exact checkpoint: version="
              + snapshot.version()
              + " path="
              + checkpoint
              + " resolved="
              + resolved);
    }
    EpsilonDecisionCheckpointBundle checkpointManifest =
        EpsilonDecisionCheckpointManager.loadManifest(checkpoint);
    if (checkpointManifest.iteration != snapshot.version()) {
      throw new IOException(
          "Decision snapshot checkpoint iteration mismatch: snapshotId="
              + snapshot.version()
              + " checkpoint="
              + checkpoint
              + " manifestIteration="
              + checkpointManifest.iteration);
    }
    String candidateId = EpsilonDecisionCheckpointManager.candidateId(checkpoint);
    if (!snapshot.candidateId().equals(candidateId)) {
      throw new IOException(
          "Decision snapshot checkpoint identity mismatch: snapshotId="
              + snapshot.version()
              + " checkpoint="
              + checkpoint
              + " expectedCandidateId="
              + snapshot.candidateId()
              + " actualCandidateId="
              + candidateId);
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
    if (snapshot.version() < 0L) {
      throw new IOException(
          "Decision snapshot registry contains a negative version: " + snapshot.version());
    }
    try {
      return new SnapshotEntry(
          snapshot.path(),
          snapshot.version(),
          snapshot.createdStep(),
          snapshot.currentArenaChampion(),
          snapshot.evalStats(),
          snapshot.candidateId());
    } catch (IllegalArgumentException e) {
      throw new IOException(
          "Decision snapshot registry contains an invalid entry: version=" + snapshot.version(), e);
    }
  }

  /**
   * スナップショット登録一覧に保存する変更不可チェックポイント登録情報。
   *
   * @param path チェックポイントの正規化済み絶対パス
   * @param version 登録一覧内の単調増加バージョン
   * @param createdStep チェックポイント作成時の累積更新回数
   * @param currentArenaChampion 現在の対局採用モデルへの参照なら {@code true}
   * @param evalStats チェックポイントの保存済み評価指標
   * @param candidateId 保存先から決まる候補の不変ID
   */
  public record SnapshotEntry(
      String path,
      long version,
      int createdStep,
      boolean currentArenaChampion,
      EvalStats evalStats,
      String candidateId) {

    /** バージョン、絶対パス、評価値、候補 ID を検証して登録情報を構築する。 */
    public SnapshotEntry {
      if (version < 0L) {
        throw new IllegalArgumentException(
            "Decision snapshot version must be non-negative: " + version);
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
      if (candidateId == null || candidateId.isBlank()) {
        throw new IllegalArgumentException("Decision snapshot candidateId must not be blank");
      }
    }
  }

  /**
   * スナップショットに付随する最小対戦比較指標。
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
     * 対戦比較未実施を表す全ゼロ指標を返す。
     *
     * @return 全項目0の評価指標
     */
    public static EvalStats empty() {
      return new EvalStats(0.0f, 0.0f, 0.0f);
    }

    /**
     * 対戦比較結果から候補の最小登録一覧指標を抽出する。
     *
     * @param result 候補と対戦相手の対戦比較結果
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

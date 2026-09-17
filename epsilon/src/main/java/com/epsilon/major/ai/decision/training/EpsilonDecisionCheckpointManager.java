package com.epsilon.major.ai.decision.training;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.major.ai.decision.EpsilonDecisionConstants;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionModelCopies;
import com.epsilon.major.ai.model.EpsilonDecisionNetwork;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.major.config.settings.DecisionInferenceSettings;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Decision チェックポイントの保存と読み込み。 */
public final class EpsilonDecisionCheckpointManager {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionCheckpointManager.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final String MODEL_PREFIX = "decision";
  private static final Pattern CANDIDATE_RUN_ID =
      Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,127}");

  private EpsilonDecisionCheckpointManager() {}

  /**
   * 新規学習開始時の更新段階・反復回数・対局数0として初期一組のデータを保存する。
   *
   * @param model 保存する標準形式の Decision モデル
   * @param dir 新規チェックポイントディレクトリ
   * @throws IOException パラメーター・マニフェスト保存またはディレクトリ置換に失敗した場合
   */
  public static void saveInitial(Model model, Path dir) throws IOException {
    save(model, dir, 0, 0, 0);
  }

  /**
   * 更新の基準となるモデルを一つのチェックポイント一組のデータとして原子的に保存する。
   *
   * @param model 保存する標準形式の Decision モデル
   * @param dir チェックポイントディレクトリ
   * @param globalStep オプティマイザー更新回数
   * @param iteration 学習反復回数
   * @param selfPlayGames 累積自己対局数
   * @throws IOException パラメーター・マニフェスト保存またはディレクトリ置換に失敗した場合
   */
  public static void save(Model model, Path dir, int globalStep, int iteration, int selfPlayGames)
      throws IOException {
    saveBundle(model, null, dir, globalStep, iteration, selfPlayGames, null);
  }

  /** 上書きを繰り返す {@code working/current}だけにモデル、AdamW、KL制御状態を同じ原子的な一組のデータとして保存する。 */
  static void saveWorking(
      DecisionLearner learner, Path checkpointDir, int globalStep, int iteration, int selfPlayGames)
      throws IOException {
    saveWorking(learner, checkpointDir, globalStep, iteration, selfPlayGames, null);
  }

  @FunctionalInterface
  interface BundleWriter {
    void write(Path directory) throws IOException;
  }

  /** 一連の学習実行/比較実験の進行状態も学習器と同じ確定境界へ含める。 */
  static void saveWorking(
      DecisionLearner learner,
      Path checkpointDir,
      int globalStep,
      int iteration,
      int selfPlayGames,
      BundleWriter additionalBundleWriter)
      throws IOException {
    Path current = working(checkpointDir);
    saveLearner(
        learner,
        current,
        globalStep,
        iteration,
        selfPlayGames,
        directory -> {
          for (String sidecar :
              List.of(DecisionSelectedPgCampaignState.FILE_NAME, "trial-state.json")) {
            Path previous = current.resolve(sidecar);
            if (Files.isRegularFile(previous)) Files.copy(previous, directory.resolve(sidecar));
          }
          if (additionalBundleWriter != null) additionalBundleWriter.write(directory);
        });
  }

  /** ロールバック用比較元など、指定パスへ学習器一組のデータを保存する。 */
  static void saveLearner(
      DecisionLearner learner,
      Path directory,
      int globalStep,
      int iteration,
      int selfPlayGames,
      BundleWriter additionalBundleWriter)
      throws IOException {
    saveBundle(
        learner.model(),
        learner,
        directory,
        globalStep,
        iteration,
        selfPlayGames,
        additionalBundleWriter);
  }

  /** モデルとAdamW/KLだけを複製し、一連の学習実行/比較実験の進行状態は引き継がない。 */
  static void copyLearnerCheckpoint(Path source, Path target) throws IOException {
    requireValidCheckpoint(source);
    List<Path> artifacts = new ArrayList<>(checkpointArtifacts(source));
    artifacts.addAll(DecisionLearnerCheckpoint.artifacts(source));
    Path destination = target.toAbsolutePath().normalize();
    Files.createDirectories(destination.getParent());
    Path temporary = Files.createTempDirectory(destination.getParent(), ".learner-copy-");
    try {
      for (Path artifact : artifacts) {
        Files.copy(artifact, temporary.resolve(artifact.getFileName()));
      }
      replaceDirectory(temporary, destination);
    } finally {
      deleteDirectoryQuietly(temporary);
    }
  }

  /** ディレクトリ置換の名前変更間で停止した場合は、最後の確定済更新中のへ戻す。 */
  static void recoverWorkingCheckpoint(Path root) throws IOException {
    Path target = working(root).toAbsolutePath().normalize();
    if (Files.exists(target) || !Files.isDirectory(target.getParent())) return;
    try (Stream<Path> paths = Files.list(target.getParent())) {
      Path backup =
          paths
              .filter(p -> p.getFileName().toString().startsWith(".current.old-"))
              .max(
                  Comparator.comparingLong(
                      p -> {
                        try {
                          return Files.getLastModifiedTime(p).toMillis();
                        } catch (IOException failure) {
                          throw new java.io.UncheckedIOException(failure);
                        }
                      }))
              .orElse(null);
      if (backup != null) {
        requireValidCheckpoint(backup);
        moveDirectory(backup, target);
      }
    }
  }

  private static void saveBundle(
      Model model,
      DecisionLearner learner,
      Path dir,
      int globalStep,
      int iteration,
      int selfPlayGames,
      BundleWriter additionalBundleWriter)
      throws IOException {
    EpsilonDecisionModelCopies.requireDecisionModel(model, "checkpoint");
    Path target = dir.toAbsolutePath().normalize();
    Path parent = target.getParent();
    if (parent == null) {
      parent = Path.of(".").toAbsolutePath().normalize();
    }
    Files.createDirectories(parent);
    Path tmp = Files.createTempDirectory(parent, "." + target.getFileName() + ".tmp-");
    boolean installed = false;
    try {
      saveContents(model, tmp, globalStep, iteration, selfPlayGames);
      CheckpointDescriptor descriptor =
          loadDescriptor(tmp, ((EpsilonDecisionNetwork) model.getBlock()).utilityProfile());
      requireCheckpointArtifacts(tmp, descriptor);
      if (learner != null) {
        learner.saveCheckpointState(tmp, descriptor.bundle());
      }
      if (additionalBundleWriter != null) additionalBundleWriter.write(tmp);
      replaceDirectory(tmp, target);
      installed = true;
      log.info(
          "Decision checkpoint saved: dir={}, step={}, iteration={}, selfPlayGames={},"
              + " optimizer={}",
          dir,
          globalStep,
          iteration,
          selfPlayGames,
          learner != null);
    } catch (Exception e) {
      throw asIOException("Failed to save Decision checkpoint: " + dir, e);
    } finally {
      if (!installed) {
        deleteDirectoryQuietly(tmp);
      }
    }
  }

  /** 指定反復回数のチェックポイントを変更不可スナップショットとして複製する。 */
  static void copyAtIteration(Path source, Path target, int expectedIteration) throws IOException {
    Path normalizedSource = source.toAbsolutePath().normalize();
    EpsilonDecisionCheckpointBundle sourceManifest = requireValidCheckpoint(normalizedSource);
    if (sourceManifest.iteration != expectedIteration) {
      throw new IOException(
          "Decision checkpoint iteration mismatch: expected="
              + expectedIteration
              + " actual="
              + sourceManifest.iteration
              + " source="
              + source);
    }
    Path resolved = normalizedSource;
    Path destination = target.toAbsolutePath().normalize();
    if (resolved.toAbsolutePath().normalize().equals(destination)) {
      return;
    }
    Path parent = destination.getParent();
    if (parent == null) {
      parent = Path.of(".").toAbsolutePath().normalize();
    }
    Files.createDirectories(parent);
    // 変更不可スナップショットへは現行チェックポイント本体だけを複製し、旧・未知の付随情報を継承しない。
    List<Path> artifactFiles = checkpointArtifacts(resolved);
    Path tmp = Files.createTempDirectory(parent, "." + destination.getFileName() + ".tmp-");
    boolean installed = false;
    try {
      for (Path artifact : artifactFiles) {
        Files.copy(
            artifact,
            tmp.resolve(artifact.getFileName()),
            StandardCopyOption.COPY_ATTRIBUTES,
            StandardCopyOption.REPLACE_EXISTING);
      }
      EpsilonDecisionCheckpointBundle manifest = requireValidCheckpoint(tmp);
      if (manifest.iteration != expectedIteration) {
        throw new IOException(
            "Copied Decision checkpoint iteration mismatch: expected="
                + expectedIteration
                + " actual="
                + manifest.iteration
                + " source="
                + source);
      }
      replaceDirectory(tmp, destination);
      installed = true;
    } finally {
      if (!installed) {
        deleteDirectoryQuietly(tmp);
      }
    }
  }

  /**
   * 優先するデバイスへチェックポイントを読み込む。
   *
   * @param dir チェックポイントディレクトリ
   * @return パラメーターを所有する標準形式の Decision モデル
   * @throws IOException 保存物がない、非互換、または読込失敗の場合
   */
  public static Model load(Path dir) throws IOException {
    return load(dir, NetworkFactory.getPreferredDevice());
  }

  /** 優先するデバイスへチェックポイントを一度だけ読み込み、凍結した推論専用モデルを返す。 */
  public static Model loadForInference(Path dir) throws IOException {
    return loadForInference(dir, NetworkFactory.getPreferredDevice());
  }

  /**
   * チェックポイントを推論用に凍結し、GPUのみ設定されたパラメーター精度へ変換する。 返却モデルは呼び出し側が所有し、サーバーより後に解放する。学習再開には{@link #load(Path,
   * Device)}を使う。
   */
  public static Model loadForInference(Path dir, Device device) throws IOException {
    return loadForInference(dir, device, DecisionInferenceSettings.defaults());
  }

  /** 保存されたネットワーク情報で復元し、この実行時の推論設定だけを適用する。 */
  public static Model loadForInference(Path dir, Device device, DecisionInferenceSettings settings)
      throws IOException {
    return EpsilonDecisionModelCopies.prepareOwnedInferenceModel(load(dir, device), settings);
  }

  /**
   * 指定デバイスへチェックポイントを読み込み、構造をログへ記録する。
   *
   * @param dir チェックポイントディレクトリ
   * @param device パラメーターを配置するデバイス
   * @return パラメーターを所有する標準形式の Decision モデル
   * @throws IOException 保存物がない、非互換、または読込失敗の場合
   */
  public static Model load(Path dir, Device device) throws IOException {
    return load(dir, device, true);
  }

  /**
   * 指定デバイスへチェックポイントを読み込む。
   *
   * @param dir チェックポイントディレクトリ
   * @param device パラメーターを配置するデバイス
   * @param logArchitecture 構造要約をログへ出すならtrue
   * @return パラメーターを所有する標準形式の Decision モデル
   * @throws IOException 保存物がない、非互換、または読込失敗の場合
   */
  public static Model load(Path dir, Device device, boolean logArchitecture) throws IOException {
    return loadModel(dir, device, logArchitecture);
  }

  private static Model loadModel(Path dir, Device device, boolean logArchitecture)
      throws IOException {
    if (dir == null) {
      throw new IOException("Decision checkpoint path is null");
    }
    CheckpointDescriptor descriptor = loadDescriptor(dir);
    requireCheckpointArtifacts(dir, descriptor);
    Model model =
        NetworkFactory.createDecisionModel(
            device,
            logArchitecture,
            descriptor.bundle().hidden,
            profileFor(descriptor.bundle().valueDefinition));
    try {
      model.load(dir, descriptor.modelPrefix());
      return model;
    } catch (Exception e) {
      model.close();
      throw new IOException("Failed to load Decision model from " + dir, e);
    }
  }

  /**
   * ディレクトリが現在スキーマで読み込める完全なチェックポイントかを返す。
   *
   * @param dir 検査するディレクトリ
   * @return マニフェスト・保存物が揃い互換ならtrue
   */
  public static boolean isValidCheckpoint(Path dir) {
    if (dir == null) {
      return false;
    }
    try {
      requireValidCheckpoint(dir);
      return true;
    } catch (IOException e) {
      return false;
    }
  }

  /**
   * ディレクトリが現在スキーマで読み込める完全なチェックポイントであることを要求する。
   *
   * <p>学習・複製・対局への投入などの本番経路では、原因を破棄する {@link #isValidCheckpoint(Path)} ではなくこちらを使う。
   *
   * @param dir 検証するチェックポイントディレクトリ
   * @return 検証済みマニフェスト
   * @throws IOException マニフェスト、保存物、構造の契約に違反した場合
   */
  public static EpsilonDecisionCheckpointBundle requireValidCheckpoint(Path dir)
      throws IOException {
    CheckpointDescriptor descriptor = loadDescriptor(dir);
    requireCheckpointArtifacts(dir, descriptor);
    return descriptor.bundle();
  }

  /**
   * 新規約の未承認・再開用チェックポイントパスを返す。
   *
   * <p>対局生成・評価はこのパスを参照しない。
   *
   * @param checkpointDir 候補の評価・保存手順親ディレクトリ
   * @return {@code working/current} パス
   */
  public static Path working(Path checkpointDir) {
    return checkpointDir.resolve("working").resolve("current");
  }

  /**
   * 継続学習用対局収集用の採用モデルを指す候補の評価・保存手順参照を厳格に解決する。
   *
   * @param checkpointDir 候補の評価・保存手順親ディレクトリ
   * @return 対局収集用の採用モデルチェックポイント。未作成ならnull
   * @throws IOException 参照または保存物が壊れている場合
   */
  public static Path resolveArenaChampionStrict(Path checkpointDir) throws IOException {
    return EpsilonDecisionChampionStore.resolveArenaChampionStrict(checkpointDir);
  }

  /** 通常推論に使う本番採用モデルへの参照を厳格に解決する。 */
  public static Path resolveProductionChampionStrict(Path checkpointDir) throws IOException {
    return EpsilonDecisionChampionStore.resolveProductionChampionStrict(checkpointDir);
  }

  /**
   * チェックポイント保存物が存在しない場合だけ {@code null} を返す厳格版。
   *
   * <p>マニフェスト/モデルの片方だけ、またはバージョン/構造不一致は新規モデルへ黙って代替処理せず
   * 直ちに例外を送出する。学習や行動選択プレイヤー起動などチェックポイントを上書き得る経路ではこちらを使う。
   *
   * @param checkpointDir 候補の評価・保存手順親ディレクトリまたは明示チェックポイントパス
   * @return 本番採用モデルまたは明示チェックポイント。保存物が全くなければnull
   * @throws IOException 部分保存物または非互換チェックポイントを検出した場合
   */
  public static Path resolveExistingStrict(Path checkpointDir) throws IOException {
    rejectLegacyCheckpointLayout(checkpointDir);
    Path production = resolveProductionChampionStrict(checkpointDir);
    if (production != null) {
      return production;
    }
    Path explicit = validateCandidateIfPresent(checkpointDir);
    if (explicit != null) {
      return explicit;
    }
    return null;
  }

  /**
   * 候補の評価・保存手順導入前の変更可能な参照を検出して明示的に拒否する。
   *
   * <p>旧配置を単に無視すると、事前学習が同じ親ディレクトリで新規モデルを作り、既存チェックポイントを暗黙に取り違える。 移行
   * 代替処理は持たず、利用者に新しい親ディレクトリでの再学習を要求する。
   */
  private static void rejectLegacyCheckpointLayout(Path checkpointDir) throws IOException {
    if (checkpointDir == null) {
      return;
    }
    Path root = checkpointDir.toAbsolutePath().normalize();
    List<Path> legacyPaths =
        List.of(
            root.resolve("latest"),
            root.resolve("paired-ach").resolve("current"),
            root.resolve("train_latest"));
    for (Path legacy : legacyPaths) {
      if (Files.exists(legacy)) {
        throw new IOException(
            "Unsupported legacy Decision checkpoint layout: "
                + legacy
                + "; use a fresh current-format checkpoint root");
      }
    }
  }

  /** 評価対象を固定して保存し、再開時は同じ保存内容であることを確認する。 */
  static synchronized Path saveCandidate(Path checkpointRoot, String runId, Path source)
      throws IOException {
    if (runId == null || !CANDIDATE_RUN_ID.matcher(runId).matches()) {
      throw new IOException("Invalid Decision candidate runId: " + runId);
    }
    EpsilonDecisionCheckpointBundle checkpoint = requireValidCheckpoint(source);
    Path candidate =
        checkpointRoot
            .toAbsolutePath()
            .normalize()
            .resolve("candidate")
            .resolve(runId)
            .resolve(iterationName(checkpoint.iteration));
    if (Files.exists(candidate)) {
      if (!hasSameCheckpointContents(source, candidate)) {
        throw new IOException("Resumed candidate has another model generation: " + candidate);
      }
    } else {
      copyAtIteration(source, candidate, checkpoint.iteration);
    }
    return candidate;
  }

  /** 候補の保存先とチェックポイント本体から識別子を求める。 */
  static String candidateId(Path checkpoint) throws IOException {
    Path normalized = checkpoint.toAbsolutePath().normalize();
    Path run = normalized.getParent();
    Path candidates = run == null ? null : run.getParent();
    if (candidates == null
        || candidates.getFileName() == null
        || !candidates.getFileName().toString().equals("candidate")
        || !CANDIDATE_RUN_ID.matcher(run.getFileName().toString()).matches()) {
      throw new IOException("Invalid Decision candidate checkpoint path: " + checkpoint);
    }
    int iteration = requireValidCheckpoint(normalized).iteration;
    if (!normalized.getFileName().toString().equals(iterationName(iteration))) {
      throw new IOException("Decision candidate checkpoint iteration mismatch: " + checkpoint);
    }
    return run.getFileName() + "/" + normalized.getFileName();
  }

  private static String iterationName(int iteration) throws IOException {
    if (iteration < 0) {
      throw new IOException("Decision candidate iteration must be non-negative: " + iteration);
    }
    return String.format(Locale.ROOT, "%05d", iteration);
  }

  /**
   * チェックポイント記述情報を検証しマニフェストを返す。
   *
   * @param dir チェックポイントディレクトリ
   * @return 現在バージョン・構造・モデル形式と互換なマニフェスト
   * @throws IOException マニフェストがない、壊れている、または非互換の場合
   */
  public static EpsilonDecisionCheckpointBundle loadManifest(Path dir) throws IOException {
    return loadDescriptor(dir).bundle();
  }

  /** チェックポイント識別情報へ含める現行単一モデル保存物。 */
  static List<Path> checkpointArtifacts(Path dir) throws IOException {
    CheckpointDescriptor descriptor = loadDescriptor(dir);
    requireCheckpointArtifacts(dir, descriptor);
    ArrayList<Path> artifacts = new ArrayList<>();
    artifacts.add(dir.resolve("manifest.json"));
    artifacts.add(dir.resolve(descriptor.bundle().modelFile));
    artifacts.add(dir.resolve("architecture.id"));
    return List.copyOf(artifacts);
  }

  /** 現行チェックポイント本体のマニフェスト・モデル・構造が同一かを比較する。 */
  static boolean hasSameCheckpointContents(Path first, Path second) throws IOException {
    List<Path> firstArtifacts = checkpointArtifacts(first);
    List<Path> secondArtifacts = checkpointArtifacts(second);
    if (firstArtifacts.size() != secondArtifacts.size()) {
      return false;
    }
    for (int index = 0; index < firstArtifacts.size(); index++) {
      Path firstArtifact = firstArtifacts.get(index);
      Path secondArtifact = secondArtifacts.get(index);
      if (!firstArtifact.getFileName().equals(secondArtifact.getFileName())
          || Files.mismatch(firstArtifact, secondArtifact) != -1L) {
        return false;
      }
    }
    return true;
  }

  private static CheckpointDescriptor loadDescriptor(Path dir) throws IOException {
    return loadDescriptor(dir, null);
  }

  private static CheckpointDescriptor loadDescriptor(
      Path dir, EpsilonUtilityProfile expectedProfile) throws IOException {
    if (dir == null) {
      throw new IOException("Decision checkpoint path is null");
    }
    Path manifest = dir.resolve("manifest.json");
    if (!Files.exists(manifest)) {
      throw new IOException("manifest.json not found in " + dir);
    }
    JsonObject json;
    try {
      json = JsonParser.parseString(Files.readString(manifest)).getAsJsonObject();
      requireManifestFields(json, "checkpointVersion");
    } catch (RuntimeException e) {
      throw new IOException("Invalid Decision checkpoint manifest: " + manifest, e);
    }
    int checkpointVersion;
    try {
      checkpointVersion = json.get("checkpointVersion").getAsInt();
    } catch (RuntimeException e) {
      throw new IOException("Invalid Decision checkpoint manifest: " + manifest, e);
    }
    if (checkpointVersion != EpsilonDecisionCheckpointBundle.CURRENT_VERSION) {
      throw new IOException("Unsupported Decision checkpoint version: " + checkpointVersion);
    }
    EpsilonDecisionCheckpointBundle bundle;
    try {
      requireManifestFields(
          json,
          "checkpointVersion",
          "architecture",
          "valueDefinition",
          "hidden",
          "modelFormat",
          "modelFile",
          "globalStep",
          "iteration",
          "selfPlayGames");
      bundle = GSON.fromJson(json, EpsilonDecisionCheckpointBundle.class);
    } catch (RuntimeException e) {
      throw new IOException("Invalid Decision checkpoint manifest: " + manifest, e);
    }
    if (bundle == null) {
      throw new IOException("Invalid Decision checkpoint manifest: " + manifest);
    }
    if (bundle.series != null && !"epsilon".equals(bundle.series)) {
      throw new IOException(
          "Decision checkpoint series mismatch: expected=epsilon actual=" + bundle.series);
    }
    requireCompatibleNetwork(bundle.architecture, bundle.hidden);
    EpsilonUtilityProfile storedProfile = profileFor(bundle.valueDefinition);
    String expectedValueDefinition =
        EpsilonDecisionHlGauss.fingerprint(
            expectedProfile == null ? storedProfile : expectedProfile);
    if (!expectedValueDefinition.equals(bundle.valueDefinition)) {
      throw new IOException(
          "Unsupported Decision checkpoint value definition: expected="
              + expectedValueDefinition
              + " actual="
              + bundle.valueDefinition);
    }
    if (!EpsilonDecisionCheckpointBundle.CURRENT_MODEL_FORMAT.equals(bundle.modelFormat)) {
      throw new IOException("Unsupported Decision checkpoint model format: " + bundle.modelFormat);
    }
    if (!EpsilonDecisionCheckpointBundle.CURRENT_MODEL_FILE.equals(bundle.modelFile)) {
      throw new IOException("Unsupported Decision checkpoint model file: " + bundle.modelFile);
    }
    return new CheckpointDescriptor(bundle, MODEL_PREFIX);
  }

  /** マニフェストに保存された価値定義を、現在のプロセス設定と独立に解決する。 */
  private static EpsilonUtilityProfile profileFor(String valueDefinition) throws IOException {
    for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
      if (profile.rankBased()
          && EpsilonDecisionHlGauss.fingerprint(profile).equals(valueDefinition)) {
        return profile;
      }
    }
    throw new IOException("Unsupported Decision checkpoint value definition: " + valueDefinition);
  }

  private static void requireCompatibleNetwork(String architecture, int hidden) throws IOException {
    if (!EpsilonDecisionConstants.ARCHITECTURE_ID.equals(architecture)) {
      throw new IOException("Unsupported Decision checkpoint architecture: " + architecture);
    }
    if (hidden <= 0 || hidden % 4 != 0) {
      throw new IOException("Unsupported Decision checkpoint hidden size: " + hidden);
    }
  }

  private static void requireManifestFields(JsonObject json, String... fields) {
    for (String field : fields) {
      if (!json.has(field) || json.get(field).isJsonNull()) {
        throw new IllegalArgumentException(
            "Decision checkpoint manifest field is missing: " + field);
      }
    }
  }

  private static void saveContents(
      Model model, Path dir, int globalStep, int iteration, int selfPlayGames) throws Exception {
    model.save(dir, MODEL_PREFIX);
    EpsilonDecisionCheckpointBundle manifest =
        new EpsilonDecisionCheckpointBundle(
            globalStep,
            iteration,
            selfPlayGames,
            ((EpsilonDecisionNetwork) model.getBlock()).hiddenSize(),
            ((EpsilonDecisionNetwork) model.getBlock()).utilityProfile());
    manifest.valueDefinition =
        EpsilonDecisionHlGauss.fingerprint(
            ((EpsilonDecisionNetwork) model.getBlock()).utilityProfile());
    Files.writeString(dir.resolve("manifest.json"), GSON.toJson(manifest));
    Files.writeString(
        dir.resolve("architecture.id"),
        EpsilonDecisionConstants.ARCHITECTURE_ID + System.lineSeparator());
  }

  private static void requireCheckpointArtifacts(Path dir, CheckpointDescriptor descriptor)
      throws IOException {
    if (dir == null) {
      throw new IOException("Decision checkpoint path is null");
    }
    ArtifactScan scan = scanArtifacts(dir);
    boolean hasExactModel = Files.isRegularFile(dir.resolve(descriptor.bundle().modelFile));
    Path architectureFile = dir.resolve("architecture.id");
    if (!scan.hasManifest()
        || !scan.hasArchitecture()
        || !hasExactModel
        || scan.modelFiles() != 1) {
      throw incompleteCheckpoint(dir, scan, hasExactModel);
    }
    String sidecarArchitecture = Files.readString(architectureFile).trim();
    if (!descriptor.bundle().architecture.equals(sidecarArchitecture)) {
      throw new IOException(
          "Decision checkpoint architecture mismatch: manifest="
              + descriptor.bundle().architecture
              + " sidecar="
              + sidecarArchitecture);
    }
  }

  private static Path validateCandidateIfPresent(Path dir) throws IOException {
    ArtifactScan scan = scanArtifacts(dir);
    if (!scan.hasAnyCheckpointArtifact()) {
      return null;
    }
    CheckpointDescriptor descriptor = loadDescriptor(dir);
    requireCheckpointArtifacts(dir, descriptor);
    return dir;
  }

  private static IOException incompleteCheckpoint(
      Path dir, ArtifactScan scan, boolean hasExactModel) {
    return new IOException(
        "Incomplete Decision checkpoint: "
            + dir
            + " manifest="
            + scan.hasManifest()
            + " architecture="
            + scan.hasArchitecture()
            + " exactModel="
            + hasExactModel
            + " modelFiles="
            + scan.modelFiles());
  }

  private static ArtifactScan scanArtifacts(Path dir) throws IOException {
    if (dir == null || !Files.isDirectory(dir)) {
      return ArtifactScan.empty();
    }
    boolean hasManifest = Files.isRegularFile(dir.resolve("manifest.json"));
    boolean hasArchitecture = Files.isRegularFile(dir.resolve("architecture.id"));
    int modelFiles = 0;
    try (Stream<Path> stream = Files.list(dir)) {
      for (Path path : stream.filter(Files::isRegularFile).toList()) {
        if (isModelParam(path, MODEL_PREFIX)) {
          modelFiles++;
        }
      }
    }
    return new ArtifactScan(hasManifest, hasArchitecture, modelFiles);
  }

  private static boolean isModelParam(Path path, String prefix) {
    String name = path.getFileName().toString();
    return name.startsWith(prefix + "-") && name.endsWith(".params");
  }

  private record CheckpointDescriptor(EpsilonDecisionCheckpointBundle bundle, String modelPrefix) {}

  private record ArtifactScan(boolean hasManifest, boolean hasArchitecture, int modelFiles) {
    private static ArtifactScan empty() {
      return new ArtifactScan(false, false, 0);
    }

    private boolean hasAnyCheckpointArtifact() {
      return hasManifest || hasArchitecture || modelFiles > 0;
    }
  }

  private static void replaceDirectory(Path source, Path target) throws IOException {
    Path backup = null;
    if (Files.exists(target)) {
      backup = uniqueSibling(target, ".old");
      moveDirectory(target, backup);
    }
    try {
      moveDirectory(source, target);
    } catch (IOException e) {
      if (backup != null && Files.exists(backup) && !Files.exists(target)) {
        try {
          moveDirectory(backup, target);
        } catch (IOException restore) {
          e.addSuppressed(restore);
        }
      }
      throw e;
    }
    if (backup != null) {
      deleteDirectoryQuietly(backup);
    }
  }

  private static void moveDirectory(Path source, Path target) throws IOException {
    for (int attempt = 0; attempt < 5; attempt++) {
      try {
        try {
          Files.move(source, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
          Files.move(source, target);
        }
        return;
      } catch (IOException e) {
        if (attempt == 4) {
          throw e;
        }
        try {
          Thread.sleep(50L * (attempt + 1L));
        } catch (InterruptedException interrupted) {
          Thread.currentThread().interrupt();
          IOException interruptedMove =
              new IOException("Interrupted while retrying Decision checkpoint move", interrupted);
          interruptedMove.addSuppressed(e);
          throw interruptedMove;
        }
      }
    }
  }

  private static Path uniqueSibling(Path target, String tag) {
    Path parent = target.getParent();
    String name = "." + target.getFileName() + tag + "-" + System.nanoTime();
    return parent == null ? Path.of(name) : parent.resolve(name);
  }

  private static void deleteDirectoryQuietly(Path dir) {
    try {
      deleteDirectoryIfExists(dir);
    } catch (IOException e) {
      log.warn("Failed to delete temporary Decision checkpoint directory: {}", dir, e);
    }
  }

  private static void deleteDirectoryIfExists(Path dir) throws IOException {
    if (dir == null || !Files.exists(dir)) {
      return;
    }
    try (Stream<Path> stream = Files.walk(dir)) {
      for (Path path : stream.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }

  private static IOException asIOException(String message, Exception e) {
    return e instanceof IOException io ? io : new IOException(message, e);
  }
}

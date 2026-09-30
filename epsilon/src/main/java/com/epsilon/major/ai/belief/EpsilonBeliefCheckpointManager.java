package com.epsilon.major.ai.belief;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.network.NetworkFactory;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 単体 belief チェックポイントの保存と読み込み。 */
public final class EpsilonBeliefCheckpointManager {

  private static final Logger log = LoggerFactory.getLogger(EpsilonBeliefCheckpointManager.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final String MODEL_PREFIX = "belief";

  private EpsilonBeliefCheckpointManager() {}

  /**
   * モデルパラメーターとマニフェストを一時ディレクトリへ書き、完成後にチェックポイントを置換する。
   *
   * @param model 保存するBelief モデル
   * @param dir チェックポイントディレクトリ
   * @param globalStep オプティマイザー更新回数
   * @param iteration 外側学習反復回数
   * @throws IOException 保存またはディレクトリ置換に失敗した場合
   */
  public static void save(Model model, Path dir, int globalStep, int iteration) throws IOException {
    Path target = dir.toAbsolutePath().normalize();
    Path parent = target.getParent();
    Files.createDirectories(parent);
    Path tmp = Files.createTempDirectory(parent, "." + target.getFileName() + ".tmp-");
    boolean installed = false;
    try {
      saveContents(model, tmp, globalStep, iteration);
      replaceDirectory(tmp, target);
      installed = true;
      log.info("Belief checkpoint saved: dir={} step={} iteration={}", dir, globalStep, iteration);
    } finally {
      if (!installed) {
        deleteDirectoryQuietly(tmp);
      }
    }
  }

  /**
   * マニフェスト互換性を検証してBelief モデルを読み込む。
   *
   * @param dir チェックポイントディレクトリ
   * @return パラメーターを所有するDJL モデル
   * @throws IOException マニフェストまたはパラメーターを読めない場合
   */
  public static Model load(Path dir) throws IOException {
    return load(dir, NetworkFactory.getPreferredDevice());
  }

  public static Model load(Path dir, Device device) throws IOException {
    var manifest = loadManifest(dir);
    int hidden = manifest.hidden;
    if (hidden == 0) {
      int epoch = ai.djl.util.Utils.getCurrentEpoch(dir, MODEL_PREFIX);
      if (epoch < 0) throw new IOException("Belief parameter file not found: " + dir);
      Path parameters =
          dir.resolve(String.format(java.util.Locale.ROOT, "belief-%04d.params", epoch));
      hidden = com.epsilon.io.LegacyBeliefParameters.readHidden(parameters);
    }
    Model model = NetworkFactory.createBeliefModel(device, hidden);
    try {
      model.load(dir, MODEL_PREFIX);
      return model;
    } catch (Exception e) {
      model.close();
      throw new IOException("Failed to load Belief model from " + dir, e);
    }
  }

  /**
   * ディレクトリにマニフェストとモデルパラメーターが揃っているかを返す。
   *
   * @param dir 検査するディレクトリ
   * @return チェックポイントの必須ファイルが存在すればtrue
   * @throws IOException ディレクトリ列挙に失敗した場合
   */
  public static boolean isValidCheckpoint(Path dir) throws IOException {
    return Files.exists(dir.resolve("manifest.json")) && hasModelFile(dir, MODEL_PREFIX);
  }

  /**
   * チェックポイントの親ディレクトリ配下の標準latest パスを返す。
   *
   * @param checkpointDir チェックポイントの親ディレクトリ
   * @return {@code checkpointDir/latest}
   */
  public static Path latest(Path checkpointDir) {
    return checkpointDir.resolve("latest");
  }

  /**
   * latest、次に親ディレクトリ自身の順で既存チェックポイントを解決する。
   *
   * @param checkpointDir チェックポイントの親ディレクトリまたは直接チェックポイントパス
   * @return 使用可能なパス。見つからなければ {@code null}
   * @throws IOException ディレクトリ検査に失敗した場合
   */
  public static Path resolveExisting(Path checkpointDir) throws IOException {
    Path latest = latest(checkpointDir);
    if (isValidCheckpoint(latest)) {
      return latest;
    }
    if (isValidCheckpoint(checkpointDir)) {
      return checkpointDir;
    }
    return null;
  }

  /**
   * マニフェストを読み、バージョンと構造を現在実装へ照合する。
   *
   * @param dir チェックポイントディレクトリ
   * @return 検証済みマニフェスト
   * @throws IOException マニフェストがない、壊れている、または非互換の場合
   */
  public static EpsilonBeliefCheckpointBundle loadManifest(Path dir) throws IOException {
    Path manifest = dir.resolve("manifest.json");
    if (!Files.exists(manifest)) {
      throw new IOException("manifest.json not found in " + dir);
    }
    EpsilonBeliefCheckpointBundle bundle =
        GSON.fromJson(Files.readString(manifest), EpsilonBeliefCheckpointBundle.class);
    if (bundle.checkpointVersion != EpsilonBeliefCheckpointBundle.CURRENT_VERSION) {
      throw new IOException("Unsupported Belief checkpoint version: " + bundle.checkpointVersion);
    }
    if (!EpsilonBeliefNetwork.ARCHITECTURE_ID.equals(bundle.architecture)) {
      throw new IOException("Unsupported Belief checkpoint architecture: " + bundle.architecture);
    }
    if (bundle.hidden < 0 || (bundle.series != null && !bundle.series.equals("epsilon")))
      throw new IOException(
          "Unsupported Belief series or hidden: " + bundle.series + "/" + bundle.hidden);
    if (bundle.inputFingerprint != null
        && !DecisionInputSchema.stateFingerprint().equals(bundle.inputFingerprint))
      throw new IOException("Unsupported Belief input: " + bundle.inputFingerprint);
    return bundle;
  }

  private static void saveContents(Model model, Path dir, int globalStep, int iteration)
      throws IOException {
    model.save(dir, MODEL_PREFIX);
    var manifest = new EpsilonBeliefCheckpointBundle(globalStep, iteration);
    manifest.series = "epsilon";
    manifest.inputFingerprint = DecisionInputSchema.stateFingerprint();
    manifest.hidden = ((EpsilonBeliefNetwork) model.getBlock()).hiddenSize();
    Files.writeString(dir.resolve("manifest.json"), GSON.toJson(manifest));
    Files.writeString(
        dir.resolve("architecture.id"),
        EpsilonBeliefNetwork.ARCHITECTURE_ID + System.lineSeparator());
  }

  private static boolean hasModelFile(Path dir, String prefix) throws IOException {
    try (var stream = Files.list(dir)) {
      return stream.anyMatch(path -> isModelParam(path, prefix));
    }
  }

  private static boolean isModelParam(Path path, String prefix) {
    String name = path.getFileName().toString();
    return name.startsWith(prefix + "-") && name.endsWith(".params");
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
    try {
      Files.move(source, target, StandardCopyOption.ATOMIC_MOVE);
    } catch (AtomicMoveNotSupportedException e) {
      Files.move(source, target);
    }
  }

  private static Path uniqueSibling(Path target, String tag) {
    String name = "." + target.getFileName() + tag + "-" + System.nanoTime();
    return target.resolveSibling(name);
  }

  private static void deleteDirectoryQuietly(Path dir) {
    try {
      deleteDirectoryIfExists(dir);
    } catch (IOException e) {
      log.warn("Failed to delete temporary Belief checkpoint directory: {}", dir, e);
    }
  }

  private static void deleteDirectoryIfExists(Path dir) throws IOException {
    if (!Files.exists(dir)) {
      return;
    }
    try (Stream<Path> stream = Files.walk(dir)) {
      for (Path path : stream.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}

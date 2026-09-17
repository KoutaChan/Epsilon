package com.epsilon.pico.ai.grp;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import com.epsilon.ai.grp.EpsilonGrpCheckpointBundle;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import com.epsilon.ai.grp.EpsilonGrpTrainer;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 単体 Epsilon GRP チェックポイントの保存と読み込み。 */
public final class EpsilonGrpCheckpointManager {

  private static final Logger log = LoggerFactory.getLogger(EpsilonGrpCheckpointManager.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final String MODEL_PREFIX = "grp";
  private static final String OPTIMIZER_STATE = "optimizer.state";

  private EpsilonGrpCheckpointManager() {}

  /**
   * モデル、オプティマイザー、マニフェストを一時ディレクトリへ書き、対象チェックポイントと原子的に置換する。
   *
   * @param model 保存する GRP モデル
   * @param trainer オプティマイザー状態の保存元
   * @param dir チェックポイントディレクトリ
   * @param globalStep オプティマイザー更新の累計更新回数
   * @param iteration GRP 世代番号
   * @throws IOException 保存物の書き込みまたはディレクトリ置換に失敗した場合
   */
  public static void save(
      Model model, EpsilonGrpTrainer trainer, Path dir, int globalStep, int iteration)
      throws IOException {
    Path target = dir.toAbsolutePath().normalize();
    Path parent = target.getParent();
    Files.createDirectories(parent);
    Path tmp = Files.createTempDirectory(parent, "." + target.getFileName() + ".tmp-");
    boolean installed = false;
    try {
      model.save(tmp, MODEL_PREFIX);
      trainer.saveOptimizerState(tmp.resolve(OPTIMIZER_STATE));
      EpsilonGrpCheckpointBundle manifest =
          new EpsilonGrpCheckpointBundle(
              globalStep,
              iteration,
              ((EpsilonGrpNetwork) model.getBlock()).hiddenSize(),
              ((EpsilonGrpNetwork) model.getBlock()).layers());
      Files.writeString(tmp.resolve("manifest.json"), GSON.toJson(manifest));
      Files.writeString(
          tmp.resolve("architecture.id"), manifest.architecture + System.lineSeparator());
      replaceDirectory(tmp, target);
      installed = true;
      log.info("GRP checkpoint saved: dir={} step={} iteration={}", dir, globalStep, iteration);
    } finally {
      if (!installed) {
        deleteDirectoryQuietly(tmp);
      }
    }
  }

  /**
   * stable 世代を、同一 JVM・別 JVM の並行書き込み処理からも上書きされない形で一度だけ保存する。
   *
   * @param model 保存する GRP モデル
   * @param trainer オプティマイザー状態の保存元
   * @param dir 新規作成する変更不可チェックポイントディレクトリ
   * @param globalStep オプティマイザー更新の累計更新回数
   * @param iteration GRP 世代番号
   * @throws IOException 対象が既に存在するか、lock・保存処理に失敗した場合
   */
  public static synchronized void saveImmutable(
      Model model, EpsilonGrpTrainer trainer, Path dir, int globalStep, int iteration)
      throws IOException {
    Path target = dir.toAbsolutePath().normalize();
    Path parent = target.getParent();
    Files.createDirectories(parent);
    Path lockPath = parent.resolve("." + target.getFileName() + ".immutable.lock");
    try (FileChannel channel =
            FileChannel.open(lockPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
        FileLock ignored = channel.lock()) {
      if (Files.exists(target)) {
        throw new IOException("Refusing to overwrite immutable GRP checkpoint: " + target);
      }
      save(model, trainer, target, globalStep, iteration);
    }
  }

  /**
   * チェックポイントを新しい GRP モデルへ読み込む。
   *
   * @param dir チェックポイントディレクトリ
   * @param device モデルを配置するデバイス
   * @return パラメーターを復元した GRP モデル。呼び出し側が解放する
   * @throws IOException マニフェストまたはモデルパラメーターが不正な場合
   */
  public static Model load(Path dir, Device device) throws IOException {
    var config = loadManifest(dir).configuration();
    Model model = NetworkFactory.createGrpModel(device, config.hidden(), config.layers());
    try {
      loadInto(model, dir);
      return model;
    } catch (IOException e) {
      model.close();
      throw e;
    }
  }

  /**
   * 既存モデルのインスタンスのパラメーターを検証済みチェックポイントへ戻す。
   *
   * @param model 復元先の GRP モデル
   * @param dir チェックポイントディレクトリ
   * @throws IOException マニフェストまたはモデルパラメーターが不正な場合
   */
  public static void loadInto(Model model, Path dir) throws IOException {
    loadManifest(dir);
    try {
      model.load(dir, MODEL_PREFIX);
    } catch (MalformedModelException e) {
      throw new IOException("Malformed GRP model parameters in " + dir, e);
    }
  }

  /**
   * チェックポイントのオプティマイザー状態を学習処理へ復元する。
   *
   * @param trainer 復元先の学習処理
   * @param dir チェックポイントディレクトリ
   * @throws IOException オプティマイザー状態が存在しないか読み込めない場合
   */
  public static void loadOptimizerState(EpsilonGrpTrainer trainer, Path dir) throws IOException {
    Path state = dir.resolve(OPTIMIZER_STATE);
    if (!Files.isRegularFile(state)) {
      throw new IOException("GRP optimizer state not found in " + dir);
    }
    trainer.loadOptimizerState(state);
  }

  /**
   * チェックポイントマニフェストと必須保存物の整合性を検証する。
   *
   * @param dir チェックポイントディレクトリ
   * @return 検証済みマニフェスト
   * @throws IOException スキーマ、構造、counter または保存物が不正な場合
   */
  public static EpsilonGrpCheckpointBundle loadManifest(Path dir) throws IOException {
    Path manifest = dir.resolve("manifest.json");
    if (!Files.exists(manifest)) {
      throw new IOException("manifest.json not found in " + dir);
    }
    EpsilonGrpCheckpointBundle bundle =
        GSON.fromJson(Files.readString(manifest), EpsilonGrpCheckpointBundle.class);
    if (bundle.checkpointVersion != 2) {
      throw new IOException("Unsupported GRP checkpoint version: " + bundle.checkpointVersion);
    }
    bundle.configuration();
    if (!hasModelFile(dir, MODEL_PREFIX)) {
      throw new IOException("GRP model parameters not found in " + dir);
    }
    if (!Files.isRegularFile(dir.resolve(OPTIMIZER_STATE))) {
      throw new IOException("GRP optimizer state not found in " + dir);
    }
    if (bundle.globalStep < 0 || bundle.iteration < 0) {
      throw new IOException(
          "GRP checkpoint counters must be non-negative: globalStep="
              + bundle.globalStep
              + " iteration="
              + bundle.iteration);
    }
    return bundle;
  }

  /**
   * ディレクトリが現行 GRP チェックポイント契約を満たすかを調べる。
   *
   * @param dir 検査するディレクトリ
   * @return マニフェストと必須保存物が有効なら {@code true}
   */
  public static boolean isValidCheckpoint(Path dir) {
    if (dir == null) {
      return false;
    }
    try {
      loadManifest(dir);
      return true;
    } catch (IOException e) {
      return false;
    }
  }

  /**
   * GRP 推論教師モデルの内容識別情報。
   *
   * <p>オプティマイザー状態は推論結果へ影響しないため含めない。保存情報、モデル構成の識別子、モデルパラメーターファイルの相対名と内容を一定の順序でハッシュ化し、同じ学習反復番号でも重みが異なれば区別する。
   *
   * @param dir 識別情報を計算するチェックポイントディレクトリ
   * @return 推論保存物全体の SHA-256
   * @throws IOException マニフェストの検証または保存物の読み込みに失敗した場合
   */
  public static String checkpointSha256(Path dir) throws IOException {
    Path checkpoint = dir.toAbsolutePath().normalize();
    loadManifest(checkpoint);
    List<Path> modelFiles;
    try (Stream<Path> stream = Files.list(checkpoint)) {
      modelFiles =
          stream
              .filter(Files::isRegularFile)
              .filter(
                  path -> {
                    String name = path.getFileName().toString();
                    return name.startsWith(MODEL_PREFIX + "-") && name.endsWith(".params");
                  })
              .sorted(Comparator.comparing(path -> path.getFileName().toString()))
              .toList();
    }
    if (modelFiles.isEmpty()) {
      throw new IOException("GRP model parameters not found in " + checkpoint);
    }
    ArrayList<Path> artifacts = new ArrayList<>(modelFiles.size() + 2);
    artifacts.add(checkpoint.resolve("manifest.json"));
    artifacts.add(checkpoint.resolve("architecture.id"));
    artifacts.addAll(modelFiles);
    for (Path artifact : artifacts) {
      if (!Files.isRegularFile(artifact)) {
        throw new IOException("GRP checkpoint identity artifact not found: " + artifact);
      }
    }
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IOException("SHA-256 is unavailable", e);
    }
    byte[] buffer = new byte[1 << 16];
    for (Path artifact : artifacts) {
      String relative = checkpoint.relativize(artifact).toString().replace('\\', '/');
      digest.update(relative.getBytes(StandardCharsets.UTF_8));
      digest.update((byte) 0);
      try (InputStream in = Files.newInputStream(artifact)) {
        int read;
        while ((read = in.read(buffer)) != -1) {
          digest.update(buffer, 0, read);
        }
      }
      digest.update((byte) 0xff);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  /**
   * チェックポイントの親ディレクトリ内の可変 {@code latest} パスを返す。
   *
   * @param checkpointDir チェックポイントの親ディレクトリ
   * @return {@code latest} 配下の保存先パス
   */
  public static Path latest(Path checkpointDir) {
    return checkpointDir.resolve("latest");
  }

  /**
   * 読み込み対象となる既存世代を変更不可配下の保存先、latest、親ディレクトリの順に解決する。
   *
   * @param checkpointDir チェックポイントの親ディレクトリまたは単体チェックポイントディレクトリ
   * @return 保存物を持つチェックポイントパス。見つからなければ {@code null}
   */
  public static Path resolveExisting(Path checkpointDir) {
    // saveAt が先に作る世代ディレクトリは以後上書きしない。並行書き込み処理が latest を
    // 差し替えていてもモデル / マニフェスト / オプティマイザーを必ず同じ世代から読めるよう、
    // 親ディレクトリからの解決では変更不可配下の保存先を優先する。
    Path stable = newestStableChildWithArtifacts(checkpointDir);
    if (stable != null) {
      return stable;
    }
    Path latest = latest(checkpointDir);
    // 保存物が一部でもあればそのパスを返し、load側で詳細を不整合を検出して直ちに例外を送出する。
    // 壊れた/旧チェックポイントを「未作成」と誤認して上書きしない。
    if (hasCheckpointArtifacts(latest)) {
      return latest;
    }
    if (hasCheckpointArtifacts(checkpointDir)) {
      return checkpointDir;
    }
    return null;
  }

  private static boolean hasCheckpointArtifacts(Path dir) {
    return dir != null
        && (Files.exists(dir.resolve("manifest.json"))
            || Files.exists(dir.resolve(OPTIMIZER_STATE))
            || hasModelFile(dir, MODEL_PREFIX));
  }

  private static Path newestStableChildWithArtifacts(Path checkpointDir) {
    if (!Files.isDirectory(checkpointDir)) {
      return null;
    }
    try (Stream<Path> stream = Files.list(checkpointDir)) {
      return stream
          .filter(Files::isDirectory)
          .filter(path -> !path.getFileName().toString().startsWith("."))
          .filter(path -> !path.getFileName().toString().equals("latest"))
          .filter(EpsilonGrpCheckpointManager::hasCheckpointArtifacts)
          .map(EpsilonGrpCheckpointManager::stableCandidate)
          .filter(Objects::nonNull)
          .max(
              Comparator.comparingInt(StableCandidate::iteration)
                  .thenComparingInt(StableCandidate::globalStep)
                  .thenComparingLong(StableCandidate::modifiedMillis))
          .map(StableCandidate::path)
          .orElse(null);
    } catch (IOException e) {
      return null;
    }
  }

  private static StableCandidate stableCandidate(Path path) {
    try {
      EpsilonGrpCheckpointBundle manifest = loadManifest(path);
      return new StableCandidate(
          path,
          manifest.iteration,
          manifest.globalStep,
          Files.getLastModifiedTime(path).toMillis());
    } catch (IOException e) {
      return null;
    }
  }

  private record StableCandidate(Path path, int iteration, int globalStep, long modifiedMillis) {}

  private static boolean hasModelFile(Path dir, String prefix) {
    try (var stream = Files.list(dir)) {
      return stream.anyMatch(
          path -> {
            String name = path.getFileName().toString();
            return name.startsWith(prefix + "-") && name.endsWith(".params");
          });
    } catch (IOException e) {
      return false;
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
      if (!Files.exists(dir)) {
        return;
      }
      try (Stream<Path> stream = Files.walk(dir)) {
        for (Path path : stream.sorted(Comparator.reverseOrder()).toList()) {
          Files.deleteIfExists(path);
        }
      }
    } catch (IOException e) {
      log.warn("Failed to delete temporary GRP checkpoint directory: {}", dir, e);
    }
  }
}

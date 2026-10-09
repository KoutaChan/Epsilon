package com.epsilon.cfr.training;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.io.CfrFiles;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.io.AtomicFiles;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.TreeMap;
import java.util.UUID;

/** 実行ディレクトリへの書き込みを排他制御し、保存済みの世代を指す公開ポインタを原子的に更新する。 */
public final class CfrRunStore implements AutoCloseable {
  private final Path root;
  private final FileChannel channel;
  private final FileLock lock;

  public CfrRunStore(Path root) throws IOException {
    this.root = root.toAbsolutePath().normalize();
    Files.createDirectories(this.root);
    channel =
        FileChannel.open(
            this.root.resolve("run.lock"), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    FileLock acquired;
    try {
      acquired = channel.tryLock();
    } catch (RuntimeException | IOException failure) {
      channel.close();
      throw failure;
    }
    if (acquired == null) {
      channel.close();
      throw new IOException("LCFR run already has a writer: " + root);
    }
    lock = acquired;
  }

  public Path root() {
    return root;
  }

  public Path resolve(String relative) throws IOException {
    return CfrFiles.inside(root, relative);
  }

  public String relative(Path path) throws IOException {
    Path exact = path.toAbsolutePath().normalize();
    if (!exact.startsWith(root) || exact.equals(root))
      throw new IOException("Artifact outside run: " + path);
    return root.relativize(exact).toString().replace('\\', '/');
  }

  public boolean exists() {
    return Files.isRegularFile(root.resolve("current.json"));
  }

  public Path newArtifact(String kind, long iteration) throws IOException {
    if (!kind.matches("[a-z-]+")) throw new IllegalArgumentException("Invalid artifact kind");
    Path path =
        root.resolve("artifacts")
            .resolve(kind + "-" + String.format("%08d", iteration) + "-" + UUID.randomUUID());
    Files.createDirectories(path);
    return path;
  }

  public void publish(CfrRunState state) throws IOException {
    checkStateReferences(root, state);
    byte[] bytes = CfrConfig.JSON.toJson(state).getBytes(StandardCharsets.UTF_8);
    String digest = CfrFiles.sha256(bytes);
    Path immutable = root.resolve("states").resolve(digest + ".json");
    if (!Files.exists(immutable)) AtomicFiles.write(immutable, bytes);
    AtomicFiles.write(
        root.resolve("current.json"),
        CfrConfig.JSON
            .toJson(new Pointer("states/" + digest + ".json", digest))
            .getBytes(StandardCharsets.UTF_8));
  }

  public CfrRunState read() throws IOException {
    return inspect(root);
  }

  /** 再開時と明示的な監査時に、サイズの大きいモデルとオプティマイザーの保存内容を検証する。 */
  public CfrRunState verify() throws IOException {
    CfrRunState state = read();
    var checkpoints = new LinkedHashSet<String>();
    if (state.boundaryRoot() != null) checkpoints.add(state.boundaryRoot());
    checkpoints.add(state.regretCheckpoint());
    checkpoints.add(state.averageCheckpoint());
    if (state.frozenCheckpoint() != null) checkpoints.add(state.frozenCheckpoint());
    for (String checkpoint : checkpoints) verifyArtifact(resolve(checkpoint));
    return state;
  }

  /** 学習側の書き込みロックを取得せずに、変更不可の保存物を指す公開ポインタを読み込む。 */
  public static CfrRunState inspect(Path directory) throws IOException {
    Path root = directory.toAbsolutePath().normalize();
    try {
      Pointer pointer =
          CfrConfig.JSON.fromJson(Files.readString(root.resolve("current.json")), Pointer.class);
      Path stateFile = CfrFiles.inside(root, pointer.path());
      byte[] bytes = Files.readAllBytes(stateFile);
      if (!CfrFiles.sha256(bytes).equals(pointer.sha256()))
        throw new IOException("Run state checksum mismatch");
      CfrRunState state =
          CfrConfig.JSON.fromJson(new String(bytes, StandardCharsets.UTF_8), CfrRunState.class);
      checkStateReferences(root, state);
      return state;
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid LCFR run state", invalid);
    }
  }

  private static void checkStateReferences(Path root, CfrRunState state) throws IOException {
    if (state == null) throw new IOException("Missing LCFR state");
    if (!CfrFiles.sha256(root.resolve("config.json")).equals(state.configHash()))
      throw new IOException("Run configuration changed after initialization");
    if (state.boundaryRoot() != null) checkArtifact(CfrFiles.inside(root, state.boundaryRoot()));
    checkArtifact(CfrFiles.inside(root, state.regretCheckpoint()));
    checkArtifact(CfrFiles.inside(root, state.averageCheckpoint()));
    if (state.frozenCheckpoint() != null)
      checkArtifact(CfrFiles.inside(root, state.frozenCheckpoint()));
  }

  /** モデルとオプティマイザーの全ファイルについて、検証用の情報を保存する。保存後に更新するファイルは含めない。 */
  public static void sealArtifact(Path directory) throws IOException {
    Map<String, String> hashes = new TreeMap<>();
    try (var files = Files.walk(directory)) {
      for (Path file : files.filter(Files::isRegularFile).sorted().toList()) {
        String name = directory.relativize(file).toString().replace('\\', '/');
        if (!name.equals("artifact.json")) hashes.put(name, CfrFiles.sha256(file));
      }
    }
    if (hashes.isEmpty()) throw new IOException("Cannot seal empty artifact " + directory);
    AtomicFiles.write(
        directory.resolve("artifact.json"),
        CfrConfig.JSON.toJson(new Artifact(1, hashes)).getBytes(StandardCharsets.UTF_8));
  }

  public static void verifyArtifact(Path directory) throws IOException {
    for (var item : readArtifact(directory).hashes().entrySet())
      if (!CfrFiles.sha256(CfrFiles.inside(directory, item.getKey())).equals(item.getValue()))
        throw new IOException("Artifact checksum mismatch: " + item.getKey());
  }

  private static void checkArtifact(Path directory) throws IOException {
    for (String name : readArtifact(directory).hashes().keySet())
      if (!Files.isRegularFile(CfrFiles.inside(directory, name)))
        throw new IOException("Missing artifact file: " + name);
  }

  private static Artifact readArtifact(Path directory) throws IOException {
    try {
      Artifact artifact =
          CfrConfig.JSON.fromJson(
              Files.readString(directory.resolve("artifact.json")), Artifact.class);
      if (artifact == null || artifact.version() != 1 || artifact.hashes().isEmpty())
        throw new IOException("Invalid artifact manifest");
      return artifact;
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid artifact at " + directory, invalid);
    }
  }

  String saveTraining(CfrModel model, String kind, long iteration, String solveId)
      throws IOException {
    Path path = newArtifact(kind, iteration);
    model.saveTraining(path);
    AtomicFiles.write(
        path.resolve("provenance.json"),
        CfrConfig.JSON
            .toJson(Map.of("sourceSolveId", solveId, "sourceIteration", iteration))
            .getBytes(StandardCharsets.UTF_8));
    CfrRunStore.sealArtifact(path);
    return relative(path);
  }

  public record Champion(String checkpoint, String solveId, long iteration) {}

  public record CollectedPolicy(long iteration, String checkpoint) {}

  public Champion champion() throws IOException {
    return CfrConfig.JSON.fromJson(
        Files.readString(root().resolve("champion.json")), Champion.class);
  }

  void publishChampion(String checkpoint, String solveId, long iteration) throws IOException {
    AtomicFiles.write(
        root().resolve("champion.json"),
        CfrConfig.JSON
            .toJson(new Champion(checkpoint, solveId, iteration))
            .getBytes(StandardCharsets.UTF_8));
  }

  static void copyGrp(Path source, Path output) throws IOException {
    Path exact =
        EpsilonGrpCheckpointManager.isValidCheckpoint(source)
            ? source
            : EpsilonGrpCheckpointManager.resolveExisting(source);
    if (exact == null) exact = EpsilonGrpCheckpointManager.resolveExisting(source.resolve("grp"));
    if (exact == null) throw new IOException("No GRP checkpoint at " + source);
    Path destination = output.resolve("grp/latest");
    Files.createDirectories(destination);
    try (var files = Files.walk(exact)) {
      for (Path file : files.filter(Files::isRegularFile).toList()) {
        Path target = destination.resolve(exact.relativize(file));
        Files.createDirectories(target.getParent());
        Files.copy(file, target, StandardCopyOption.REPLACE_EXISTING);
      }
    }
  }

  /** 収集に実際に使った凍結方策を、平均化の検証に使える反復履歴として保存する。 */
  void recordCollectedPolicy(CfrRunState state) throws IOException {
    AtomicFiles.write(
        resolve("solves/" + state.solveId() + "/snapshots/" + state.iteration() + ".json"),
        CfrConfig.JSON
            .toJson(new CollectedPolicy(state.iteration(), state.frozenCheckpoint()))
            .getBytes(StandardCharsets.UTF_8));
  }

  public void report(String filename, Object report) throws IOException {
    AtomicFiles.write(
        resolve("reports/" + filename),
        CfrConfig.JSON.toJson(report).getBytes(StandardCharsets.UTF_8));
  }

  private record Pointer(String path, String sha256) {}

  private record Artifact(int version, Map<String, String> hashes) {}

  @Override
  public void close() throws IOException {
    try {
      lock.close();
    } finally {
      channel.close();
    }
  }
}

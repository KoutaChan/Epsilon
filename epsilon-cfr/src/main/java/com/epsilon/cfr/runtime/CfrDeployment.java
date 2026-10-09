package com.epsilon.cfr.runtime;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.io.CfrFiles;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.training.CfrRunState;
import com.epsilon.cfr.training.CfrRunStore;
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 検証済みの平均方策と学習元の情報をまとめた配布用モデル。学習用モデルは配布対象に含めない。 */
public final class CfrDeployment implements AutoCloseable {
  public record Manifest(
      int version,
      String series,
      String role,
      String gameMode,
      String modelDirectory,
      String sourceSolveId,
      long sourceIteration,
      String sampling,
      EpsilonUtilityProfile utilityProfile) {}

  private record Provenance(String sourceSolveId, long sourceIteration) {}

  final CfrModel model;
  final int maxBatch;
  final long seed;
  private final Manifest manifest;

  private CfrDeployment(CfrModel model, int maxBatch, long seed, Manifest manifest) {
    this.model = model;
    this.maxBatch = maxBatch;
    this.seed = seed;
    this.manifest = manifest;
  }

  public Manifest manifest() {
    return manifest;
  }

  public static CfrDeployment open(Path directory, Map<String, String> options) {
    try {
      Map<String, String> values = new LinkedHashMap<>(options);
      String deviceName = values.remove("device");
      String maxBatchText = values.remove("maxBatch");
      String seedText = values.remove("seed");
      if (!values.isEmpty())
        throw new IllegalArgumentException("Unknown CFR deployment options: " + values.keySet());
      Device device = parseDevice(deviceName == null ? "cpu" : deviceName);
      int maxBatch = maxBatchText == null ? 128 : Integer.parseInt(maxBatchText);
      long seed = seedText == null ? 1 : Long.parseLong(seedText);
      if (maxBatch < 1) throw new IllegalArgumentException("maxBatch must be positive");
      CfrRunStore.verifyArtifact(directory);
      Manifest manifest = readManifest(directory);
      CfrModel model =
          CfrModel.loadModel(CfrFiles.inside(directory, manifest.modelDirectory()), device);
      if (model.role() != CfrModel.Role.AVERAGE) {
        model.close();
        throw new IOException("Only an average-policy model can be deployed");
      }
      if (model.config().utilityProfile() != manifest.utilityProfile()) {
        model.close();
        throw new IOException("CFR deployment/model utility profile mismatch");
      }
      return new CfrDeployment(model, maxBatch, seed, manifest);
    } catch (IOException failure) {
      throw new UncheckedIOException("Cannot open CFR deployment " + directory, failure);
    }
  }

  private static Manifest readManifest(Path directory) throws IOException {
    try {
      Manifest manifest =
          CfrConfig.JSON.fromJson(
              Files.readString(directory.resolve("deployment.json")), Manifest.class);
      if (manifest == null
          || manifest.version() != 2
          || !"epsilon-cfr".equals(manifest.series())
          || !"AVERAGE_DEPLOYMENT".equals(manifest.role())
          || !"policy".equals(manifest.sampling())
          || !Set.of("HAND_GRP", "MATCH_MC").contains(manifest.gameMode())
          || !"model".equals(manifest.modelDirectory())
          || manifest.sourceSolveId() == null
          || manifest.sourceSolveId().isBlank()
          || manifest.sourceIteration() < 0
          || manifest.utilityProfile() == null
          || !manifest.utilityProfile().rankBased())
        throw new IOException("Invalid CFR deployment manifest");
      if (Files.exists(directory.resolve("model/optimizer.bin")))
        throw new IOException("Deployment contains training optimizer state");
      return manifest;
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid CFR deployment manifest", invalid);
    }
  }

  DecisionContext context(
      long episodeId, long decisionId, PublicObservation observation, List<Action> legal) {
    return new DecisionContext(episodeId, decisionId, observation.observer(), observation, legal);
  }

  /** 確定済みの平均方策チェックポイントを配布形式へ書き出す。書き出し中も既存の学習は継続できる。 */
  public static Manifest export(Path run, Path output, String selection) throws IOException {
    if (!Set.of("average", "champion").contains(selection))
      throw new IllegalArgumentException("selection must be average or champion");
    Path destination = output.toAbsolutePath().normalize();
    if (Files.exists(destination)) throw new FileAlreadyExistsException(destination.toString());
    Path runRoot = run.toAbsolutePath().normalize();
    CfrRunState state = CfrRunStore.inspect(runRoot);
    CfrConfig config = CfrConfig.read(runRoot.resolve("config.json"));
    String checkpoint = state.averageCheckpoint(), solveId = state.solveId();
    long iteration = state.iteration();
    if (selection.equals("champion")) {
      CfrRunStore.Champion champion;
      try {
        champion =
            CfrConfig.JSON.fromJson(
                Files.readString(runRoot.resolve("champion.json")), CfrRunStore.Champion.class);
      } catch (RuntimeException invalid) {
        throw new IOException("Invalid champion pointer", invalid);
      }
      if (champion == null
          || champion.checkpoint() == null
          || champion.solveId() == null
          || champion.iteration() < 0) throw new IOException("Invalid champion pointer");
      checkpoint = champion.checkpoint();
      solveId = champion.solveId();
      iteration = champion.iteration();
    }
    Path source = CfrFiles.inside(runRoot, checkpoint);
    CfrRunStore.verifyArtifact(source);
    Provenance provenance;
    try {
      provenance =
          CfrConfig.JSON.fromJson(
              Files.readString(source.resolve("provenance.json")), Provenance.class);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid checkpoint provenance", invalid);
    }
    if (provenance == null
        || provenance.sourceSolveId() == null
        || provenance.sourceSolveId().isBlank()
        || provenance.sourceIteration() < 0) throw new IOException("Invalid checkpoint provenance");
    if (selection.equals("champion")
        && (!provenance.sourceSolveId().equals(solveId)
            || provenance.sourceIteration() != iteration))
      throw new IOException("Champion/checkpoint provenance mismatch");
    solveId = provenance.sourceSolveId();
    iteration = provenance.sourceIteration();
    Manifest manifest =
        new Manifest(
            2,
            "epsilon-cfr",
            "AVERAGE_DEPLOYMENT",
            config.collection().mode(),
            "model",
            solveId,
            iteration,
            "policy",
            config.utilityProfile());
    Path parent = destination.getParent();
    Files.createDirectories(parent);
    Path staging = Files.createTempDirectory(parent, ".cfr-export-");
    boolean published = false;
    try {
      try (CfrModel model = CfrModel.loadModel(source, Device.cpu())) {
        if (model.role() != CfrModel.Role.AVERAGE)
          throw new IOException("Selected checkpoint is not an average policy");
        if (model.config().utilityProfile() != config.utilityProfile())
          throw new IOException("CFR run/checkpoint utility profile mismatch");
        model.saveModel(staging.resolve("model"));
      }
      Files.writeString(staging.resolve("deployment.json"), CfrConfig.JSON.toJson(manifest));
      CfrRunStore.sealArtifact(staging);
      if (Files.exists(destination)) throw new FileAlreadyExistsException(destination.toString());
      Files.move(staging, destination, StandardCopyOption.ATOMIC_MOVE);
      published = true;
      return manifest;
    } finally {
      if (!published
          && staging.toAbsolutePath().normalize().startsWith(parent)
          && staging.getFileName().toString().startsWith(".cfr-export-")) {
        try (var files = Files.walk(staging)) {
          for (Path file : files.sorted(Comparator.reverseOrder()).toList())
            Files.deleteIfExists(file);
        }
      }
    }
  }

  private static Device parseDevice(String name) {
    if (name.equals("cpu")) return Device.cpu();
    if (name.matches("gpu:[0-9]+")) return Device.gpu(Integer.parseInt(name.substring(4)));
    throw new IllegalArgumentException("device must be cpu or gpu:N");
  }

  @Override
  public void close() {
    model.close();
  }
}

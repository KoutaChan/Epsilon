package com.epsilon.cfr.training;

import com.epsilon.cfr.collection.CfrPolicy;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.io.CfrFiles;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.model.CfrModelConfig;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.network.NetworkDeviceResolver;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** LCFRの反復を、重複登録なく再開できる永続化された工程として実行する。 */
public final class CfrCampaign {
  private static final Logger LOG = LoggerFactory.getLogger(CfrCampaign.class);

  private CfrCampaign() {}

  public static void train(CfrConfig config, Path directory) throws IOException {
    try (CfrRunStore store = new CfrRunStore(directory)) {
      if (store.exists() || Files.exists(store.root().resolve("config.json")))
        throw new IOException("Run already exists; use resume: " + directory);
      config = CfrWarmStartRunner.snapshot(store, config);
      config.write(store.root().resolve("config.json"));
      CfrRunState state = initialize(store, config, 0, null, null);
      store.publish(state);
      execute(store, config, state);
    }
  }

  public static void resume(Path directory) throws IOException {
    try (CfrRunStore store = new CfrRunStore(directory)) {
      CfrConfig config = CfrConfig.read(directory.resolve("config.json"));
      CfrRunState state =
          store.exists() ? store.verify() : initialize(store, config, 0, null, null);
      if (!store.exists()) store.publish(state);
      execute(store, config, state);
    }
  }

  private static CfrRunState initialize(
      CfrRunStore store,
      CfrConfig config,
      int solve,
      String inheritedAverage,
      String calibratedBoundary)
      throws IOException {
    String solveId = "solve-" + String.format("%04d", solve) + "-" + UUID.randomUUID();
    var devices = NetworkDeviceResolver.learner(config.network().devices());
    CfrModelConfig modelConfig =
        new CfrModelConfig(
            config.network().hidden(),
            config.learning().microBatchSize(),
            config.learning().learningRate(),
            config.learning().weightDecay(),
            config.learning().gradientClip(),
            SeedMixer.derive(config.seed(), 0x4d4f44454cL, solve),
            config.collection().inferenceBatch(),
            config.learning().computePrecision(),
            config.learning().tensorTransfer(),
            config.utilityProfile());
    boolean nanoInitialization = solve == 0 && config.initialization().enabled();
    String initialAverage =
        nanoInitialization
            ? CfrWarmStartRunner.run(store, config, modelConfig, devices)
            : inheritedAverage;
    try (CfrModel regret = new CfrModel(modelConfig, CfrModel.Role.REGRET, devices);
        CfrModel average =
            initialAverage == null
                ? new CfrModel(modelConfig, CfrModel.Role.AVERAGE, devices)
                : CfrModel.loadTraining(store.resolve(initialAverage), devices)) {
      if (nanoInitialization) CfrWarmStartRunner.initializeRegret(store, config, regret);
      else if (inheritedAverage != null) regret.initializeRepresentationFrom(average);
      String regretPath = store.saveTraining(regret, "regret", 0, solveId);
      String averagePath =
          inheritedAverage == null
              ? store.saveTraining(average, "average", 0, solveId)
              : inheritedAverage;
      String boundaryPath = calibratedBoundary;
      String boundaryHash = "disabled";
      Path nanoSource = nanoInitialization ? CfrWarmStartRunner.source(store, config) : null;
      if (config.collection().mode().equals("HAND_GRP")) {
        if (nanoSource != null && Files.isDirectory(nanoSource.resolve("grp"))) {
          boundaryPath = store.relative(nanoSource);
        } else if (boundaryPath == null
            && !nanoInitialization
            && !config.boundary().checkpoint().isEmpty()) {
          Path imported = store.newArtifact("boundary", 0);
          CfrRunStore.copyGrp(Path.of(config.boundary().checkpoint()), imported);
          CfrRunStore.sealArtifact(imported);
          boundaryPath = store.relative(imported);
        }
        if (boundaryPath == null) {
          Path calibrated = store.newArtifact("boundary", 0);
          LOG.info(
              "LCFR bootstrapping boundary from {} normal-policy matches",
              config.boundary().bootstrapGames());
          CfrCalibration.run(
              calibrated,
              config,
              nanoInitialization ? average : CfrPolicy.uniform(),
              config.boundary().bootstrapGames(),
              SeedMixer.derive(config.seed(), 0x424f4f54L, solve));
          CfrRunStore.sealArtifact(calibrated);
          boundaryPath = store.relative(calibrated);
        }
        try (var grp =
            EpsilonGrpTrainingSession.openInferenceOnly(
                store.resolve(boundaryPath),
                NetworkDeviceResolver.grp(config.network().devices()).primary())) {
          boundaryHash = grp.identity().checkpointSha256();
        }
      }
      if (inheritedAverage == null) store.publishChampion(averagePath, solveId, 0);
      return new CfrRunState(
          2,
          solveId,
          solve,
          1,
          nanoInitialization ? CfrRunState.Phase.COLLECTING : CfrRunState.Phase.READY,
          0,
          0,
          regretPath,
          averagePath,
          nanoInitialization ? averagePath : null,
          0,
          boundaryPath,
          boundaryHash,
          CfrFiles.sha256(store.root().resolve("config.json")));
    }
  }

  private static void execute(CfrRunStore store, CfrConfig config, CfrRunState initial)
      throws IOException {
    CfrRunState state = initial;
    while (state.phase() != CfrRunState.Phase.COMPLETE) {
      checkInterrupt();
      if (state.iteration() > config.iterations()) {
        if (state.solveIndex() + 1 >= config.solves()) {
          state =
              state.progress(
                  CfrRunState.Phase.COMPLETE,
                  state.regretUpdate(),
                  state.averageUpdate(),
                  state.regretCheckpoint(),
                  state.averageCheckpoint(),
                  null,
                  state.replayGeneration());
          store.publish(state);
          CfrRunMaintenance.prune(store, state);
          break;
        }
        CfrRunStore.Champion champion = store.champion();
        String nextBoundary = null;
        if (config.collection().mode().equals("HAND_GRP")) {
          Path calibration = store.newArtifact("boundary", state.iteration());
          try (CfrModel policy =
              CfrModel.loadModel(
                  store.resolve(champion.checkpoint()),
                  NetworkDeviceResolver.inference(config.network().devices()).primary())) {
            CfrCalibration.run(
                calibration,
                config,
                policy,
                config.boundary().calibrationGames(),
                SeedMixer.derive(config.seed(), 0x43414c49L, state.solveIndex() + 1));
          }
          CfrRunStore.sealArtifact(calibration);
          nextBoundary = store.relative(calibration);
        }
        state =
            initialize(store, config, state.solveIndex() + 1, champion.checkpoint(), nextBoundary);
        store.publish(state);
        continue;
      }
      LOG.info(
          "LCFR solve={} iteration={} phase={}", state.solveId(), state.iteration(), state.phase());
      state = CfrIterationRunner.advance(store, config, state);
    }
    LOG.info("LCFR run complete: {}", store.root());
  }

  static void checkInterrupt() throws IOException {
    if (Thread.currentThread().isInterrupted())
      throw new IOException("LCFR interrupted; resume from the last committed phase");
  }
}

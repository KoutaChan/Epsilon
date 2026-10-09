package com.epsilon.cfr.training;

import static com.epsilon.cfr.training.CfrCampaign.checkInterrupt;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.cfr.collection.CfrCollector;
import com.epsilon.cfr.collection.CfrEpisodeSink;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrReplayStore;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.LogWeightedReplay;
import com.epsilon.cfr.io.CfrFiles;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.model.CfrModelConfig;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.io.AtomicFiles;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.network.NetworkDeviceResolver;
import com.epsilon.network.NetworkDevices;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.stream.LongStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Nanoからの初期化だけを担当し、教師標本をLCFRの累積履歴へ混ぜない。 */
final class CfrWarmStartRunner {
  private static final Logger LOG = LoggerFactory.getLogger(CfrWarmStartRunner.class);
  private static final String PROGRESS = "initialization/progress.json";

  private CfrWarmStartRunner() {}

  /** 初期化に使うNanoと関連データを実行ディレクトリ内に固定する。呼び出し側は、この処理を終えてから{@code config.json}を公開する。 */
  static CfrConfig snapshot(CfrRunStore store, CfrConfig config) throws IOException {
    if (!config.initialization().enabled()) return config;
    Path input = Path.of(config.initialization().nanoCheckpoint()).toAbsolutePath().normalize();
    Path decisionRoot =
        Files.isDirectory(input.resolve("decision")) ? input.resolve("decision") : input;
    Path exact = EpsilonDecisionCheckpointManager.resolveExistingStrict(decisionRoot);
    if (exact == null) throw new IOException("No Nano Decision checkpoint at " + input);
    var manifest = EpsilonDecisionCheckpointManager.requireValidCheckpoint(exact);
    if (manifest.hidden != config.network().hidden())
      throw new IOException(
          "Nano hidden="
              + manifest.hidden
              + " differs from CFR network.hidden="
              + config.network().hidden());
    Path snapshot = store.newArtifact("nano-source", 0);
    EpsilonDecisionCheckpointManager.copyInferenceSnapshot(exact, snapshot.resolve("decision"));
    if (config.collection().mode().equals("HAND_GRP")) {
      Path grp =
          config.boundary().checkpoint().isEmpty()
              ? EpsilonGrpCheckpointManager.resolveExisting(input.resolve("grp"))
              : Path.of(config.boundary().checkpoint());
      if (grp != null) CfrRunStore.copyGrp(grp, snapshot);
    }

    write(
        snapshot.resolve("source.json"),
        Map.of(
            "requested",
            input.toString(),
            "decision",
            exact.toString(),
            "iteration",
            manifest.iteration,
            "globalStep",
            manifest.globalStep));
    CfrRunStore.sealArtifact(snapshot);
    return config.withInitialization(
        config.initialization().withCheckpoint(store.relative(snapshot)));
  }

  static Path source(CfrRunStore store, CfrConfig config) throws IOException {
    return store.resolve(config.initialization().nanoCheckpoint());
  }

  static void initializeRegret(CfrRunStore store, CfrConfig config, CfrModel regret)
      throws IOException {
    try (Model nano =
        EpsilonDecisionCheckpointManager.load(
            source(store, config).resolve("decision"), Device.cpu())) {
      var transferred = regret.initializeRepresentationFromNano(nano);
      store.report("initialization-regret-transfer.json", transferred);
    }
  }

  /** 平均方策を蒸留し、変更不可として保存した学習チェックポイントへの相対パスを返す。 */
  static String run(
      CfrRunStore store, CfrConfig config, CfrModelConfig modelConfig, NetworkDevices devices)
      throws IOException {
    Path source = source(store, config);
    CfrRunStore.verifyArtifact(source);
    Path progressFile = store.resolve(PROGRESS);
    String configHash = CfrFiles.sha256(store.root().resolve("config.json"));
    Progress progress;
    if (Files.isRegularFile(progressFile)) {
      try {
        progress = CfrConfig.JSON.fromJson(Files.readString(progressFile), Progress.class);
      } catch (RuntimeException invalid) {
        throw new IOException("Invalid Nano initialization progress", invalid);
      }
      if (progress == null
          || progress.version != 1
          || !configHash.equals(progress.configHash)
          || progress.trainGeneration < -1
          || progress.validationGeneration < -1
          || progress.updates < 0
          || progress.updates > config.initialization().updates()
          || (progress.updates > 0 && progress.averageCheckpoint == null)
          || (progress.corpusComplete
              && (progress.trainGeneration < 0 || progress.validationGeneration < 0))
          || (progress.complete
              && (!progress.corpusComplete
                  || progress.updates != config.initialization().updates())))
        throw new IOException("Nano initialization progress/configuration mismatch");
    } else {
      progress = new Progress(1, configHash, -1, -1, false, null, 0, false);
      write(progressFile, progress);
    }
    if (progress.complete) {
      CfrRunStore.verifyArtifact(store.resolve(progress.averageCheckpoint));
      return progress.averageCheckpoint;
    }
    Progress[] committed = {progress};
    try (CfrReplayStore train = openCorpus(store, config, false, progress.trainGeneration);
        CfrReplayStore validation =
            openCorpus(store, config, true, progress.validationGeneration)) {
      if (!progress.corpusComplete) {
        List<Long> missing =
            LongStream.range(0, config.initialization().games())
                .filter(id -> !(heldout(id) ? validation : train).containsEpisode(1, id))
                .boxed()
                .toList();
        if (!missing.isEmpty()) {
          LOG.info("Nano initialization: collecting {} teacher games", missing.size());
          try (NanoTeacherPolicy teacher =
              new NanoTeacherPolicy(
                  source.resolve("decision"),
                  NetworkDeviceResolver.inference(config.network().devices()).primary(),
                  config.collection().inferenceBatch())) {
            var collector =
                new CfrCollector(
                    new CfrCollector.Config(
                        CfrCollector.Mode.MATCH_MC,
                        0,
                        config.collection().inferenceBatch(),
                        config.collection().concurrentGames(),
                        SeedMixer.derive(config.seed(), 0x4e414e4fL),
                        false,
                        true,
                        config.collection().multiTransitionMaxBatch(),
                        config.collection().arena(),
                        config.collection().batching(),
                        config.utilityProfile()),
                    new DecisionBranchComparisonSettings(false, false, true, 1, 8, .01));
            collector.collect(
                missing,
                teacher,
                null,
                CfrEpisodeSink.consuming(
                    collected -> {
                      try {
                        checkInterrupt();
                        long id = collected.episode().episodeId();
                        boolean test = heldout(id);
                        CfrReplayStore target = test ? validation : train;
                        target.appendEpisode(1, id, supervisedRows(collected.episode()));
                        long generation =
                            nextGeneration(
                                target,
                                test
                                    ? committed[0].validationGeneration
                                    : committed[0].trainGeneration);
                        target.save(generation);
                        Progress before = committed[0];
                        Progress next =
                            new Progress(
                                1,
                                configHash,
                                test ? before.trainGeneration : generation,
                                test ? generation : before.validationGeneration,
                                false,
                                null,
                                0,
                                false);
                        write(progressFile, next);
                        committed[0] = next;
                        LOG.info(
                            "Nano initialization: teacher game {}/{} committed",
                            id + 1,
                            config.initialization().games());
                      } catch (IOException failure) {
                        throw new UncheckedIOException(failure);
                      }
                    }));
          } catch (UncheckedIOException failure) {
            throw failure.getCause();
          }
        }
        progress = committed[0];
        progress =
            new Progress(
                1,
                configHash,
                progress.trainGeneration,
                progress.validationGeneration,
                true,
                null,
                0,
                false);
        write(progressFile, progress);
        committed[0] = progress;
      }
      LogWeightedReplay trainingRows = train.averageSampler();
      LogWeightedReplay validationRows = validation.averageSampler();
      if (trainingRows.size() == 0 || validationRows.size() == 0)
        throw new IOException("Nano distillation needs nonempty training and held-out decisions");
      progress = committed[0];
      if (progress.averageCheckpoint != null)
        CfrRunStore.verifyArtifact(store.resolve(progress.averageCheckpoint));
      try (CfrModel average =
          progress.averageCheckpoint == null
              ? new CfrModel(modelConfig, CfrModel.Role.AVERAGE, devices)
              : CfrModel.loadTraining(store.resolve(progress.averageCheckpoint), devices)) {
        if (progress.averageCheckpoint == null) {
          try (Model nano =
              EpsilonDecisionCheckpointManager.load(source.resolve("decision"), Device.cpu())) {
            store.report(
                "initialization-average-transfer.json",
                average.initializeRepresentationFromNano(nano));
          }
          store.report(
              "initialization-before.json",
              measure(average, validationRows, config.learning().microBatchSize()));
        }
        for (int update = progress.updates + 1;
            update <= config.initialization().updates();
            update++) {
          checkInterrupt();
          var random = new SplittableRandom(SeedMixer.derive(config.seed(), 0x44495354L, update));
          var metrics = average.train(trainingRows.sample(config.learning().batchSize(), random));
          if (update % config.learning().checkpointEveryUpdates() == 0
              || update == config.initialization().updates()) {
            String checkpoint =
                store.saveTraining(average, "initial-average", update, "initialization");
            Progress next =
                new Progress(
                    1,
                    configHash,
                    progress.trainGeneration,
                    progress.validationGeneration,
                    true,
                    checkpoint,
                    update,
                    false);
            write(progressFile, next);
            committed[0] = next;
            store.report("initialization-update-" + update + ".json", metrics);
            LOG.info(
                "Nano initialization: update {}/{} loss={}",
                update,
                config.initialization().updates(),
                metrics.loss());
          }
        }
        DistillationMetrics heldout =
            measure(average, validationRows, config.learning().microBatchSize());
        store.report("initialization-heldout.json", heldout);
        progress = committed[0];
        Progress complete =
            new Progress(
                1,
                configHash,
                progress.trainGeneration,
                progress.validationGeneration,
                true,
                progress.averageCheckpoint,
                progress.updates,
                true);
        write(progressFile, complete);
        return complete.averageCheckpoint;
      }
    }
  }

  static List<CfrTrainingRow> supervisedRows(CfrEpisode episode) {
    return episode.decisions().stream()
        .map(
            decision ->
                new CfrTrainingRow(
                    new CfrSampleId(1, episode.episodeId(), decision.decisionId(), decision.seat()),
                    decision.legalActionIds(),
                    decision.selectedSlot(),
                    decision.targetPolicy(),
                    0,
                    Double.NEGATIVE_INFINITY,
                    0,
                    decision.observation()))
        .toList();
  }

  static boolean heldout(long episodeId) {
    return episodeId % 5 == 0;
  }

  private static CfrReplayStore openCorpus(
      CfrRunStore store, CfrConfig config, boolean validation, long generation) throws IOException {
    String id = validation ? "nano-validation" : "nano-training";
    Path path = store.resolve("initialization/" + id);
    if (generation >= 0) return CfrReplayStore.loadGeneration(path, id, generation);
    int capacity =
        validation
            ? Math.max(1, Math.min(4096, config.initialization().capacity() / 5))
            : config.initialization().capacity();
    long seed = SeedMixer.derive(config.seed(), validation ? 0x56414c49L : 0x54524149L);
    return new CfrReplayStore(
        path,
        id,
        1,
        capacity,
        seed,
        seed ^ 1,
        config.replay().maxRowBytes(),
        Math.min(64L << 20, config.replay().maxRowBytes() * config.replay().cacheRows()),
        config.replay().maxStoredBytes());
  }

  private static long nextGeneration(CfrReplayStore store, long previous) {
    long next = previous + 1;
    while (Files.exists(CfrReplayStore.snapshotPath(store.directory(), next))) next++;
    return next;
  }

  static DistillationMetrics measure(CfrModel model, LogWeightedReplay rows, int batchSize) {
    double ce = 0, entropy = 0;
    int matches = 0;
    for (int from = 0; from < rows.size(); from += batchSize) {
      List<CfrTrainingRow> batch = new ArrayList<>();
      for (int index = from; index < Math.min(rows.size(), from + batchSize); index++)
        batch.add(rows.rowAt(index));
      var predictions =
          model.evaluateObservations(batch.stream().map(CfrTrainingRow::observation).toList());
      for (int row = 0; row < batch.size(); row++) {
        double[] target = batch.get(row).targetPolicy();
        double[] prediction = predictions.get(row).targetPolicy();
        int expected = 0, actual = 0;
        for (int action = 0; action < target.length; action++) {
          if (target[action] > target[expected]) expected = action;
          if (prediction[action] > prediction[actual]) actual = action;
          if (target[action] > 0) {
            ce -= target[action] * Math.log(Math.max(1e-300, prediction[action]));
            entropy -= target[action] * Math.log(target[action]);
          }
        }
        if (expected == actual) matches++;
      }
    }
    return new DistillationMetrics(
        rows.size(),
        ce / rows.size(),
        Math.max(0, (ce - entropy) / rows.size()),
        (double) matches / rows.size());
  }

  static record DistillationMetrics(
      int samples, double crossEntropy, double klDivergence, double top1Agreement) {}

  private static void write(Path path, Object value) throws IOException {
    AtomicFiles.write(path, CfrConfig.JSON.toJson(value).getBytes(StandardCharsets.UTF_8));
  }

  private record Progress(
      int version,
      String configHash,
      long trainGeneration,
      long validationGeneration,
      boolean corpusComplete,
      String averageCheckpoint,
      int updates,
      boolean complete) {}
}

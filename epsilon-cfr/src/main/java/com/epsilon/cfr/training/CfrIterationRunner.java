package com.epsilon.cfr.training;

import static com.epsilon.cfr.training.CfrCampaign.checkInterrupt;

import ai.djl.Device;
import com.epsilon.cfr.algorithm.LcfrTargets;
import com.epsilon.cfr.collection.CfrCollector;
import com.epsilon.cfr.collection.CfrEpisodeSink;
import com.epsilon.cfr.collection.FrozenGrpBoundary;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrEpisodeWriter;
import com.epsilon.cfr.data.CfrMatchFile;
import com.epsilon.cfr.data.CfrReplayStore;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.LogWeightedReplay;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.runtime.CfrPolicyFleet;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.network.NetworkDeviceResolver;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 一反復の収集・再生登録・学習を、保存済みの工程境界から進める。 */
final class CfrIterationRunner {
  private static final Logger LOG = LoggerFactory.getLogger(CfrIterationRunner.class);

  private CfrIterationRunner() {}

  static CfrRunState advance(CfrRunStore store, CfrConfig config, CfrRunState state)
      throws IOException {
    switch (state.phase()) {
      case READY -> {
        Path frozen = store.newArtifact("collector", state.iteration());
        try (CfrModel model =
            CfrModel.loadModel(store.resolve(state.regretCheckpoint()), Device.cpu())) {
          model.saveModel(frozen);
        }
        CfrRunStore.sealArtifact(frozen);
        state =
            state.progress(
                CfrRunState.Phase.COLLECTING,
                0,
                0,
                state.regretCheckpoint(),
                state.averageCheckpoint(),
                store.relative(frozen),
                state.replayGeneration());
        store.publish(state);
      }
      case COLLECTING -> {
        collect(store, config, state);
        state =
            state.progress(
                CfrRunState.Phase.SEALED,
                0,
                0,
                state.regretCheckpoint(),
                state.averageCheckpoint(),
                state.frozenCheckpoint(),
                state.replayGeneration());
        store.publish(state);
      }
      case SEALED -> {
        long trajectories = 0;
        long decisions = 0;
        for (long game = 0; game < config.collection().games(); game++) {
          var summary = CfrMatchFile.summary(matchPath(store, state, game));
          if (summary.gameId() != game) throw new IOException("Match file identity mismatch");
          trajectories = Math.addExact(trajectories, summary.trajectories());
          decisions = Math.addExact(decisions, summary.decisions());
        }
        store.report(
            reportName(state, "collection-total"),
            Map.of(
                "games",
                config.collection().games(),
                "trajectories",
                trajectories,
                "decisions",
                decisions));
        long generation = state.replayGeneration() + 1;
        Path replayDirectory = store.resolve("solves/" + state.solveId() + "/replay");
        while (Files.exists(CfrReplayStore.snapshotPath(replayDirectory, generation))) generation++;
        try (CfrReplayStore replay = openReplay(store, config, state)) {
          replay.collectGarbage();
          var storage = replay.stats();
          if (storage.livePayloadBytes() > 0
              && storage.storedBytes() / 2 > storage.livePayloadBytes()
              && storage.livePayloadBytes()
                  <= config.replay().maxStoredBytes() - storage.storedBytes()) {
            // 新しい世代の保存が確定するまでは、保存済みの世代が参照する古いデータも保持する。
            replay.compact();
          }
          for (long game = 0; game < config.collection().games(); game++) {
            checkInterrupt();
            List<CfrTrainingRow> rows = new ArrayList<>();
            // 局内の重要度比を局ごとに計算し、反復全体の実軌跡数で重みを正規化する。
            for (CfrEpisode trajectory : CfrMatchFile.read(matchPath(store, state, game))) {
              if (trajectory.episodeId() != game)
                throw new IOException("Match file identity mismatch");
              rows.addAll(LcfrTargets.generate(trajectory, state.iteration(), trajectories));
            }
            replay.appendEpisode(state.iteration(), game, rows);
          }
          replay.save(generation);
          store.report(reportName(state, "replay"), replay.stats());
        }
        state =
            state.progress(
                CfrRunState.Phase.REPLAY_COMMITTED,
                0,
                0,
                state.regretCheckpoint(),
                state.averageCheckpoint(),
                state.frozenCheckpoint(),
                generation);
        store.publish(state);
      }
      case REPLAY_COMMITTED -> {
        CfrRunMaintenance.prune(store, state);
        state = learn(store, config, state, true);
      }
      case TRAINING_REGRET -> state = learn(store, config, state, true);
      case TRAINING_AVERAGE -> state = learn(store, config, state, false);
      case TRAINED -> {
        if (config.evaluatesAt(state.iteration()))
          CfrEvaluationRunner.evaluate(store, config, state);
        state = state.nextIteration();
        store.publish(state);
      }
      case COMPLETE -> throw new AssertionError();
    }
    return state;
  }

  private static void collect(CfrRunStore store, CfrConfig config, CfrRunState state)
      throws IOException {
    List<Long> missing = new ArrayList<>();
    for (long game = 0; game < config.collection().games(); game++) {
      Path path = matchPath(store, state, game);
      if (!Files.exists(path)) missing.add(game);
      else if (CfrMatchFile.summary(path).gameId() != game)
        throw new IOException("Match identity mismatch " + path);
    }
    store.recordCollectedPolicy(state);
    var devices = NetworkDeviceResolver.inference(config.network().devices());
    try (CfrModel source =
            CfrModel.loadModel(store.resolve(state.frozenCheckpoint()), devices.primary());
        CfrPolicyFleet fleet = new CfrPolicyFleet(source, devices, config.collection());
        EpsilonGrpTrainingSession grp =
            state.boundaryRoot() == null
                ? null
                : EpsilonGrpTrainingSession.openInferenceOnly(
                    store.resolve(state.boundaryRoot()),
                    NetworkDeviceResolver.grp(config.network().devices()).primary())) {
      if (grp != null && !grp.identity().checkpointSha256().equals(state.boundaryHash()))
        throw new IOException("Fixed GRP identity changed");
      CfrCollector collector =
          new CfrCollector(
              new CfrCollector.Config(
                  CfrCollector.Mode.valueOf(config.collection().mode()),
                  config.collection().exploration(),
                  config.collection().inferenceBatch(),
                  config.collection().concurrentGames(),
                  SeedMixer.derive(
                      config.seed(), 0x434f4c4cL, state.solveIndex(), state.iteration()),
                  config.collection().centerUtility(),
                  true,
                  config.collection().multiTransitionMaxBatch(),
                  config.collection().arena(),
                  config.collection().batching(),
                  config.utilityProfile()),
              config.collection().branchComparison());
      long started = System.nanoTime();
      CfrCollector.CollectionStats stats;
      try (var writer = new CfrEpisodeWriter(matchPath(store, state, 0).getParent())) {
        stats =
            collector.collect(
                missing,
                fleet,
                grp == null
                    ? null
                    : new FrozenGrpBoundary(grp.inference(), config.utilityProfile()),
                CfrEpisodeSink.persisting(writer));
      }
      // 非同期で保存するすべてのファイルがディスクへ同期され、公開されてから収集完了とする。
      stats =
          new CfrCollector.CollectionStats(
              stats.games(),
              stats.hands(),
              stats.trajectories(),
              stats.decisions(),
              stats.inferenceBatches(),
              stats.largestBatch(),
              (System.nanoTime() - started) / 1e9,
              stats.branchComparisons(),
              stats.branchEpisodes(),
              stats.branchInferenceRows());
      store.report(reportName(state, "collection-" + UUID.randomUUID()), stats);
    } catch (UncheckedIOException failure) {
      throw failure.getCause();
    }
  }

  private static CfrRunState learn(
      CfrRunStore store, CfrConfig config, CfrRunState start, boolean regret) throws IOException {
    if (!regret && !config.evaluatesAt(start.iteration())) {
      CfrRunState done =
          start.progress(
              CfrRunState.Phase.TRAINED,
              start.regretUpdate(),
              0,
              start.regretCheckpoint(),
              start.averageCheckpoint(),
              start.frozenCheckpoint(),
              start.replayGeneration());
      store.publish(done);
      return done;
    }
    int from = regret ? start.regretUpdate() : start.averageUpdate();
    int until = regret ? config.learning().regretUpdates() : config.learning().averageUpdates();
    CfrRunState state = start;
    try (CfrReplayStore replay = openReplay(store, config, state);
        CfrModel learner =
            CfrModel.loadTraining(
                store.resolve(regret ? state.regretCheckpoint() : state.averageCheckpoint()),
                NetworkDeviceResolver.learner(config.network().devices()))) {
      if (learner.role() != (regret ? CfrModel.Role.REGRET : CfrModel.Role.AVERAGE)) {
        throw new IOException("Checkpoint role differs from the learner objective");
      }
      LogWeightedReplay sampler = regret ? replay.regretSampler() : replay.averageSampler();
      if (sampler.metrics().positiveWeightRows() == 0)
        throw new IOException(
            "LCFR objective has no positive-weight samples; inspect collection weights");
      store.report(
          reportName(state, regret ? "regret-weights" : "average-weights"), sampler.metrics());
      for (int update = from; update < until; update++) {
        checkInterrupt();
        long drawSeed =
            SeedMixer.derive(
                config.seed(),
                regret ? 0x5245504cL : 0x4156474cL,
                state.solveIndex(),
                state.iteration(),
                update);
        var metrics =
            learner.train(
                sampler.sample(config.learning().batchSize(), new SplittableRandom(drawSeed)));
        LOG.info(
            "LCFR {} iteration={} update={}/{} loss={} samples={}",
            regret ? "regret" : "average",
            state.iteration(),
            update + 1,
            until,
            metrics.loss(),
            metrics.samples());
        if ((update + 1) % config.learning().checkpointEveryUpdates() == 0 || update + 1 == until) {
          String checkpoint =
              store.saveTraining(
                  learner, regret ? "regret" : "average", state.iteration(), state.solveId());
          state =
              state.progress(
                  regret ? CfrRunState.Phase.TRAINING_REGRET : CfrRunState.Phase.TRAINING_AVERAGE,
                  regret ? update + 1 : state.regretUpdate(),
                  regret ? state.averageUpdate() : update + 1,
                  regret ? checkpoint : state.regretCheckpoint(),
                  regret ? state.averageCheckpoint() : checkpoint,
                  state.frozenCheckpoint(),
                  state.replayGeneration());
          store.publish(state);
          store.report(
              reportName(state, (regret ? "regret-" : "average-") + (update + 1)), metrics);
        }
      }
    }
    state =
        state.progress(
            regret ? CfrRunState.Phase.TRAINING_AVERAGE : CfrRunState.Phase.TRAINED,
            state.regretUpdate(),
            state.averageUpdate(),
            state.regretCheckpoint(),
            state.averageCheckpoint(),
            state.frozenCheckpoint(),
            state.replayGeneration());
    store.publish(state);
    return state;
  }

  private static CfrReplayStore openReplay(CfrRunStore store, CfrConfig config, CfrRunState state)
      throws IOException {
    Path directory = store.resolve("solves/" + state.solveId() + "/replay");
    if (state.replayGeneration() > 0)
      return CfrReplayStore.loadGeneration(directory, state.solveId(), state.replayGeneration());
    var settings = config.replay();
    return new CfrReplayStore(
        directory,
        state.solveId(),
        settings.regretCapacity(),
        settings.averageCapacity(),
        SeedMixer.derive(config.seed(), 0x52524553L, state.solveIndex()),
        SeedMixer.derive(config.seed(), 0x41524553L, state.solveIndex()),
        settings.maxRowBytes(),
        Math.min(256L << 20, settings.cacheRows() * settings.maxRowBytes()),
        settings.maxStoredBytes());
  }

  private static Path matchPath(CfrRunStore store, CfrRunState state, long id) throws IOException {
    return store.resolve(
        "solves/"
            + state.solveId()
            + "/matches/"
            + state.iteration()
            + "/"
            + String.format("%08d", id)
            + ".bin");
  }

  static String reportName(CfrRunState state, String suffix) {
    return state.solveId() + "-" + state.iteration() + "-" + suffix + ".json";
  }
}

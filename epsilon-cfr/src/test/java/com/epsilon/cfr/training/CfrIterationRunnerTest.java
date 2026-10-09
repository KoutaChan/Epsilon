package com.epsilon.cfr.training;

import com.epsilon.cfr.algorithm.LcfrTargets;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.config.settings.EpsilonSettings;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrReplayStore;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.cfr.io.CfrFiles;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** モデルの構築や対局を行わず、リプレイの公開が中断された場合の復旧を検証する。 */
public class CfrIterationRunnerTest {
  @DataProvider
  public Object[][] nonEvaluationIterations() {
    return new Object[][] {{1L}, {4L}, {8L}, {9L}, {11L}};
  }

  @Test(dataProvider = "nonEvaluationIterations")
  public void skipsAverageFittingAndPreservesReplayAndSavedModels(long iteration) throws Exception {
    Path directory = Files.createTempDirectory("cfr-average-schedule-");
    CfrConfig config = CfrConfig.load(EpsilonSettings.of(Map.of("epsilon.cfr.iterations", "25")));
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        config.write(directory.resolve("config.json"));
        String regret = artifact(store, "regret");
        String average = artifact(store, "average");
        String frozen = artifact(store, "collector");
        Path replayDirectory = store.resolve("solves/solve-test/replay");
        try (var replay =
            new CfrReplayStore(
                replayDirectory, "solve-test", 16, 16, 3, 4, 1024, 0, Long.MAX_VALUE)) {
          replay.appendEpisode(iteration, 0, List.of(row(iteration)));
          replay.save(1);
        }
        Path replaySnapshot = CfrReplayStore.snapshotPath(replayDirectory, 1);
        byte[] savedReplay = Files.readAllBytes(replaySnapshot);
        CfrRunState state =
            new CfrRunState(
                2,
                "solve-test",
                0,
                iteration,
                CfrRunState.Phase.TRAINING_AVERAGE,
                config.learning().regretUpdates(),
                0,
                regret,
                average,
                frozen,
                1,
                null,
                "disabled",
                CfrFiles.sha256(directory.resolve("config.json")));
        store.publish(state);

        // 確定済みの仮ファイルはモデルとして読み込めないため、学習をスキップしたことも確認できる。
        CfrRunState trained = CfrIterationRunner.advance(store, config, store.verify());
        Assert.assertEquals(trained.phase(), CfrRunState.Phase.TRAINED);
        Assert.assertEquals(trained.regretUpdate(), config.learning().regretUpdates());
        Assert.assertEquals(trained.averageUpdate(), 0);
        Assert.assertEquals(trained.regretCheckpoint(), regret);
        Assert.assertEquals(trained.averageCheckpoint(), average);
        Assert.assertEquals(trained.frozenCheckpoint(), frozen);
        Assert.assertEquals(trained.replayGeneration(), 1L);
        Assert.assertEquals(store.verify(), trained);

        CfrRunState next = CfrIterationRunner.advance(store, config, trained);
        Assert.assertEquals(next.phase(), CfrRunState.Phase.READY);
        Assert.assertEquals(next.iteration(), iteration + 1);
        Assert.assertEquals(next.averageCheckpoint(), average);
        Assert.assertEquals(next.replayGeneration(), 1L);
        Assert.assertEquals(store.verify(), next);
        Assert.assertEquals(Files.readAllBytes(replaySnapshot), savedReplay);
        try (var replay = CfrReplayStore.loadGeneration(replayDirectory, "solve-test", 1)) {
          Assert.assertEquals(replay.stats().episodes(), 1L);
          var teacher = replay.averageSampler().sample(1, new SplittableRandom(1)).getFirst();
          Assert.assertEquals(teacher.id().iteration(), iteration);
          Assert.assertEquals(teacher.targetPolicy(), new double[] {.6, .4});
        }
      }
    } finally {
      delete(directory);
    }
  }

  private static String artifact(CfrRunStore store, String kind) throws IOException {
    Path artifact = store.newArtifact(kind, 0);
    Files.writeString(artifact.resolve("weights.bin"), kind);
    CfrRunStore.sealArtifact(artifact);
    return store.relative(artifact);
  }

  private static CfrTrainingRow row(long iteration) {
    return new CfrTrainingRow(
        new CfrSampleId(iteration, 0, 0, 0),
        new int[] {0, 1},
        0,
        new double[] {.6, .4},
        1,
        0,
        0,
        ObservationPayload.copyOf(new byte[256]));
  }

  @DataProvider
  public Object[][] committedGenerations() {
    return new Object[][] {{0L}, {1L}};
  }

  @Test(dataProvider = "committedGenerations")
  public void sealedIterationIgnoresUnpublishedReplayGeneration(long committedGeneration)
      throws Exception {
    Path directory = Files.createTempDirectory("cfr-replay-recovery-");
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrConfig config = config();
        CfrRunState sealed = sealedState(store, config, committedGeneration);
        Path replayDirectory = store.resolve("solves/" + sealed.solveId() + "/replay");
        try (CfrReplayStore replay =
            new CfrReplayStore(
                replayDirectory,
                sealed.solveId(),
                16,
                16,
                123,
                456,
                config.replay().maxRowBytes(),
                0,
                Long.MAX_VALUE)) {
          if (committedGeneration > 0) {
            replay.appendEpisode(1, 0, LcfrTargets.generate(episode(0, 1), 1, 1));
            replay.save(committedGeneration);
          }
          // リプレイ保存後、キャンペーン状態を公開する前の異常終了を再現する。
          // 未公開のリプレイを確定済みの収集データと異なる内容にし、load(current)の誤読を検出する。
          replay.appendEpisode(
              sealed.iteration(), 0, LcfrTargets.generate(episode(0, 99), sealed.iteration(), 1));
          replay.save(committedGeneration + 1);
        }
        CfrTrainingTestFiles.writeMatch(episodePath(store, sealed), episode(0, 2));
        store.publish(sealed);

        CfrRunState recovered = CfrIterationRunner.advance(store, config, store.read());

        Assert.assertEquals(recovered.phase(), CfrRunState.Phase.REPLAY_COMMITTED);
        Assert.assertEquals(recovered.replayGeneration(), committedGeneration + 2);
        Assert.assertEquals(store.read(), recovered);
        try (CfrReplayStore replay =
            CfrReplayStore.loadGeneration(
                replayDirectory, sealed.solveId(), recovered.replayGeneration())) {
          Assert.assertEquals(replay.stats().episodes(), committedGeneration + 1);
          Assert.assertEquals(replay.stats().seenRows(), committedGeneration + 1);
          Assert.assertEquals(replay.verifyPayloads(), (int) committedGeneration + 1);
          Assert.assertTrue(replay.containsEpisode(sealed.iteration(), 0));
          var rows = replay.regretSampler().sample(100, new SplittableRandom(5));
          Assert.assertTrue(
              rows.stream().anyMatch(row -> row.id().iteration() == sealed.iteration()));
          Assert.assertTrue(
              rows.stream()
                  .filter(row -> row.id().iteration() == sealed.iteration())
                  .allMatch(row -> row.utility() == 2));
        }
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void normalizesEachHandByTheCollectedTrajectoryCount() throws Exception {
    Path directory = Files.createTempDirectory("cfr-hand-normalization-");
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrConfig config =
            CfrConfig.load(
                EpsilonSettings.of(
                    Map.of(
                        "epsilon.cfr.collection.games", "1",
                        "epsilon.cfr.replay.regretCapacity", "16",
                        "epsilon.cfr.replay.averageCapacity", "16")));
        CfrRunState sealed = sealedState(store, config, 0);
        CfrEpisode first = episode(0, 2);
        CfrEpisode second =
            new CfrEpisode(
                0,
                List.of(
                    new CfrDecision(
                        1,
                        0,
                        new int[] {10, 20},
                        0,
                        new double[] {.25, .75},
                        .5,
                        ObservationPayload.copyOf(new byte[] {4, 5, 6}))),
                new double[] {3, 0, 0, 0});
        CfrTrainingTestFiles.writeMatch(episodePath(store, sealed), first, second);
        store.publish(sealed);
        CfrRunState committed = CfrIterationRunner.advance(store, config, sealed);
        try (var replay =
            CfrReplayStore.loadGeneration(
                store.resolve("solves/" + sealed.solveId() + "/replay"),
                sealed.solveId(),
                committed.replayGeneration())) {
          Assert.assertEquals(replay.stats().episodes(), 1L);
          Assert.assertEquals(replay.stats().seenRows(), 2L);
          var sampler = replay.averageSampler();
          Assert.assertEquals(sampler.size(), 2);
          for (int index = 0; index < sampler.size(); index++) {
            var row = sampler.rowAt(index);
            Assert.assertEquals(row.id().episodeId(), 0L);
            Assert.assertEquals(
                row.logAverageWeight(), Math.log(sealed.iteration()) - Math.log(2), 1e-12);
          }
          Assert.assertNotEquals(sampler.rowAt(0).id(), sampler.rowAt(1).id());
        }
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void invalidSealedEpisodePreservesPublishedRecoveryPoint() throws Exception {
    Path directory = Files.createTempDirectory("cfr-sealed-invalid-");
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrConfig config = config();
        CfrRunState sealed = sealedState(store, config, 0);
        CfrTrainingTestFiles.writeMatch(episodePath(store, sealed), episode(1, 2));
        store.publish(sealed);
        byte[] pointer = Files.readAllBytes(directory.resolve("current.json"));

        Assert.expectThrows(
            IOException.class, () -> CfrIterationRunner.advance(store, config, sealed));

        Assert.assertEquals(Files.readAllBytes(directory.resolve("current.json")), pointer);
        Assert.assertEquals(store.read(), sealed);
        Assert.assertFalse(
            Files.exists(
                CfrReplayStore.snapshotPath(
                    store.resolve("solves/" + sealed.solveId() + "/replay"), 1)));
      }
    } finally {
      delete(directory);
    }
  }

  private static CfrEpisode episode(long id, double utility) {
    return new CfrEpisode(
        id,
        List.of(
            new CfrDecision(
                0,
                0,
                new int[] {10, 20},
                0,
                new double[] {.5, .5},
                .5,
                ObservationPayload.copyOf(new byte[] {1, 2, 3}))),
        new double[] {utility, 0, 0, 0});
  }

  private static CfrConfig config() {
    CfrConfig defaults = CfrConfig.defaults();
    return new CfrConfig(
        defaults.seed(),
        1,
        3,
        CfrConfig.defaults().utilityProfile(),
        new CfrConfig.Collection(
            "MATCH_MC",
            1,
            1,
            1,
            .1,
            false,
            1,
            2,
            4,
            new DecisionDuelArenaSettings(
                Math.min(32, Runtime.getRuntime().availableProcessors()), 2, 64),
            new InferenceBatchingSettings(0),
            new DecisionBranchComparisonSettings(false, false, true, 1, 8, .01)),
        defaults.learning(),
        new CfrConfig.Replay(16, 16, 1 << 20, 1024, 1),
        defaults.network(),
        defaults.boundary(),
        defaults.evaluation(),
        CfrConfig.defaults().initialization());
  }

  private static CfrRunState sealedState(
      CfrRunStore store, CfrConfig config, long committedGeneration) throws IOException {
    config.write(store.root().resolve("config.json"));
    Path model = store.newArtifact("model", 0);
    Files.write(model.resolve("weights.bin"), new byte[] {1, 2, 3});
    CfrRunStore.sealArtifact(model);
    return new CfrRunState(
        2,
        "solve-test",
        0,
        committedGeneration + 1,
        CfrRunState.Phase.SEALED,
        0,
        0,
        store.relative(model),
        store.relative(model),
        store.relative(model),
        committedGeneration,
        null,
        "disabled",
        CfrFiles.sha256(store.root().resolve("config.json")));
  }

  private static Path episodePath(CfrRunStore store, CfrRunState state) throws IOException {
    return store.resolve(
        "solves/" + state.solveId() + "/matches/" + state.iteration() + "/00000000.bin");
  }

  private static void delete(Path directory) throws IOException {
    try (var files = Files.walk(directory)) {
      for (Path path : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}

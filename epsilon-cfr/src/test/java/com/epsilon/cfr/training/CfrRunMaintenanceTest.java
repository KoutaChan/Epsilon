package com.epsilon.cfr.training;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.data.CfrReplayStore;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.cfr.io.CfrFiles;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrRunMaintenanceTest {
  @Test
  public void pruningRetainsRecoveryGenerationsAndPreservesUncommittedAndOtherSolveData()
      throws Exception {
    Path directory = Files.createTempDirectory("cfr-maintenance-");
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrRunState state = state(store, CfrRunState.Phase.REPLAY_COMMITTED, 3, 3);
        Path replayDirectory = store.resolve("solves/solve-test/replay");
        try (var replay =
            new CfrReplayStore(
                replayDirectory, "solve-test", 16, 16, 3, 4, 1024, 0, Long.MAX_VALUE)) {
          for (int generation = 1; generation <= 4; generation++) {
            replay.appendEpisode(generation, 0, List.of(row(generation)));
            replay.save(generation);
          }
        }
        Path old = spool(store, "solve-test", 1);
        Path committed = spool(store, "solve-test", 3);
        Path future = spool(store, "solve-test", 4);
        Path other = spool(store, "other-solve", 1);
        Path protectedArtifact = store.resolve(state.regretCheckpoint()).resolve("weights.bin");
        store.publish(state);
        CfrRunMaintenance.prune(store, state);

        Assert.assertFalse(Files.exists(old));
        Assert.assertFalse(Files.exists(committed));
        Assert.assertTrue(Files.exists(future));
        Assert.assertTrue(Files.exists(other));
        Assert.assertTrue(Files.exists(protectedArtifact));
        Assert.assertFalse(Files.exists(CfrReplayStore.snapshotPath(replayDirectory, 1)));
        Assert.assertTrue(Files.exists(CfrReplayStore.snapshotPath(replayDirectory, 2)));
        Assert.assertTrue(Files.exists(CfrReplayStore.snapshotPath(replayDirectory, 3)));
        Assert.assertFalse(Files.exists(CfrReplayStore.snapshotPath(replayDirectory, 4)));
        Assert.assertEquals(store.read(), state);
        try (var latest = CfrReplayStore.load(replayDirectory, "solve-test")) {
          Assert.assertEquals(latest.stats().episodes(), 3);
          Assert.assertEquals(latest.verifyPayloads(), 3);
          Assert.assertEquals(latest.storedBytes(), latest.stats().livePayloadBytes());
        }
        try (var previous = CfrReplayStore.loadGeneration(replayDirectory, "solve-test", 2)) {
          Assert.assertEquals(previous.verifyPayloads(), 2);
        }
        CfrRunMaintenance.prune(store, state);
        Assert.assertTrue(Files.exists(future));
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void staleStateCannotDeleteFilesAndUncommittedCollectionKeepsSpool() throws Exception {
    Path directory = Files.createTempDirectory("cfr-maintenance-state-");
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrRunState sealed = state(store, CfrRunState.Phase.SEALED, 1, 0);
        Path spool = spool(store, "solve-test", 1);
        store.publish(sealed);
        CfrRunMaintenance.prune(store, sealed);
        Assert.assertTrue(Files.exists(spool));
        CfrRunState unpublished =
            sealed.progress(
                CfrRunState.Phase.REPLAY_COMMITTED,
                0,
                0,
                sealed.regretCheckpoint(),
                sealed.averageCheckpoint(),
                sealed.frozenCheckpoint(),
                1);
        byte[] pointer = Files.readAllBytes(directory.resolve("current.json"));
        Assert.expectThrows(IOException.class, () -> CfrRunMaintenance.prune(store, unpublished));
        Assert.assertTrue(Files.exists(spool));
        Assert.assertEquals(Files.readAllBytes(directory.resolve("current.json")), pointer);
      }
    } finally {
      delete(directory);
    }
  }

  private static CfrTrainingRow row(long iteration) {
    return new CfrTrainingRow(
        new CfrSampleId(iteration, 0, 0, 0),
        new int[] {0, 1},
        0,
        new double[] {.5, .5},
        1,
        0,
        0,
        ObservationPayload.copyOf(new byte[256]));
  }

  private static Path spool(CfrRunStore store, String solve, long iteration) throws IOException {
    Path file = store.resolve("solves/" + solve + "/matches/" + iteration + "/00000000.bin");
    Files.createDirectories(file.getParent());
    Files.write(file, new byte[] {1});
    return file;
  }

  private static CfrRunState state(
      CfrRunStore store, CfrRunState.Phase phase, long iteration, long replayGeneration)
      throws IOException {
    CfrConfig.defaults().write(store.root().resolve("config.json"));
    Path artifact = store.newArtifact("model", 0);
    Files.write(artifact.resolve("weights.bin"), new byte[] {1, 2, 3});
    CfrRunStore.sealArtifact(artifact);
    return new CfrRunState(
        2,
        "solve-test",
        0,
        iteration,
        phase,
        0,
        0,
        store.relative(artifact),
        store.relative(artifact),
        store.relative(artifact),
        replayGeneration,
        null,
        "disabled",
        CfrFiles.sha256(store.root().resolve("config.json")));
  }

  private static void delete(Path directory) throws IOException {
    try (var files = Files.walk(directory)) {
      for (Path path : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}

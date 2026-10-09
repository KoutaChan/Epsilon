package com.epsilon.cfr.data;

import com.epsilon.cfr.algorithm.LcfrTargets;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrBranchDataTest {
  @Test
  public void branchEpisodeAndExplicitTargetsSurviveBinaryAndDurableReplay() throws Exception {
    Path directory = Files.createTempDirectory("cfr-branch-codec-");
    try {
      var decisions =
          List.of(
              new CfrDecision(
                  4,
                  0,
                  new int[] {4, 5},
                  1,
                  new double[] {.7, .3},
                  .4,
                  ObservationPayload.copyOf(new byte[] {1})),
              new CfrDecision(
                  8,
                  1,
                  new int[] {4, 5},
                  0,
                  new double[] {.6, .4},
                  .5,
                  ObservationPayload.copyOf(new byte[] {2})));
      var episode =
          new CfrEpisode(
              7,
              decisions,
              new double[] {1, -.2, .1, 0},
              List.of(
                  new CfrBranchOutcome(4, .4, 0, 1, -.4, Math.log(.75)),
                  new CfrBranchOutcome(8, 1, -1, 0, 0, Double.NEGATIVE_INFINITY)));
      Path file = directory.resolve("match.bin");
      new CfrMatchFile.Appender(file, episode.episodeId())
          .append(new CfrEpisodeFrame(episode, 0, true));
      var decoded = CfrMatchFile.read(file).getFirst();
      Assert.assertEquals(decoded.branchOutcomes(), episode.branchOutcomes());
      var rows = LcfrTargets.generate(episode, 2, 8);
      var decodedRows = LcfrTargets.generate(decoded, 2, 8);
      for (int index = 0; index < rows.size(); index++) {
        Assert.assertTrue(rows.get(index).contentEquals(decodedRows.get(index)));
      }
      for (CfrTrainingRow row : rows) {
        var bytes = new ByteArrayOutputStream();
        CfrReplayCodec.writeRow(new DataOutputStream(bytes), row);
        Assert.assertEquals(row.serializedBytes(), (long) bytes.size());
      }
      Path replayDirectory = directory.resolve("replay");
      try (var replay =
          new CfrReplayStore(replayDirectory, "branch", 8, 8, 4, 5, 4096, 0, 100000)) {
        replay.appendEpisode(2, 7, rows);
        replay.save(1);
      }
      try (var replay = CfrReplayStore.load(replayDirectory, "branch")) {
        for (CfrTrainingRow restored : replay.regretSampler().sample(20, new Random(7))) {
          CfrTrainingRow expected =
              rows.stream().filter(row -> row.id().equals(restored.id())).findFirst().orElseThrow();
          Assert.assertTrue(restored.contentEquals(expected));
        }
        Assert.assertEquals(replay.verifyPayloads(), rows.size());
      }
    } finally {
      try (var files = Files.walk(directory)) {
        for (Path path : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  @Test
  public void explicitRegretTargetsAreImmutableAndPartOfRowIdentity() {
    double[] targets = {.6, -.4};
    var row =
        new CfrTrainingRow(
            new CfrSampleId(1, 0, 0, 0),
            new int[] {0, 1},
            0,
            new double[] {.4, .6},
            1,
            0,
            0,
            ObservationPayload.copyOf(new byte[0]),
            targets);
    targets[0] = 99;
    row.explicitRegretTargets()[0] = 88;
    row.regretTargets()[0] = 77;
    Assert.assertEquals(row.regretTargets(), new double[] {.6, -.4});
    var ordinary =
        new CfrTrainingRow(
            row.id(),
            row.legalActionIds(),
            row.selectedSlot(),
            row.targetPolicy(),
            row.utility(),
            row.logRegretWeight(),
            row.logAverageWeight(),
            row.observation());
    Assert.assertFalse(row.contentEquals(ordinary));
    Assert.assertEquals(row.serializedBytes() - ordinary.serializedBytes(), 16L);
  }

  @Test
  public void invalidBranchReferencesAndMetadataAreRejected() {
    CfrDecision decision =
        new CfrDecision(
            2,
            0,
            new int[] {4, 5},
            0,
            new double[] {.6, .4},
            .5,
            ObservationPayload.copyOf(new byte[0]));
    for (CfrBranchOutcome invalid :
        List.of(
            new CfrBranchOutcome(1, 1, 1, 1, 0, 0),
            new CfrBranchOutcome(2, 1, 0, 1, 0, 0),
            new CfrBranchOutcome(2, 1, 2, 1, 0, 0),
            new CfrBranchOutcome(2, .4, 1, 1, 0, 0))) {
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> new CfrEpisode(0, List.of(decision), new double[4], List.of(invalid)));
    }
    var valid = new CfrBranchOutcome(2, 1, 1, 1, 0, 0);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new CfrEpisode(0, List.of(decision), new double[4], List.of(valid, valid)));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new CfrBranchOutcome(2, .5, -1, 0, 0, Double.NEGATIVE_INFINITY));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> new CfrBranchOutcome(2, 1, 1, 0, 0, 0));
  }
}

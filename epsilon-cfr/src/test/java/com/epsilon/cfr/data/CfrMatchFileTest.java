package com.epsilon.cfr.data;

import static org.testng.Assert.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.annotations.Test;

public final class CfrMatchFileTest {
  @Test
  public void publishesOnlyTheLastFrameAndRetainsAllTrajectories() throws Exception {
    Path directory = Files.createTempDirectory("cfr-match");
    Path target = directory.resolve("match.bin");
    try {
      var writer = new CfrMatchFile.Appender(target, 7);
      var first = episode(7, 2, .25);
      var second = episode(7, 10, -.75);
      writer.append(new CfrEpisodeFrame(first, 0, false));
      assertFalse(Files.exists(target));
      assertTrue(Files.exists(directory.resolve("match.bin.part")));
      expectThrows(
          IOException.class, () -> CfrMatchFile.summary(directory.resolve("match.bin.part")));
      writer.append(new CfrEpisodeFrame(second, 1, true));
      assertFalse(Files.exists(directory.resolve("match.bin.part")));
      assertEquals(CfrMatchFile.summary(target), new CfrMatchFile.Summary(7, 2, 2));
      List<CfrEpisode> restored = CfrMatchFile.read(target);
      assertEquals(restored.size(), 2);
      assertEpisode(restored.get(0), first);
      assertEpisode(restored.get(1), second);
      expectThrows(IOException.class, () -> writer.append(new CfrEpisodeFrame(second, 2, true)));
    } finally {
      clean(directory);
    }
  }

  @Test
  public void newFirstFrameReplacesPartialDataButNeverPublishedData() throws Exception {
    Path directory = Files.createTempDirectory("cfr-match-retry");
    Path target = directory.resolve("match.bin");
    try {
      new CfrMatchFile.Appender(target, 5).append(new CfrEpisodeFrame(episode(5, 3, .2), 0, false));
      CfrEpisode replacement = episode(5, 1, -.5);
      new CfrMatchFile.Appender(target, 5).append(new CfrEpisodeFrame(replacement, 0, true));
      assertEquals(CfrMatchFile.summary(target), new CfrMatchFile.Summary(5, 1, 1));
      assertEpisode(CfrMatchFile.read(target).getFirst(), replacement);
      byte[] published = Files.readAllBytes(target);
      expectThrows(
          IOException.class,
          () ->
              new CfrMatchFile.Appender(target, 5)
                  .append(new CfrEpisodeFrame(episode(5, 8, .1), 0, true)));
      assertEquals(Files.readAllBytes(target), published);
    } finally {
      clean(directory);
    }
  }

  @Test
  public void rejectsMissingFramesIdentityChangesAndRepeatedDecisionIds() throws Exception {
    Path directory = Files.createTempDirectory("cfr-match-order");
    Path target = directory.resolve("match.bin");
    try {
      var writer = new CfrMatchFile.Appender(target, 3);
      expectThrows(
          IOException.class, () -> writer.append(new CfrEpisodeFrame(episode(3, 1, 0), 1, false)));
      expectThrows(
          IOException.class, () -> writer.append(new CfrEpisodeFrame(episode(4, 1, 0), 0, false)));
      writer.append(new CfrEpisodeFrame(episode(3, 10, 0), 0, false));
      expectThrows(
          IOException.class, () -> writer.append(new CfrEpisodeFrame(episode(3, 10, 0), 1, true)));
      writer.append(new CfrEpisodeFrame(new CfrEpisode(3, List.of(), new double[4]), 1, false));
      writer.append(new CfrEpisodeFrame(episode(3, 11, 0), 2, true));
      assertEquals(CfrMatchFile.summary(target), new CfrMatchFile.Summary(3, 3, 2));
      assertEquals(CfrMatchFile.read(target).size(), 3);
    } finally {
      clean(directory);
    }
  }

  @Test
  public void summaryDoesNotDecodeObservationsButFullReadChecksCompression() throws Exception {
    Path directory = Files.createTempDirectory("cfr-match-checksum");
    Path target = directory.resolve("match.bin");
    try {
      new CfrMatchFile.Appender(target, 2).append(new CfrEpisodeFrame(episode(2, 1, .3), 0, true));
      byte[] bytes = Files.readAllBytes(target);
      long compressedBytes = ByteBuffer.wrap(bytes).getLong(16 + 12);
      bytes[Math.toIntExact(16 + 20 + compressedBytes - 1)] ^= 1;
      Files.write(target, bytes);
      assertEquals(CfrMatchFile.summary(target), new CfrMatchFile.Summary(2, 1, 1));
      expectThrows(IOException.class, () -> CfrMatchFile.read(target));
    } finally {
      clean(directory);
    }
  }

  @Test
  public void rejectsTruncationTrailingBytesFooterCorruptionAndFrameOrder() throws Exception {
    Path directory = Files.createTempDirectory("cfr-match-corrupt");
    Path target = directory.resolve("match.bin");
    try {
      new CfrMatchFile.Appender(target, 2).append(new CfrEpisodeFrame(episode(2, 1, .3), 0, true));
      byte[] valid = Files.readAllBytes(target);
      Files.write(target, java.util.Arrays.copyOf(valid, valid.length - 1));
      expectThrows(IOException.class, () -> CfrMatchFile.summary(target));
      Files.write(target, java.util.Arrays.copyOf(valid, valid.length + 1));
      expectThrows(IOException.class, () -> CfrMatchFile.read(target));
      byte[] corrupt = valid.clone();
      corrupt[corrupt.length - 1] ^= 1;
      Files.write(target, corrupt);
      expectThrows(IOException.class, () -> CfrMatchFile.summary(target));
      corrupt = valid.clone();
      ByteBuffer.wrap(corrupt).putInt(16 + 4, 1);
      Files.write(target, corrupt);
      expectThrows(IOException.class, () -> CfrMatchFile.read(target));
      corrupt = valid.clone();
      ByteBuffer.wrap(corrupt).putLong(16 + 12, Long.MAX_VALUE);
      Files.write(target, corrupt);
      expectThrows(IOException.class, () -> CfrMatchFile.read(target));
    } finally {
      clean(directory);
    }
  }

  @Test
  public void compressedMatchRoundTripsAndRejectsCorruptionAndTruncation() throws Exception {
    Path directory = Files.createTempDirectory("cfr-episode-zstd-");
    try {
      byte[] observation = new byte[1 << 20];
      var episode =
          new CfrEpisode(
              7,
              List.of(
                  new CfrDecision(
                      0,
                      1,
                      new int[] {4, 5},
                      1,
                      new double[] {1, 0},
                      .3,
                      ObservationPayload.copyOf(observation))),
              new double[] {1, 2, 3, 4});
      Path path = directory.resolve("episode.bin");
      new CfrMatchFile.Appender(path, episode.episodeId())
          .append(new CfrEpisodeFrame(episode, 0, true));
      assertTrue(Files.size(path) < observation.length / 100);
      CfrEpisode decoded = CfrMatchFile.read(path).getFirst();
      assertEquals(decoded.episodeId(), episode.episodeId());
      assertEquals(decoded.utilities(), episode.utilities());
      assertEquals(decoded.decisions().getFirst().selectedTargetProbability(), 0d);
      assertEquals(decoded.decisions().getFirst().observation().toByteArray(), observation);
      assertEquals(decoded.decisions().getFirst().behaviorProbability(), .3);
      byte[] bytes = Files.readAllBytes(path);
      byte[] corrupt = bytes.clone();
      corrupt[corrupt.length / 2] ^= 1;
      Files.write(path, corrupt);
      expectThrows(IOException.class, () -> CfrMatchFile.read(path).getFirst());
      Files.write(path, java.util.Arrays.copyOf(bytes, bytes.length - 5));
      expectThrows(IOException.class, () -> CfrMatchFile.read(path).getFirst());
    } finally {
      clean(directory);
    }
  }

  private static CfrEpisode episode(long gameId, long decisionId, double utility) {
    return new CfrEpisode(
        gameId,
        List.of(
            new CfrDecision(
                decisionId,
                0,
                new int[] {1, 2},
                0,
                new double[] {.4, .6},
                .5,
                ObservationPayload.copyOf(new byte[] {1, 3, 5}))),
        new double[] {utility, -utility, 0, 0},
        List.of(new CfrBranchOutcome(decisionId, .5, 1, .5, -.1, 0)));
  }

  private static void assertEpisode(CfrEpisode actual, CfrEpisode expected) {
    assertEquals(actual.episodeId(), expected.episodeId());
    assertEquals(actual.utilities(), expected.utilities());
    assertEquals(actual.branchOutcomes(), expected.branchOutcomes());
    assertEquals(actual.decisions().size(), expected.decisions().size());
    for (int index = 0; index < actual.decisions().size(); index++) {
      CfrDecision a = actual.decisions().get(index), b = expected.decisions().get(index);
      assertEquals(a.decisionId(), b.decisionId());
      assertEquals(a.seat(), b.seat());
      assertEquals(a.legalActionIds(), b.legalActionIds());
      assertEquals(a.targetPolicy(), b.targetPolicy());
      assertEquals(a.selectedSlot(), b.selectedSlot());
      assertEquals(a.behaviorProbability(), b.behaviorProbability());
      assertEquals(a.observation(), b.observation());
    }
  }

  private static void clean(Path directory) throws IOException {
    try (var files = Files.list(directory)) {
      for (Path file : files.toList()) Files.deleteIfExists(file);
    }
    Files.delete(directory);
  }
}

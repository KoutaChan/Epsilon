package com.epsilon.cfr.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrReplayTest {
  @Test
  public void immutableRowsAndChecksummedReplayRoundTrip() throws Exception {
    Path directory = Files.createTempDirectory("cfr-replay-codec-");
    try {
      byte[] observation = {4, 5};
      CfrTrainingRow row = row(1, 0, observation);
      observation[0] = 100;
      row.observation().toByteArray()[0] = 99;
      Assert.assertEquals(row.observation().toByteArray()[0], 4);
      Path snapshot;
      try (var replay = new CfrReplayStore(directory, "round-trip", 2, 2, 42, 43, 1024, 0, 10000)) {
        replay.appendEpisode(1, 1, List.of(row));
        snapshot = replay.save(1);
      }
      try (var replay = CfrReplayStore.loadGeneration(directory, "round-trip", 1)) {
        Assert.assertTrue(row.contentEquals(replay.regretSampler().rowAt(0)));
        Assert.assertTrue(row.contentEquals(replay.averageSampler().rowAt(0)));
      }
      byte[] bytes = Files.readAllBytes(snapshot);
      bytes[bytes.length / 2] ^= 1;
      Files.write(snapshot, bytes);
      Assert.expectThrows(
          IOException.class, () -> CfrReplayStore.loadGeneration(directory, "round-trip", 1));
    } finally {
      delete(directory);
    }
  }

  @Test
  public void priorityRetentionIsOrderIndependentAndApproximatelyUniformAcrossSeeds()
      throws Exception {
    var stream = new ArrayList<CfrTrainingRow>();
    for (int i = 0; i < 100; i++) stream.add(row(i, i % 9, new byte[i]));
    Path directory = Files.createTempDirectory("cfr-retention-");
    try (var first =
            new CfrReplayStore(
                directory.resolve("first"), "order", 10, 10, 42, 43, 1000, 0, 100000);
        var second =
            new CfrReplayStore(
                directory.resolve("second"), "order", 10, 10, 42, 43, 1000, 0, 100000)) {
      for (var row : stream) first.appendEpisode(1, row.id().episodeId(), List.of(row));
      Collections.shuffle(stream, new Random(23));
      for (var row : stream) second.appendEpisode(1, row.id().episodeId(), List.of(row));
      for (var row : stream)
        Assert.assertFalse(second.appendEpisode(1, row.id().episodeId(), List.of(row)));
      Assert.assertEquals(retainedIds(first.regretSampler()), retainedIds(second.regretSampler()));
      Assert.assertEquals(
          retainedIds(first.averageSampler()), retainedIds(second.averageSampler()));
      var expected =
          stream.stream()
              .map(CfrTrainingRow::id)
              .sorted(
                  (a, b) -> Long.compareUnsigned(a.retentionPriority(42), b.retentionPriority(42)))
              .limit(10)
              .sorted()
              .toList();
      Assert.assertEquals(retainedIds(first.regretSampler()), expected);
    } finally {
      delete(directory);
    }
    int[] inclusions = new int[100];
    for (int seed = 0; seed < 1000; seed++) {
      final int prioritySeed = seed;
      stream.stream()
          .map(CfrTrainingRow::id)
          .sorted(
              (a, b) ->
                  Long.compareUnsigned(
                      a.retentionPriority(prioritySeed), b.retentionPriority(prioritySeed)))
          .limit(10)
          .forEach(id -> inclusions[(int) id.episodeId()]++);
    }
    for (int count : inclusions)
      Assert.assertTrue(count > 60 && count < 140, "uniform inclusion count=" + count);
  }

  private static List<CfrSampleId> retainedIds(LogWeightedReplay sampler) {
    var ids = new ArrayList<CfrSampleId>(sampler.size());
    for (int index = 0; index < sampler.size(); index++) ids.add(sampler.rowAt(index).id());
    return List.copyOf(ids);
  }

  @Test
  public void weightedSamplingHandlesZerosLargeCommonOffsetsAndFrequency() {
    var rows =
        List.of(
            row(0, 0, new byte[0]),
            row(1, Math.log(3), new byte[0]),
            row(2, Double.NEGATIVE_INFINITY, new byte[0]));
    var sampler =
        new LogWeightedReplay(rows.size(), index -> rows.get(index).logRegretWeight(), rows::get);
    int[] counts = new int[3];
    Random random = new Random(883);
    for (int i = 0; i < 100_000; i++) counts[(int) sampler.sample(random).id().episodeId()]++;
    Assert.assertTrue(Math.abs(counts[0] - 25_000) < 600);
    Assert.assertEquals(counts[2], 0);
    Assert.assertEquals(sampler.metrics().effectiveSampleSize(), 1.6, 1e-12);
    Assert.assertEquals(sampler.metrics().maximumReplayProbability(), .75, 1e-12);
    var largeWeightRows = List.of(row(0, 1e308, new byte[0]), row(1, 1e308, new byte[0]));
    var huge =
        new LogWeightedReplay(
            largeWeightRows.size(),
            index -> largeWeightRows.get(index).logRegretWeight(),
            largeWeightRows::get);
    Assert.assertEquals(huge.metrics().effectiveSampleSize(), 2, 0);
    int first = 0;
    for (int i = 0; i < 10000; i++) if (huge.sample(random).id().episodeId() == 0) first++;
    Assert.assertTrue(first > 4800 && first < 5200);
    var empty =
        new LogWeightedReplay(1, index -> rows.get(2).logRegretWeight(), index -> rows.get(2));
    Assert.expectThrows(IllegalStateException.class, () -> empty.sample(random));
  }

  @Test
  public void durableReplayRestoresExactGenerationAndIdempotentInsertionWithBoundedCache()
      throws Exception {
    Path directory = Files.createTempDirectory("cfr-replay-");
    try {
      try (var replay =
          new CfrReplayStore(directory, "solve-a", 8, 5, 42, 43, 2048, 1024, 1_000_000)) {
        for (int episode = 0; episode < 50; episode++) {
          var rows = List.of(row(episode, 0, new byte[512]));
          Assert.assertTrue(replay.appendEpisode(1, episode, rows));
          Assert.assertFalse(replay.appendEpisode(1, episode, rows));
        }
        replay.save(1);
        var before = replay.regretSampler().sample(100, new Random(21));
        Assert.assertTrue(replay.stats().cachedBytes() <= 1024);
        Assert.assertEquals(replay.stats().seenRows(), 50);
        replay.appendEpisode(1, 50, List.of(row(50, 0, new byte[512])));
        replay.save(2);
        try (var restored = CfrReplayStore.loadGeneration(directory, "solve-a", 1)) {
          Assert.assertEquals(
              ids(restored.regretSampler().sample(100, new Random(21))), ids(before));
          Assert.assertFalse(restored.containsEpisode(1, 50));
          Assert.assertFalse(restored.appendEpisode(1, 1, List.of(row(1, 0, new byte[512]))));
          Assert.expectThrows(
              IllegalArgumentException.class,
              () -> restored.appendEpisode(1, 1, List.of(row(1, 2, new byte[512]))));
        }
        try (var latest = CfrReplayStore.load(directory, "solve-a")) {
          Assert.assertTrue(latest.containsEpisode(1, 50));
          Assert.assertEquals(latest.stats().regretRows(), 8);
          Assert.assertEquals(latest.stats().averageRows(), 5);
        }
        Assert.expectThrows(IOException.class, () -> CfrReplayStore.load(directory, "wrong-solve"));
        Assert.expectThrows(IOException.class, () -> replay.save(1));
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void storageLimitRejectsWithoutChangingRetentionAndCompactionHonorsSavedGenerations()
      throws Exception {
    Path directory = Files.createTempDirectory("cfr-budget-");
    try {
      long budget = Math.max(100, CfrPayloadStore.maximumCompressedBytes(row(0, 0, new byte[0])));
      try (var replay = new CfrReplayStore(directory, "solve", 10, 10, 2, 3, 100, 0, budget)) {
        replay.appendEpisode(1, 0, List.of(row(0, 0, new byte[0])));
        Assert.expectThrows(
            IOException.class, () -> replay.appendEpisode(1, 1, List.of(row(1, 0, new byte[0]))));
        Assert.assertEquals(replay.stats().seenRows(), 1);
        Assert.assertFalse(replay.containsEpisode(1, 1));
      }
    } finally {
      delete(directory);
    }
    directory = Files.createTempDirectory("cfr-gc-");
    try {
      try (var replay = new CfrReplayStore(directory, "solve", 3, 4, 2, 3, 4096, 0, 10_000_000)) {
        for (int i = 0; i < 100; i++)
          replay.appendEpisode(1, i, List.of(row(i, 0, new byte[1024])));
        Path oldSnapshot = replay.save(1);
        long initial = replay.storedBytes();
        var pinned = replay.regretSampler();
        replay.compact();
        replay.save(2);
        Assert.assertEquals(replay.collectGarbage(), 0);
        Files.delete(oldSnapshot);
        Assert.assertEquals(replay.collectGarbage(), 0);
        Assert.assertNotNull(pinned.sample(new Random(1)));
        replay.releaseSamplers();
        Assert.assertTrue(replay.collectGarbage() >= initial);
        Assert.assertEquals(replay.storedBytes(), replay.stats().livePayloadBytes());
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void compactionCopiesFramesUsingExactCompressedHeadroom() throws Exception {
    Path directory = Files.createTempDirectory("cfr-compaction-budget-");
    try {
      CfrTrainingRow sample = row(0, 0, new byte[512]);
      long budget = CfrPayloadStore.maximumCompressedBytes(sample);
      try (var replay =
          new CfrReplayStore(
              directory, "solve", 2, 2, 5, 6, sample.serializedBytes(), 2048, budget)) {
        replay.appendEpisode(1, 0, List.of(sample));
        replay.save(1);
        long before = replay.storedBytes();
        Assert.assertTrue(before * 2 < budget);
        Path original;
        try (var files = Files.list(directory.resolve("payloads"))) {
          original = files.findFirst().orElseThrow();
        }
        replay.compact();
        Assert.assertEquals(replay.storedBytes(), before * 2);
        Assert.assertEquals(replay.verifyPayloads(), 1);
        replay.save(2);
        try (var files = Files.list(directory.resolve("payloads"))) {
          Path copy = files.filter(path -> !path.equals(original)).findFirst().orElseThrow();
          Assert.assertEquals(Files.readAllBytes(copy), Files.readAllBytes(original));
        }
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void normalPayloadReadsSkipShaWhileExplicitVerificationBypassesTheCache()
      throws Exception {
    Path directory = Files.createTempDirectory("cfr-payload-verify-");
    try {
      try (var payloads = new CfrPayloadStore(directory, 2048)) {
        CfrTrainingRow sample = row(0, 0, new byte[512]);
        CfrPayloadStore.Ref reference = payloads.append(sample);
        payloads.seal();
        String incorrectHash =
            (reference.sha256().charAt(0) == '0' ? "1" : "0") + reference.sha256().substring(1);
        var incorrect =
            new CfrPayloadStore.Ref(
                reference.file(), reference.offset(), reference.length(), incorrectHash);
        Assert.assertTrue(sample.contentEquals(payloads.read(incorrect)));
        Assert.assertTrue(payloads.cachedBytes() > 0);
        Assert.expectThrows(IOException.class, () -> payloads.readVerified(incorrect));
        Assert.assertTrue(sample.contentEquals(payloads.readVerified(reference)));
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void envelopeChecksumCoversSkippedDataLargerThanTheReadBuffer() throws Exception {
    Path directory = Files.createTempDirectory("cfr-envelope-skip-");
    try {
      Path file = directory.resolve("envelope.bin");
      byte[] skipped = new byte[200_000];
      new Random(21).nextBytes(skipped);
      CfrReplayCodec.writeEnvelope(
          file,
          CfrReplayCodec.REPLAY,
          out -> {
            out.writeInt(skipped.length);
            out.write(skipped);
            out.writeLong(123);
          });
      CfrReplayCodec.Reader<Long> reader =
          in -> {
            in.skipNBytes(in.readInt());
            return in.readLong();
          };
      Assert.assertEquals(
          CfrReplayCodec.readEnvelope(file, CfrReplayCodec.REPLAY, reader).longValue(), 123L);
      byte[] bytes = Files.readAllBytes(file);
      bytes[100_000] ^= 1;
      Files.write(file, bytes);
      Assert.expectThrows(
          IOException.class,
          () -> CfrReplayCodec.readEnvelope(file, CfrReplayCodec.REPLAY, reader));
    } finally {
      delete(directory);
    }
  }

  @Test
  public void explicitVerificationDetectsPayloadCorruptionAndLoadingRejectsMissingSegments()
      throws Exception {
    Path directory = Files.createTempDirectory("cfr-corrupt-");
    try {
      try (var replay = new CfrReplayStore(directory, "solve", 2, 2, 5, 6, 2048, 0, 10000)) {
        replay.appendEpisode(1, 0, List.of(row(0, 0, new byte[512])));
        replay.save(1);
      }
      Path segment;
      try (var files = Files.list(directory.resolve("payloads"))) {
        segment = files.findFirst().orElseThrow();
      }
      byte[] bytes = Files.readAllBytes(segment);
      bytes[bytes.length / 2] ^= 1;
      Files.write(segment, bytes);
      try (var replay = CfrReplayStore.loadGeneration(directory, "solve", 1)) {
        Assert.expectThrows(IOException.class, replay::verifyPayloads);
      }
      Files.delete(segment);
      Assert.expectThrows(
          IOException.class, () -> CfrReplayStore.loadGeneration(directory, "solve", 1));
    } finally {
      delete(directory);
    }
  }

  private static CfrTrainingRow row(long episode, double logWeight, byte[] observation) {
    return new CfrTrainingRow(
        new CfrSampleId(1, episode, 0, 0),
        new int[] {0, 1},
        0,
        new double[] {.5, .5},
        1,
        logWeight,
        logWeight,
        ObservationPayload.copyOf(observation));
  }

  private static List<CfrSampleId> ids(List<CfrTrainingRow> rows) {
    return rows.stream().map(CfrTrainingRow::id).toList();
  }

  private static void delete(Path directory) throws IOException {
    try (var files = Files.walk(directory)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);
    }
  }
}

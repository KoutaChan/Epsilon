package com.epsilon.prism.luck;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.core.GameState;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 標本の書き出しと読み出し。
 *
 * <p>境界で復元した出来事を捕捉（硬貨・O°・層・重み）し、局末に Φ を付けて shard に書き、封印した目録から読み戻すと、標本が書いた順に
 * ビットで同じになる（Y は行為者から見た相対席の順、標本は複数の枠にまたがる）。枠の CRC と圧縮の本体の破れ、指紋の違う目録、封印して
 * いないコホートは読まない。
 */
public class LuckSampleStoreTest {

  /** 最初の半荘の種。 */
  private static final long SEED = 31_000;

  /** コホートの番号。 */
  private static final int ITERATION = 3;

  @Test(timeOut = 300_000)
  public void samplesRoundTripThroughTheStore() throws IOException {
    Path cohort = Files.createTempDirectory("luck-store");
    List<LuckSample> written = collect(cohort, new LuckSampling(17L, 0.75), 14);
    assertTrue(written.size() > 5_000, "samples " + written.size());
    LuckSampleStore store = LuckSampleStore.open(cohort);
    assertEquals(store.samples(), written.size());
    assertEquals(store.manifest().iteration(), ITERATION);
    try (LuckSampleStore.Cursor cursor = store.cursor()) {
      for (LuckSample sample : written) {
        assertTrue(cursor.next());
        assertEquals(cursor.sample(), sample);
      }
      assertFalse(cursor.next());
    }
  }

  @Test(timeOut = 300_000)
  public void rejectsBrokenFramesAndForeignManifests() throws IOException {
    Path cohort = Files.createTempDirectory("luck-broken");
    List<LuckSample> written = collect(cohort, new LuckSampling(5L, 1.0), 1);
    Path directory = LuckSampleWriter.directory(cohort);
    Path shard =
        directory.resolve(LuckSampleStore.open(cohort).manifest().shards().getFirst().file());
    byte[] original = Files.readAllBytes(shard);

    byte[] crc = original.clone();
    crc[2 * Integer.BYTES] ^= 1;
    Files.write(shard, crc);
    IOException mismatch = expectThrows(IOException.class, () -> readAll(cohort));
    assertTrue(mismatch.getMessage().contains("CRC"), mismatch.getMessage());

    byte[] body = original.clone();
    int compressed = ByteBuffer.wrap(original).order(ByteOrder.LITTLE_ENDIAN).getInt(0);
    body[LuckSampleFormat.FRAME_HEADER + compressed / 2] ^= 0x5a;
    Files.write(shard, body);
    expectThrows(IOException.class, () -> readAll(cohort));

    Files.write(shard, original);
    assertEquals(readAll(cohort), written.size());

    Path manifest = directory.resolve(LuckSampleWriter.MANIFEST);
    Files.writeString(
        manifest,
        Files.readString(manifest).replace(LuckSampleFormat.fingerprint(), "0000000000000000"));
    expectThrows(IOException.class, () -> LuckSampleStore.open(cohort));

    Path unsealed = Files.createTempDirectory("luck-unsealed");
    try (LuckSampleWriter writer = new LuckSampleWriter(unsealed, ITERATION)) {
      writer.append(written.getFirst());
    }
    expectThrows(IOException.class, () -> LuckSampleStore.open(unsealed));
  }

  /** 半荘を打って主経路の出来事を捕捉し、局末に仮の Φ を付けて書き、書いた標本を書いた順に返す。 */
  private static List<LuckSample> collect(Path cohort, LuckSampling sampling, int games)
      throws IOException {
    List<LuckSample> written = new ArrayList<>();
    LuckCapture capture = new LuckCapture(sampling);
    LuckObservation read = new LuckObservation();
    LuckStratum stratum = new LuckStratum();
    try (LuckSampleWriter writer = new LuckSampleWriter(cohort, ITERATION)) {
      LuckGames.play(
          SEED,
          games,
          null,
          new LuckGames.Visitor() {
            @Override
            public LuckSink sink() {
              return capture;
            }

            @Override
            public void roundStarted(int game, int round) {
              capture.root(game, LuckSampling.MAIN_PATH, LuckSampling.handRoot(round));
            }

            @Override
            public void roundEnded(int game, int round) {
              LuckBatch batch = capture.detach();
              if (batch == null) {
                return;
              }
              double[] phi = {round + 0.25, -game - 0.5, 1.75, -round - 1.5};
              long[] words = batch.arena().words();
              for (int sample = 0; sample < batch.size(); sample++) {
                int base = batch.base(sample);
                long[] observation =
                    Arrays.copyOfRange(words, base, base + LuckSchema.longs(words, base));
                read.bind(observation, 0);
                assertEquals(batch.stratum(sample), stratum.of(read));
                assertEquals(batch.weight(sample), sampling.weight(), 0f);
                float[] y = new float[GameState.NUM_PLAYERS];
                for (int seat = 0; seat < y.length; seat++) {
                  y[seat] = (float) phi[read.absoluteSeat(seat)];
                }
                written.add(
                    new LuckSample(
                        game, observation, y, batch.stratum(sample), batch.weight(sample)));
              }
              try {
                batch.settle(phi, writer);
              } catch (IOException failure) {
                throw new UncheckedIOException(failure);
              }
            }
          });
      writer.seal();
    }
    return written;
  }

  private static long readAll(Path cohort) throws IOException {
    long count = 0;
    try (LuckSampleStore.Cursor cursor = LuckSampleStore.open(cohort).cursor()) {
      while (cursor.next()) {
        count++;
      }
    }
    return count;
  }
}

package com.epsilon.prism.rows;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.auxiliary.OracleCapture;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationSchema;
import com.epsilon.prism.observe.ObservationWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.testng.annotations.Test;

/**
 * 行の直列化とコホートの書き手。
 *
 * <p>GREEDY の固定の種の判断の記録 O と神託に、値・p0・候補集合・根の欄・帳簿・局末の記録を乱数で入れた行が、フレームを通って全ての欄で元に
 * 戻る（NaN の位置と ±0 も）。CRC の破れを例外で捕まえ、書けない行（帳簿の欄の数の違い、fp16 で表せない帳簿）を書く前に止める。書き手は
 * 目録に記録の指紋と帳簿の欄の数を書き、索引の順にフレームを読むと元の行に戻る。指紋の違う目録は読む前に止める。
 */
public class RowCodecTest {

  /** 帳簿の欄の数。 */
  private static final int FEATURES = 17;

  @Test
  public void framesRoundTrip() throws IOException {
    RowCodec codec = new RowCodec(FEATURES);
    for (Row row : rows(300)) {
      ByteBuffer frame = ByteBuffer.wrap(codec.encode(row));
      assertRow(codec.decode(frame), row);
      assertFalse(frame.hasRemaining());
    }
  }

  @Test
  public void brokenFramesAndUnwritableRowsAreRejected() {
    RowCodec codec = new RowCodec(FEATURES);
    Row row = rows(1).get(0);
    byte[] frame = codec.encode(row);

    // 索引の組（頭の最後から 4 バイト目）を A と B で入れ替える。索引としては正しい値なので CRC だけが捕まえる。
    byte[] group = frame.clone();
    group[RowCodec.FRAME_HEADER - 4] ^= 1;
    IOException mismatch =
        expectThrows(IOException.class, () -> codec.decode(ByteBuffer.wrap(group)));
    assertTrue(mismatch.getMessage().contains("CRC"), mismatch.getMessage());
    byte[] body = frame.clone();
    body[body.length - 1] ^= 0x5A;
    expectThrows(IOException.class, () -> codec.decode(ByteBuffer.wrap(body)));

    expectThrows(IllegalArgumentException.class, () -> new RowCodec(FEATURES + 1).encode(row));
    float[] ledger = row.ledger().clone();
    ledger[row.played() * FEATURES] = 0.1f;
    expectThrows(IllegalArgumentException.class, () -> codec.encode(withLedger(row, ledger)));
  }

  @Test
  public void cohortWriterRecordsFingerprintsAndFrames() throws IOException {
    Path directory = Files.createTempDirectory("prism-rows");
    List<Row> rows = rows(120);
    try (CohortWriter writer = new CohortWriter(directory, 3, FEATURES, 2)) {
      for (Row row : rows) {
        writer.accept(row);
      }
      writer.seal();
    }
    CohortWriter.Manifest manifest = CohortWriter.readManifest(directory);
    assertEquals(manifest.iteration(), 3);
    assertEquals(manifest.rows(), rows.size());
    assertEquals(manifest.ledgerFeatures(), FEATURES);
    assertEquals(manifest.fingerprints(), CohortWriter.Fingerprints.current());

    Map<Long, Row> written = new HashMap<>();
    for (Row row : rows) {
      written.put(row.decisionId(), row);
    }
    RowCodec codec = new RowCodec(manifest.ledgerFeatures());
    int read = 0;
    long bytes = 0;
    for (CohortWriter.Shard shard : manifest.shards()) {
      Path file = directory.resolve(shard.file());
      ByteBuffer frames = ByteBuffer.wrap(Files.readAllBytes(file));
      ByteBuffer index = ByteBuffer.wrap(Files.readAllBytes(CohortWriter.index(file)));
      index.order(ByteOrder.LITTLE_ENDIAN);
      while (index.hasRemaining()) {
        int length = index.getInt();
        RowCodec.Index entry =
            RowCodec.index(length, index.get(), index.get(), index.get(), index.get(), index.get());
        int start = frames.position();
        Row row = codec.decode(frames);
        assertEquals(frames.position() - start, length);
        assertEquals(entry, RowCodec.index(row, length));
        assertRow(row, written.get(row.decisionId()));
        read++;
      }
      assertFalse(frames.hasRemaining());
      bytes += shard.bytes();
    }
    assertEquals(read, rows.size());
    assertEquals(manifest.bytes(), bytes);

    Path path = directory.resolve(CohortWriter.MANIFEST);
    Files.writeString(
        path, Files.readString(path).replace(OutcomeCodec.fingerprint(), "0000000000000000"));
    expectThrows(IOException.class, () -> CohortWriter.readManifest(directory));
  }

  /** GREEDY の固定の種の判断ごとに、O と神託を実際に捕まえ、ほかの欄を乱数で入れた行（神託は 5 行に 1 行なし）。 */
  private static List<Row> rows(int count) {
    List<Row> rows = new ArrayList<>();
    ObservationWriter writer = new ObservationWriter();
    ObservationArena arena = new ObservationArena(256);
    OracleCapture oracles = new OracleCapture();
    SplittableRandom random = new SplittableRandom(7);
    new DecisionWalk(DecisionWalk.Policy.GREEDY)
        .run(
            500,
            3,
            (engine, point, round) -> {
              if (rows.size() == count) {
                return;
              }
              int seat = point.player();
              arena.clear();
              int base =
                  writer.capture(
                      engine.observation(seat), point.kind(), point.legalActions(), arena);
              long[] words = arena.words();
              long[] observation =
                  Arrays.copyOfRange(words, base, base + ObservationSchema.longs(words, base));
              long[] oracle =
                  rows.size() % 5 == 0 ? new long[0] : oracles.capture(engine.getState(), seat);
              rows.add(row(rows.size(), point.legalActions().size(), observation, oracle, random));
            });
    assertEquals(rows.size(), count);
    return rows;
  }

  /** 行の番号 {@code id} の行。行の種類・類・用途・組は番号で回し、測った手は打った手と乱数で選んだ手。 */
  private static Row row(
      int id, int actions, long[] observation, long[] oracle, SplittableRandom random) {
    Row.Source source = Row.Source.values()[id % Row.Source.values().length];
    int played = random.nextInt(actions);
    float[] values = new float[actions];
    float[] prior = new float[actions];
    float[] ledger = new float[actions * FEATURES];
    long[] outcomes = new long[actions * OutcomeCodec.WORDS];
    Arrays.fill(values, Float.NaN);
    Arrays.fill(ledger, Float.NaN);
    int support = 0;
    for (int action = 0; action < actions; action++) {
      prior[action] = random.nextFloat();
      if (action != played && random.nextBoolean()) {
        continue;
      }
      support |= 1 << action;
      values[action] = action == played && id % 7 == 0 ? -0f : random.nextFloat() * 2 - 1;
      for (int feature = 0; feature < FEATURES; feature++) {
        float term = (float) random.nextGaussian() * (feature == 3 ? 20 : 1);
        ledger[action * FEATURES + feature] =
            feature == 5 ? -0f : Float.float16ToFloat(Float.floatToFloat16(term));
      }
      if (random.nextInt(4) != 0) {
        for (int word = 0; word < OutcomeCodec.WORDS; word++) {
          outcomes[action * OutcomeCodec.WORDS + word] = random.nextLong();
        }
      }
    }
    boolean root = source == Row.Source.ROOT;
    return new Row(
        id,
        id * 31L + 7,
        DecisionClass.of(id % DecisionClass.COUNT),
        id % 5 == 0 ? Row.Purpose.HOLDOUT : Row.Purpose.TRAIN,
        (id / 2 & 1) == 0 ? Row.Group.A : Row.Group.B,
        source,
        played,
        values,
        prior,
        support,
        root ? played : -1,
        root ? 0.25f : Float.NaN,
        root ? 0.0075f : random.nextFloat(),
        source == Row.Source.ROLLIN ? id * 7L : -1,
        ledger,
        outcomes,
        oracle,
        observation);
  }

  private static Row withLedger(Row row, float[] ledger) {
    return new Row(
        row.gameId(),
        row.decisionId(),
        row.decisionClass(),
        row.purpose(),
        row.group(),
        row.source(),
        row.played(),
        row.values(),
        row.prior(),
        row.support(),
        row.deviation(),
        row.deviationProbability(),
        row.propensity(),
        row.rootDecisionId(),
        ledger,
        row.outcomes(),
        row.oracle(),
        row.observation());
  }

  /** 全ての欄が同じ（浮動小数はビットで比べるので、NaN の位置と ±0 も同じ）。 */
  private static void assertRow(Row actual, Row expected) {
    String where = "row " + expected.gameId();
    assertEquals(actual.gameId(), expected.gameId(), where);
    assertEquals(actual.decisionId(), expected.decisionId(), where);
    assertEquals(actual.decisionClass(), expected.decisionClass(), where);
    assertEquals(actual.purpose(), expected.purpose(), where);
    assertEquals(actual.group(), expected.group(), where);
    assertEquals(actual.source(), expected.source(), where);
    assertEquals(actual.played(), expected.played(), where);
    assertTrue(Arrays.equals(actual.values(), expected.values()), where);
    assertTrue(Arrays.equals(actual.prior(), expected.prior()), where);
    assertEquals(actual.support(), expected.support(), where);
    assertEquals(actual.deviation(), expected.deviation(), where);
    assertEquals(
        Float.floatToRawIntBits(actual.deviationProbability()),
        Float.floatToRawIntBits(expected.deviationProbability()),
        where);
    assertEquals(
        Float.floatToRawIntBits(actual.propensity()),
        Float.floatToRawIntBits(expected.propensity()),
        where);
    assertEquals(actual.rootDecisionId(), expected.rootDecisionId(), where);
    assertTrue(Arrays.equals(actual.ledger(), expected.ledger()), where);
    assertTrue(Arrays.equals(actual.outcomes(), expected.outcomes()), where);
    assertTrue(Arrays.equals(actual.oracle(), expected.oracle()), where);
    assertTrue(Arrays.equals(actual.observation(), expected.observation()), where);
  }
}

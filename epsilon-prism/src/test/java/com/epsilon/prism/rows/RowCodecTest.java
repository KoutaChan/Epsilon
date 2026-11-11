package com.epsilon.prism.rows;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.auxiliary.OracleCapture;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
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
 * 行の直列化と、コホートのライターのテスト。
 *
 * <p>テストの行は、GREEDY で乱数の種を固定して打った判断の記録 O と Oracle に、乱数で作った値・p0・候補集合・根のフィールド・帳簿・
 * 局末の記録を組み合わせたもの。確かめるのは次のこと。
 *
 * <ul>
 *   <li>行をフレームにして読み戻すと、全てのフィールドが元に戻る（NaN の位置と ±0 も）。
 *   <li>CRC が合わないフレームは例外で検出する。書けない行（帳簿の特徴の数が違う行、fp16 で表せない帳簿）は書く前に止める。
 *   <li>ライターはマニフェストに記録のフィンガープリントと帳簿の特徴の数を書く。インデックスの順にフレームを読むと元の行に戻る。
 *   <li>フィンガープリントの違うマニフェストは、読む前に止める。
 * </ul>
 */
public class RowCodecTest {

  /** 帳簿の特徴の数。 */
  private static final int FEATURES = 17;

  @Test
  public void framesRoundTrip() throws IOException {
    RowCodec codec = new RowCodec(FEATURES, TENHOU_4P);
    for (Row row : rows(300)) {
      ByteBuffer frame = ByteBuffer.wrap(codec.encode(row));
      assertRow(codec.decode(frame), row);
      assertFalse(frame.hasRemaining());
    }
  }

  @Test
  public void brokenFramesAndUnwritableRowsAreRejected() {
    RowCodec codec = new RowCodec(FEATURES, TENHOU_4P);
    Row row = rows(1).get(0);
    byte[] frame = codec.encode(row);

    // インデックスにも入る半荘の用途（ヘッダーの末尾から 4 バイト目）を、学習とホールドアウトで入れ替える。インデックスとしては正しい値なので、
    // CRC でしか検出できない。
    byte[] purpose = frame.clone();
    purpose[RowCodec.FRAME_HEADER - 4] ^= 1;
    IOException mismatch =
        expectThrows(IOException.class, () -> codec.decode(ByteBuffer.wrap(purpose)));
    assertTrue(mismatch.getMessage().contains("CRC"), mismatch.getMessage());
    byte[] body = frame.clone();
    body[body.length - 1] ^= 0x5A;
    expectThrows(IOException.class, () -> codec.decode(ByteBuffer.wrap(body)));

    expectThrows(IllegalArgumentException.class, () -> new RowCodec(FEATURES + 1, TENHOU_4P).encode(row));
    float[] ledger = row.ledger().clone();
    ledger[row.played() * FEATURES] = 0.1f;
    expectThrows(IllegalArgumentException.class, () -> codec.encode(withLedger(row, ledger)));
  }

  @Test
  public void cohortWriterRecordsFingerprintsAndFrames() throws IOException {
    Path directory = Files.createTempDirectory("prism-rows");
    List<Row> rows = rows(120);
    try (CohortWriter writer = new CohortWriter(directory, TENHOU_4P, 3, FEATURES, 2, CohortWriter.Tap.NONE)) {
      for (Row row : rows) {
        writer.accept(row);
      }
      writer.seal();
    }
    CohortWriter.Manifest manifest = CohortWriter.readManifest(directory);
    assertEquals(manifest.iteration(), 3);
    assertEquals(manifest.rows(), rows.size());
    assertEquals(manifest.ledgerFeatures(), FEATURES);
    assertEquals(manifest.fingerprints(), CohortWriter.Fingerprints.of(TENHOU_4P));
    assertEquals(
        manifest.fingerprints(),
        new CohortWriter.Fingerprints("bb3db98cb8b5b43a", "a71ba8e0c3e8c105", "77275bf2f7b5ae35"));

    Map<Long, Row> written = new HashMap<>();
    for (Row row : rows) {
      written.put(row.decisionId(), row);
    }
    RowCodec codec = new RowCodec(manifest.ledgerFeatures(), TENHOU_4P);
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
            RowCodec.index(length, index.get(), index.get(), index.get(), index.get());
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
        path, Files.readString(path).replace(OutcomeCodec.of(TENHOU_4P).fingerprint(), "0000000000000000"));
    expectThrows(IOException.class, () -> CohortWriter.readManifest(directory));

    // O のフィンガープリントは四人打ちなのに、局末の記録のフィンガープリントが三人打ちのマニフェストは、ルールが食い違うので読む前に止める
    Files.writeString(
        path,
        Files.readString(path)
            .replace("0000000000000000", OutcomeCodec.of(Variant.TENHOU_3P).fingerprint()));
    expectThrows(IOException.class, () -> CohortWriter.readManifest(directory));
  }

  /** GREEDY で乱数の種を固定して打ち、判断ごとに行を 1 つ作る。O と Oracle は実際に取り込み、ほかのフィールドは乱数で入れる（5 行に 1 行は Oracle なし）。 */
  private static List<Row> rows(int count) {
    List<Row> rows = new ArrayList<>();
    ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
    RuleWorkspace ws = new RuleWorkspace(TENHOU_4P);
    SplittableRandom random = new SplittableRandom(7);
    new TableWalk(TENHOU_4P, TableWalk.Policy.GREEDY)
        .run(
            500,
            3,
            (engine, slot) -> {
              if (rows.size() == count) {
                return;
              }
              Decisions decisions = engine.decisions();
              int seat = decisions.seat(slot);
              batch.clear();
              int index = writer.capture(engine.view(seat), decisions, slot, batch);
              long[] observation =
                  Arrays.copyOfRange(batch.words(), batch.start(index), batch.start(index) + batch.longs(index));
              long[] oracle =
                  rows.size() % 5 == 0 ? new long[0] : OracleCapture.capture(engine.full(), seat, ws);
              rows.add(row(rows.size(), decisions.legalCount(slot), observation, oracle, random));
            });
    assertEquals(rows.size(), count);
    return rows;
  }

  /** 番号 {@code id} の行を作る。行の種類・判断のクラス・用途は番号で順に回す。値を測った手は、打った手と乱数で選んだ手にする。 */
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

  /** 全てのフィールドが同じことを確かめる（浮動小数点数はビットで比べるので、NaN の位置と ±0 も同じ）。 */
  private static void assertRow(Row actual, Row expected) {
    String where = "row " + expected.gameId();
    assertEquals(actual.gameId(), expected.gameId(), where);
    assertEquals(actual.decisionId(), expected.decisionId(), where);
    assertEquals(actual.decisionClass(), expected.decisionClass(), where);
    assertEquals(actual.purpose(), expected.purpose(), where);
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

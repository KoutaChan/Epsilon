package com.epsilon.prism.rows;

import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.observe.Observation;
import com.github.luben.zstd.Zstd;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.zip.CRC32C;

/**
 * 行（{@link Row}）の直列化。1 行を 1 フレームにし、行ごとに zstd（level 1）で圧縮して CRC32C を掛ける。行ごとなので、学習側は
 * 索引から任意の行を独立に展開できる。
 *
 * <pre>
 * frame = [i32 本体の圧縮後の長さ][i32 展開後の長さ][i32 CRC32C(索引 ‖ 展開後の本体)][索引][zstd 本体]
 * 索引   = [u8 purpose][u8 group][u8 source][u8 class][u8 measured]
 *          （{@link Index}。行を選ぶ値で、本体には重ねない）
 * 本体   = [u16 O の long の数][i64 × O]
 *          [i64 gameId][i64 decisionId][u8 played]
 *          [f32 values × A][f32 prior × A][i32 support]
 *          [i8 deviation][f32 deviationProbability][f32 propensity][i64 rootDecisionId]
 *          測った手（位置の昇順）ごとに [f16 帳簿 × F][i64 局末の記録 × {@link OutcomeCodec#WORDS}]
 *          [u8 神託の long の数][i64 × 神託]
 * </pre>
 *
 * <p>A（合法手の数）は O が持ち、F（帳簿の欄の数）はコホートの目録が持つので、本体には書かない。列挙は番号で、数は little endian で書く。
 * 帳簿は fp16 で厳密に表せる値だけを書き（収集が確率的丸めで作る値）、表せない値は例外にする。読むときは CRC を照らしてから、O の長さ、
 * 測った手の数（索引）、神託の長さを照らす。
 *
 * <p>O を読んで合法手の数を出すので、書くスレッド・読むスレッドごとに 1 つ持つ。
 */
public final class RowCodec {

  private static final int INDEX_BYTES = 5;

  /** フレームの頭の索引の位置（長さ 2 つと CRC の後）。 */
  private static final int INDEX_OFFSET = 3 * Integer.BYTES;

  /** フレームの頭の CRC の位置。 */
  private static final int CRC_OFFSET = 2 * Integer.BYTES;

  /** フレームの頭のバイト数。 */
  public static final int FRAME_HEADER = INDEX_OFFSET + INDEX_BYTES;

  private static final int MAX_LEDGER_FEATURES = 64;

  /** 展開後の本体の長さの上限（壊れた頭で大きな配列を作らない）。 */
  private static final int MAX_RAW = 1 << 20;

  private static final int LEVEL = 1;

  private static final Row.Purpose[] PURPOSES = Row.Purpose.values();
  private static final Row.Group[] GROUPS = Row.Group.values();
  private static final Row.Source[] SOURCES = Row.Source.values();

  private final int ledgerFeatures;
  private final Observation observation = new Observation();

  /**
   * 本体を展開せずに読める行の索引（行を選ぶのに要る値。フレームの頭の索引と、シャードの索引のファイル {@link CohortWriter#index} の
   * 1 件が持つ）。
   *
   * @param frameLength フレーム全体の長さ
   * @param purpose 半荘の用途
   * @param group 半荘の組
   * @param source 行の種類
   * @param decisionClass 判断の類
   * @param measured 測った手（values が NaN でない手）の数
   */
  public record Index(
      int frameLength,
      Row.Purpose purpose,
      Row.Group group,
      Row.Source source,
      DecisionClass decisionClass,
      int measured) {}

  /**
   * 直列化を作る。
   *
   * @param ledgerFeatures 帳簿の欄の数（コホートの目録の値）
   */
  public RowCodec(int ledgerFeatures) {
    if (ledgerFeatures < 0 || ledgerFeatures > MAX_LEDGER_FEATURES) {
      throw new IllegalArgumentException("ledger features out of range: " + ledgerFeatures);
    }
    this.ledgerFeatures = ledgerFeatures;
  }

  public int ledgerFeatures() {
    return ledgerFeatures;
  }

  /**
   * 行の索引を作る。
   *
   * @param row 行
   * @param frameLength フレーム全体の長さ
   * @return 索引
   */
  public static Index index(Row row, int frameLength) {
    return new Index(
        frameLength,
        row.purpose(),
        row.group(),
        row.source(),
        row.decisionClass(),
        row.measuredCount());
  }

  /**
   * 索引の 5 バイト（purpose・group・source・class・measured）から索引を作る。
   *
   * @param frameLength フレーム全体の長さ
   * @return 索引
   * @throws IOException フレームの長さが頭より短いか、列挙の番号が範囲の外のとき
   */
  public static Index index(
      int frameLength, byte purpose, byte group, byte source, byte decisionClass, byte measured)
      throws IOException {
    if (frameLength < FRAME_HEADER
        || purpose < 0
        || purpose >= PURPOSES.length
        || group < 0
        || group >= GROUPS.length
        || source < 0
        || source >= SOURCES.length
        || decisionClass < 0
        || decisionClass >= DecisionClass.COUNT) {
      throw new IOException("Corrupt row index");
    }
    return new Index(
        frameLength,
        PURPOSES[purpose],
        GROUPS[group],
        SOURCES[source],
        DecisionClass.of(decisionClass),
        Byte.toUnsignedInt(measured));
  }

  /**
   * 1 行のフレームを作る。
   *
   * @param row 行
   * @return フレーム
   * @throws IllegalArgumentException 帳簿の欄の数・O・神託の長さが合わないか、帳簿の値が fp16 で表せないとき
   */
  public byte[] encode(Row row) {
    if (row.ledgerFeatures() != ledgerFeatures) {
      throw new IllegalArgumentException(
          "row has " + row.ledgerFeatures() + " ledger features, the cohort " + ledgerFeatures);
    }
    long[] o = row.observation();
    observation.bind(o, 0);
    if (observation.longs() != o.length || observation.legalActions() != row.actions()) {
      throw new IllegalArgumentException(
          "observation of "
              + o.length
              + " longs with "
              + observation.legalActions()
              + " legal actions does not fit a row of "
              + row.actions()
              + " actions");
    }
    long[] oracle = row.oracle();
    if (oracle.length != 0 && OracleCodec.longs(oracle, 0) != oracle.length) {
      throw new IllegalArgumentException("oracle of " + oracle.length + " longs is malformed");
    }
    int actions = row.actions();
    int measured = row.measuredCount();
    ByteBuffer raw =
        ByteBuffer.allocate(bodySize(o.length, actions, measured, oracle.length))
            .order(ByteOrder.LITTLE_ENDIAN);
    raw.putShort((short) o.length);
    putLongs(raw, o);
    raw.putLong(row.gameId()).putLong(row.decisionId());
    raw.put((byte) row.played());
    for (float value : row.values()) {
      raw.putFloat(value);
    }
    for (float prior : row.prior()) {
      raw.putFloat(prior);
    }
    raw.putInt(row.support());
    raw.put((byte) row.deviation());
    raw.putFloat(row.deviationProbability()).putFloat(row.propensity());
    raw.putLong(row.rootDecisionId());
    float[] ledger = row.ledger();
    long[] outcomes = row.outcomes();
    for (int action = 0; action < actions; action++) {
      if (!row.measured(action)) {
        continue;
      }
      for (int feature = 0; feature < ledgerFeatures; feature++) {
        raw.putShort(half(ledger[action * ledgerFeatures + feature]));
      }
      for (int word = 0; word < OutcomeCodec.WORDS; word++) {
        raw.putLong(outcomes[action * OutcomeCodec.WORDS + word]);
      }
    }
    raw.put((byte) oracle.length);
    putLongs(raw, oracle);
    if (raw.hasRemaining()) {
      throw new IllegalStateException("row body size mismatch");
    }
    return frame(raw.array(), row, measured);
  }

  /**
   * フレームを展開して行に戻す。{@code source} の位置はフレームの先頭で、読んだ分だけ進む。
   *
   * @param source フレームを含むバイト列
   * @return 行
   * @throws IOException フレームが壊れているか、CRC・O・測った手の数・神託の長さが合わないとき
   */
  public Row decode(ByteBuffer source) throws IOException {
    if (source.remaining() < FRAME_HEADER) {
      throw new IOException("Truncated row frame header");
    }
    ByteBuffer header =
        source.slice(source.position(), FRAME_HEADER).order(ByteOrder.LITTLE_ENDIAN);
    int compressedLength = header.getInt(0);
    int rawLength = header.getInt(Integer.BYTES);
    int expected = header.getInt(CRC_OFFSET);
    if (compressedLength < 0
        || compressedLength > source.remaining() - FRAME_HEADER
        || rawLength < 0
        || rawLength > MAX_RAW) {
      throw new IOException("Corrupt row frame header");
    }
    byte[] index = new byte[INDEX_BYTES];
    header.get(INDEX_OFFSET, index);
    byte[] compressed = new byte[compressedLength];
    source.position(source.position() + FRAME_HEADER).get(compressed);
    byte[] raw;
    try {
      raw = Zstd.decompress(compressed, rawLength);
    } catch (RuntimeException corrupt) {
      throw new IOException("Corrupt row frame body", corrupt);
    }
    if (raw.length != rawLength) {
      throw new IOException("Row frame length mismatch");
    }
    CRC32C crc = new CRC32C();
    crc.update(index);
    crc.update(raw);
    if ((int) crc.getValue() != expected) {
      throw new IOException("Row frame CRC mismatch");
    }
    Index entry =
        index(FRAME_HEADER + compressedLength, index[0], index[1], index[2], index[3], index[4]);
    try {
      return body(ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN), entry);
    } catch (RuntimeException corrupt) {
      throw new IOException("Corrupt row frame body", corrupt);
    }
  }

  private Row body(ByteBuffer in, Index index) throws IOException {
    long[] o = getLongs(in, Short.toUnsignedInt(in.getShort()));
    observation.bind(o, 0);
    if (observation.longs() != o.length) {
      throw new IOException("Observation length mismatch");
    }
    int actions = observation.legalActions();
    long gameId = in.getLong();
    long decisionId = in.getLong();
    int played = Byte.toUnsignedInt(in.get());
    float[] values = getFloats(in, actions);
    float[] prior = getFloats(in, actions);
    int support = in.getInt();
    int deviation = in.get();
    float deviationProbability = in.getFloat();
    float propensity = in.getFloat();
    long rootDecisionId = in.getLong();
    float[] ledger = new float[actions * ledgerFeatures];
    Arrays.fill(ledger, Float.NaN);
    long[] outcomes = new long[actions * OutcomeCodec.WORDS];
    int measured = 0;
    for (int action = 0; action < actions; action++) {
      if (Float.isNaN(values[action])) {
        continue;
      }
      measured++;
      for (int feature = 0; feature < ledgerFeatures; feature++) {
        ledger[action * ledgerFeatures + feature] = Float.float16ToFloat(in.getShort());
      }
      for (int word = 0; word < OutcomeCodec.WORDS; word++) {
        outcomes[action * OutcomeCodec.WORDS + word] = in.getLong();
      }
    }
    if (measured != index.measured()) {
      throw new IOException(
          "Row measures " + measured + " actions but its index says " + index.measured());
    }
    long[] oracle = getLongs(in, Byte.toUnsignedInt(in.get()));
    if (oracle.length != 0 && OracleCodec.longs(oracle, 0) != oracle.length) {
      throw new IOException("Oracle length mismatch");
    }
    if (in.hasRemaining()) {
      throw new IOException("Trailing bytes in a row frame");
    }
    return new Row(
        gameId,
        decisionId,
        index.decisionClass(),
        index.purpose(),
        index.group(),
        index.source(),
        played,
        values,
        prior,
        support,
        deviation,
        deviationProbability,
        propensity,
        rootDecisionId,
        ledger,
        outcomes,
        oracle,
        o);
  }

  /** 圧縮して頭を付け、CRC を索引と展開後の本体に掛ける。 */
  private static byte[] frame(byte[] body, Row row, int measured) {
    byte[] compressed = Zstd.compress(body, LEVEL);
    ByteBuffer frame =
        ByteBuffer.allocate(FRAME_HEADER + compressed.length).order(ByteOrder.LITTLE_ENDIAN);
    frame.putInt(compressed.length).putInt(body.length).putInt(0);
    frame.put((byte) row.purpose().ordinal());
    frame.put((byte) row.group().ordinal());
    frame.put((byte) row.source().ordinal());
    frame.put((byte) row.decisionClass().ordinal());
    frame.put((byte) measured);
    frame.put(compressed);
    byte[] bytes = frame.array();
    CRC32C crc = new CRC32C();
    crc.update(bytes, INDEX_OFFSET, INDEX_BYTES);
    crc.update(body);
    frame.putInt(CRC_OFFSET, (int) crc.getValue());
    return bytes;
  }

  private int bodySize(int observationLongs, int actions, int measured, int oracleLongs) {
    if (observationLongs > 0xFFFF || oracleLongs > 0xFF) {
      throw new IllegalArgumentException(
          "record too long: O " + observationLongs + ", oracle " + oracleLongs);
    }
    return Short.BYTES
        + Long.BYTES * observationLongs
        + 2 * Long.BYTES
        + Byte.BYTES
        + 2 * Float.BYTES * actions
        + Integer.BYTES
        + Byte.BYTES
        + 2 * Float.BYTES
        + Long.BYTES
        + measured * (Short.BYTES * ledgerFeatures + Long.BYTES * OutcomeCodec.WORDS)
        + Byte.BYTES
        + Long.BYTES * oracleLongs;
  }

  /** fp16 で厳密に表せる帳簿の値を fp16 にする。 */
  private static short half(float value) {
    short half = Float.floatToFloat16(value);
    if (Float.floatToRawIntBits(Float.float16ToFloat(half)) != Float.floatToRawIntBits(value)) {
      throw new IllegalArgumentException("ledger value is not exact in fp16: " + value);
    }
    return half;
  }

  private static void putLongs(ByteBuffer out, long[] words) {
    for (long word : words) {
      out.putLong(word);
    }
  }

  private static long[] getLongs(ByteBuffer in, int count) {
    long[] words = new long[count];
    for (int index = 0; index < count; index++) {
      words[index] = in.getLong();
    }
    return words;
  }

  private static float[] getFloats(ByteBuffer in, int count) {
    float[] values = new float[count];
    for (int index = 0; index < count; index++) {
      values[index] = in.getFloat();
    }
    return values;
  }
}

package com.epsilon.prism.rows;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.github.luben.zstd.Zstd;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.CRC32C;

/**
 * 行の直列化（版 1）。1行を1フレームにし、行単位で zstd 圧縮と CRC32C を掛ける。行単位なので、学習側は索引から任意の行を独立に展開できる。
 *
 * <pre>
 * frame = [i32 本体の長さ][i32 展開後の長さ][i32 CRC32C(展開後)][u8 purpose][u8 group][u8 source][u8 class][u8 measured][zstd 本体]
 *         （purpose から measured までは行を選ぶのに要る値を、本体を展開せずに読めるように複製したもの。measured は測った手の数）
 * 展開後 = [u8 版][i32 iteration][i64 gameId][i64 decisionId][u8 seat][u8 kind][u8 stage][u8 class][u8 purpose][u8 group][u8 source]
 *          [u16 played][u16 actions][f32 values × actions][f32 prior × actions][u8 support の数][u8 support × 数]
 *          [i16 deviation][f32 deviationProbability][f32 propensity][i64 rootDecisionId][i32 payload の長さ][payload]
 * </pre>
 *
 * 列挙は番号で書く（{@link PrismRow.Purpose}・{@link PrismRow.Group}・{@link PrismRow.Source}・{@link DecisionClass}・{@link Kind}・{@link Stage}）。
 */
public final class RowFormat {
  public static final int VERSION = 1;
  public static final int FRAME_HEADER = 17;
  private static final int LEVEL = 1;
  private static final int MAX_RAW = 4 << 20;
  private static final PrismRow.Purpose[] PURPOSES = PrismRow.Purpose.values();
  private static final PrismRow.Group[] GROUPS = PrismRow.Group.values();
  private static final PrismRow.Source[] SOURCES = PrismRow.Source.values();
  private static final int KINDS = Kind.values().length, STAGES = Stage.values().length;

  private RowFormat() {}

  /** 固定部 32 バイト、values と prior（8 × actions）、support（1 + 数）、根の欄 18 バイト、payload の長さ 4 バイト、payload。 */
  private static int rawSize(PrismRow row, int payloadLength) {
    return 32 + 8 * row.actions() + 1 + row.support().length + 18 + 4 + payloadLength;
  }

  public static byte[] encodeRaw(PrismRow row) {
    if (row.values().length != row.actions() || row.prior().length != row.actions())
      throw new IllegalArgumentException("Row arrays must have one entry per legal action");
    if (row.support().length > 255) throw new IllegalArgumentException("Support is too large");
    byte[] payload = row.payload().toByteArray();
    ByteBuffer out = ByteBuffer.allocate(rawSize(row, payload.length)).order(ByteOrder.LITTLE_ENDIAN);
    out.put((byte) VERSION);
    out.putInt(row.iteration());
    out.putLong(row.gameId());
    out.putLong(row.decisionId());
    out.put((byte) row.seat());
    out.put((byte) row.kind().ordinal());
    out.put((byte) row.stage().ordinal());
    out.put((byte) row.decisionClass().ordinal());
    out.put((byte) row.purpose().ordinal());
    out.put((byte) row.group().ordinal());
    out.put((byte) row.source().ordinal());
    out.putShort((short) row.played());
    out.putShort((short) row.actions());
    for (int a = 0; a < row.actions(); a++) out.putFloat(row.values()[a]);
    for (int a = 0; a < row.actions(); a++) out.putFloat(row.prior()[a]);
    out.put((byte) row.support().length);
    out.put(row.support());
    out.putShort((short) row.deviation());
    out.putFloat(row.deviationProbability());
    out.putFloat(row.propensity());
    out.putLong(row.rootDecisionId());
    out.putInt(payload.length);
    out.put(payload);
    if (out.hasRemaining()) throw new IllegalStateException("Row size mismatch");
    return out.array();
  }

  /** フレーム全体を返す。 */
  public static byte[] encodeFrame(PrismRow row) {
    byte[] raw = encodeRaw(row);
    byte[] compressed = Zstd.compress(raw, LEVEL);
    CRC32C crc = new CRC32C();
    crc.update(raw);
    ByteBuffer frame =
        ByteBuffer.allocate(FRAME_HEADER + compressed.length).order(ByteOrder.LITTLE_ENDIAN);
    frame.putInt(compressed.length).putInt(raw.length).putInt((int) crc.getValue());
    Index index = index(row, FRAME_HEADER + compressed.length);
    frame.put((byte) index.purpose().ordinal());
    frame.put((byte) index.group().ordinal());
    frame.put((byte) index.source().ordinal());
    frame.put((byte) index.decisionClass().ordinal());
    frame.put((byte) index.measured());
    frame.put(compressed);
    return frame.array();
  }

  /**
   * 本体を展開せずに読める索引（行を選ぶのに要る値）。
   *
   * @param frameLength フレーム全体の長さ
   * @param measured 測った手（values が NaN でない手）の数（255 で頭打ち）
   */
  public record Index(
      int frameLength,
      PrismRow.Purpose purpose,
      PrismRow.Group group,
      PrismRow.Source source,
      DecisionClass decisionClass,
      int measured) {}

  /** 行の索引。 */
  public static Index index(PrismRow row, int frameLength) {
    return new Index(
        frameLength,
        row.purpose(),
        row.group(),
        row.source(),
        row.decisionClass(),
        Math.min(255, row.measuredCount()));
  }

  /** 索引の 5 バイト（purpose・group・source・class・measured）から索引を作る。範囲外なら IOException。 */
  public static Index index(int frameLength, byte purpose, byte group, byte source, byte cls, byte measured)
      throws IOException {
    if (frameLength < FRAME_HEADER
        || purpose < 0
        || purpose >= PURPOSES.length
        || group < 0
        || group >= GROUPS.length
        || source < 0
        || source >= SOURCES.length
        || cls < 0
        || cls >= DecisionClass.COUNT) throw new IOException("Corrupt row index");
    return new Index(
        frameLength,
        PURPOSES[purpose],
        GROUPS[group],
        SOURCES[source],
        DecisionClass.of(cls),
        Byte.toUnsignedInt(measured));
  }

  /** source の位置から索引を読む。位置は変えない。 */
  public static Index readIndex(ByteBuffer source, int position) throws IOException {
    ByteBuffer header = source.slice(position, FRAME_HEADER).order(ByteOrder.LITTLE_ENDIAN);
    int compressedLength = header.getInt(0);
    if (compressedLength < 0) throw new IOException("Corrupt row index");
    return index(
        FRAME_HEADER + compressedLength,
        header.get(12),
        header.get(13),
        header.get(14),
        header.get(15),
        header.get(16));
  }

  /** フレームの本体を展開して検査する。source の位置はフレームの先頭で、読んだ分だけ進む。 */
  public static PrismRow decodeFrame(ByteBuffer source) throws IOException {
    ByteBuffer header =
        source.slice(source.position(), FRAME_HEADER).order(ByteOrder.LITTLE_ENDIAN);
    int compressedLength = header.getInt(), rawLength = header.getInt(), expected = header.getInt();
    if (compressedLength < 0 || rawLength < 0 || rawLength > MAX_RAW)
      throw new IOException("Corrupt row frame header");
    byte[] compressed = new byte[compressedLength];
    source.position(source.position() + FRAME_HEADER);
    source.get(compressed);
    byte[] raw = Zstd.decompress(compressed, rawLength);
    if (raw.length != rawLength) throw new IOException("Row frame length mismatch");
    CRC32C crc = new CRC32C();
    crc.update(raw);
    if ((int) crc.getValue() != expected) throw new IOException("Row frame CRC mismatch");
    return decodeRaw(ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN));
  }

  private static PrismRow decodeRaw(ByteBuffer in) throws IOException {
    int version = in.get();
    if (version != VERSION) throw new IOException("Unsupported row version " + version);
    int iteration = in.getInt();
    long gameId = in.getLong(), decisionId = in.getLong();
    int seat = in.get(), kind = in.get(), stage = in.get(), cls = in.get();
    int purpose = in.get(), group = in.get(), source = in.get();
    if (purpose < 0
        || purpose >= PURPOSES.length
        || group < 0
        || group >= GROUPS.length
        || source < 0
        || source >= SOURCES.length
        || cls < 0
        || cls >= DecisionClass.COUNT
        || kind < 0
        || kind >= KINDS
        || stage < 0
        || stage >= STAGES) throw new IOException("Corrupt row enums");
    int played = Short.toUnsignedInt(in.getShort());
    int actions = Short.toUnsignedInt(in.getShort());
    if (played >= actions) throw new IOException("Corrupt row played action " + played);
    float[] values = new float[actions], prior = new float[actions];
    for (int a = 0; a < actions; a++) values[a] = in.getFloat();
    for (int a = 0; a < actions; a++) prior[a] = in.getFloat();
    byte[] support = new byte[Byte.toUnsignedInt(in.get())];
    in.get(support);
    int deviation = in.getShort();
    if (deviation < -1 || deviation >= actions)
      throw new IOException("Corrupt row deviation " + deviation);
    float deviationProbability = in.getFloat(), propensity = in.getFloat();
    long rootDecisionId = in.getLong();
    int payloadLength = in.getInt();
    if (payloadLength != in.remaining()) throw new IOException("Row payload length mismatch");
    byte[] payload = new byte[payloadLength];
    in.get(payload);
    return new PrismRow(
        iteration,
        gameId,
        decisionId,
        seat,
        Kind.of(kind),
        Stage.of((byte) stage),
        DecisionClass.of(cls),
        PURPOSES[purpose],
        GROUPS[group],
        SOURCES[source],
        played,
        actions,
        values,
        prior,
        support,
        deviation,
        deviationProbability,
        propensity,
        rootDecisionId,
        ObservationPayload.takeOwnership(payload));
  }
}

package com.epsilon.cfr.data;

import com.epsilon.nano.ai.decision.data.DecisionInputRowCodec;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;

/** 符号化済み観測を所有する変更不可のバイト列。推論結果、対局記録、学習行の間で同じ保存データを共有し、内部配列を変更可能な形では公開しない。 */
public final class ObservationPayload {
  private static final ThreadLocal<byte[]> TRANSFER_BUFFER =
      ThreadLocal.withInitial(() -> new byte[64 * 1024]);
  private final byte[] bytes;
  private final DecisionInputRowCodec.EncodedRow input;

  private ObservationPayload(byte[] owned) {
    this(owned, null);
  }

  private ObservationPayload(byte[] owned, DecisionInputRowCodec.EncodedRow input) {
    bytes = owned;
    this.input = input;
  }

  /** Nanoが独立して所有する一行へ識別子を付ける。未使用遷移の領域は保存する際にだけ展開する。 */
  public static ObservationPayload fromDecisionInput(
      int marker, DecisionInputRowCodec.EncodedRow input) {
    Objects.requireNonNull(input);
    validateSize(Math.addExact(Integer.BYTES, input.size()));
    byte[] prefix = {
      (byte) (marker >>> 24), (byte) (marker >>> 16), (byte) (marker >>> 8), (byte) marker
    };
    return new ObservationPayload(prefix, input);
  }

  /** 符号化が完了した新規配列の所有権を受け取る。呼び出し側は返却後に配列へアクセスしてはならない。 引き続き外部で使用する配列にはcopyOfを使う。 */
  public static ObservationPayload takeOwnership(byte[] owned) {
    if (owned == null) throw new IllegalArgumentException("Observation payload is required");
    validateSize(owned.length);
    return new ObservationPayload(owned);
  }

  /** 呼び出し側が引き続き所有する配列を、一度だけ複製して取り込む。 */
  public static ObservationPayload copyOf(byte[] source) {
    if (source == null) throw new IllegalArgumentException("Observation payload is required");
    validateSize(source.length);
    return new ObservationPayload(source.clone());
  }

  /** バッファの残りの範囲を複製する。呼び出し側の位置と上限は変えない。 */
  public static ObservationPayload copyOf(ByteBuffer source) {
    if (source == null) throw new IllegalArgumentException("Observation payload is required");
    validateSize(source.remaining());
    byte[] bytes = new byte[source.remaining()];
    source.duplicate().get(bytes);
    return new ObservationPayload(bytes);
  }

  private static void validateSize(int length) {
    if (length > CfrDataChecks.MAX_OBSERVATION_BYTES)
      throw new IllegalArgumentException("Observation payload exceeds maximum size");
  }

  public int size() {
    return bytes.length + (input == null ? 0 : input.size());
  }

  /** 外部で変更可能な独立した配列が必要な場合にだけ、内容を複製する。 */
  public byte[] toByteArray() {
    if (input == null) return bytes.clone();
    byte[] result = Arrays.copyOf(bytes, size());
    input.copyTo(result, bytes.length);
    return result;
  }

  /** 読み取り位置を独立に持つビューを返す。内部配列の取得や書き換えはできない。 */
  public ByteBuffer readOnlyBuffer() {
    return ByteBuffer.wrap(input == null ? bytes : toByteArray()).asReadOnlyBuffer();
  }

  /** 復号用に独立した読み取り位置を持つストリームを開く。内部のバイト列は複製しない。 */
  public InputStream openStream() {
    return new PayloadInputStream(bytes, input == null ? null : input.openStream());
  }

  // ByteArrayInputStream.transferToは転送先へ内部配列を渡すため、読み取りだけを許す。
  // InputStreamから継承するtransferToも、このreadを通じて独立した転送バッファを使う。
  private static final class PayloadInputStream extends InputStream {
    private final byte[] bytes;
    private final InputStream tail;
    private int position;

    PayloadInputStream(byte[] bytes, InputStream tail) {
      this.bytes = bytes;
      this.tail = tail;
    }

    @Override
    public int read() throws IOException {
      if (position < bytes.length) return Byte.toUnsignedInt(bytes[position++]);
      return tail == null ? -1 : tail.read();
    }

    @Override
    public int read(byte[] target, int offset, int length) throws IOException {
      Objects.checkFromIndexSize(offset, length, target.length);
      if (length == 0) return 0;
      int count = Math.min(length, bytes.length - position);
      if (count == 0) return tail == null ? -1 : tail.read(target, offset, length);
      System.arraycopy(bytes, position, target, offset, count);
      position += count;
      return count;
    }

    @Override
    public long skip(long count) throws IOException {
      int skipped = (int) Math.max(0, Math.min(count, bytes.length - position));
      position += skipped;
      return skipped + (tail == null || count <= skipped ? 0 : tail.skip(count - skipped));
    }

    @Override
    public int available() throws IOException {
      return bytes.length - position + (tail == null ? 0 : tail.available());
    }
  }

  // 保存形式を扱う同じパッケージのコードだけに、所有配列からの直接書き込みを許す。
  void write(DataOutputStream output) throws IOException {
    output.writeInt(size());
    output.write(bytes);
    if (input != null) input.writeTo(output, TRANSFER_BUFFER.get());
  }

  static ObservationPayload read(DataInputStream input) throws IOException {
    int length =
        CfrBinaryIO.readCount(input, CfrDataChecks.MAX_OBSERVATION_BYTES, "observation bytes");
    byte[] owned = new byte[length];
    input.readFully(owned);
    return new ObservationPayload(owned);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) return true;
    if (!(other instanceof ObservationPayload payload) || size() != payload.size()) return false;
    if (input == null && payload.input == null) return Arrays.equals(bytes, payload.bytes);
    try (var first = openStream();
        var second = payload.openStream()) {
      for (int value; (value = first.read()) >= 0; ) {
        if (value != second.read()) return false;
      }
      return true;
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  @Override
  public int hashCode() {
    if (input == null) return Arrays.hashCode(bytes);
    int result = 1;
    try (var source = openStream()) {
      for (int value; (value = source.read()) >= 0; ) result = 31 * result + (byte) value;
      return result;
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }
}

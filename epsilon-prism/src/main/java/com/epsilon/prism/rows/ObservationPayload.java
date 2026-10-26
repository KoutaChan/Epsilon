package com.epsilon.prism.rows;

import com.epsilon.nano.ai.decision.data.DecisionInputRowCodec;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Objects;

/** 符号化済み観測を所有する変更不可のバイト列。推論結果と学習行の間で同じ保存データを共有し、内部配列を変更可能な形では公開しない。 */
public final class ObservationPayload {
  private static final int MAX_BYTES = 16 * 1024 * 1024;
  private final byte[] bytes;
  private final DecisionInputRowCodec.EncodedRow input;

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

  /** 符号化が完了した新規配列の所有権を受け取る。呼び出し側は返却後に配列へアクセスしてはならない。 */
  public static ObservationPayload takeOwnership(byte[] owned) {
    if (owned == null) throw new IllegalArgumentException("Observation payload is required");
    validateSize(owned.length);
    return new ObservationPayload(owned, null);
  }

  private static void validateSize(int length) {
    if (length > MAX_BYTES)
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

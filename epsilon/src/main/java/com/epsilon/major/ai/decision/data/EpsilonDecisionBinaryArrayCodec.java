package com.epsilon.major.ai.decision.data;

import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.Objects;

/**
 * Decision の一時保存データで使うプリミティブ配列を、バイナリ形式へ符号化・復号する。
 *
 * <p>呼び出しごとに {@code byte[]} を作ると、一時保存領域自体より GC とゼロ初期化が支配的になりやすい。この補助処理は呼び出し元ごとの {@link Scratch}
 * にバイトバッファを保持し、配列長が増えたときだけ拡張する。
 *
 * <p>論理フォーマットは「int 長さ + 型ごとのリトルエンディアンの未加工のバイト列」。既存の {@link DataOutputStream#writeFloat(float)}
 * はビッグエンディアンのかつ要素ごとの呼び出しになるため使わない。
 */
public final class EpsilonDecisionBinaryArrayCodec {

  private EpsilonDecisionBinaryArrayCodec() {}

  /** 符号化・復号処理呼び出し元ごとに再利用する一時バイトバッファ。スレッドセーフではない。 */
  public static final class Scratch {
    private byte[] bytes = new byte[0];

    private byte[] bytes(int byteCount) {
      if (byteCount < 0) {
        throw new IllegalArgumentException("byteCount must be non-negative");
      }
      if (bytes.length >= byteCount) {
        return bytes;
      }
      int next = Math.max(byteCount, bytes.length + Math.max(bytes.length >>> 1, 4096));
      bytes = new byte[next];
      return bytes;
    }

    private ByteBuffer littleEndianBuffer(int byteCount) {
      return ByteBuffer.wrap(bytes(byteCount), 0, byteCount).order(ByteOrder.LITTLE_ENDIAN);
    }
  }

  public static void writeFloatArray(DataOutputStream out, float[] values, Scratch scratch)
      throws IOException {
    writeFloatArray(out, values, values.length, scratch);
  }

  public static void writeFloatArray(
      DataOutputStream out, float[] values, int length, Scratch scratch) throws IOException {
    int n = normalizedLength(values, length, "float");
    out.writeInt(n);
    if (n == 0) {
      return;
    }
    int byteCount = multiplyExact(n, Float.BYTES, "float array");
    scratch.littleEndianBuffer(byteCount).asFloatBuffer().put(values, 0, n);
    out.write(scratch.bytes, 0, byteCount);
  }

  public static float[] readFloatArray(DataInputStream in, Scratch scratch) throws IOException {
    int n = in.readInt();
    if (n < 0) {
      throw new IOException("Negative float array length: " + n);
    }
    if (n == 0) {
      return new float[0];
    }
    int byteCount = multiplyExact(n, Float.BYTES, "float array");
    byte[] bytes = scratch.bytes(byteCount);
    in.readFully(bytes, 0, byteCount);
    float[] values = new float[n];
    scratch.littleEndianBuffer(byteCount).asFloatBuffer().get(values);
    return values;
  }

  /** 配列を生成せず、再利用一時バッファ上のリトルエンディアンの float ビューとして読む。次の符号化・復号処理読込までだけ有効。 */
  static FloatBuffer readFloatBuffer(DataInputStream in, int expectedLength, Scratch scratch)
      throws IOException {
    int length = readExactLength(in, expectedLength, "float array");
    int byteCount = multiplyExact(length, Float.BYTES, "float array");
    byte[] bytes = scratch.bytes(byteCount);
    in.readFully(bytes, 0, byteCount);
    return scratch.littleEndianBuffer(byteCount).asFloatBuffer();
  }

  /** 配列を生成せず、保存データ上のfloat配列を読み飛ばす。 */
  static void skipFloatArray(DataInputStream in) throws IOException {
    skipArray(in, Float.BYTES, "float array");
  }

  public static void writeLongArray(
      DataOutputStream out, long[] values, int length, Scratch scratch) throws IOException {
    int n = normalizedLength(values, length, "long");
    out.writeInt(n);
    if (n == 0) {
      return;
    }
    int byteCount = multiplyExact(n, Long.BYTES, "long array");
    scratch.littleEndianBuffer(byteCount).asLongBuffer().put(values, 0, n);
    out.write(scratch.bytes, 0, byteCount);
  }

  public static long[] readLongArray(DataInputStream in, int maxLength, Scratch scratch)
      throws IOException {
    int n = in.readInt();
    if (n < 0 || n > maxLength) {
      throw new IOException("Invalid long array length: " + n + ", max=" + maxLength);
    }
    if (n == 0) {
      return new long[0];
    }
    int byteCount = multiplyExact(n, Long.BYTES, "long array");
    byte[] bytes = scratch.bytes(byteCount);
    in.readFully(bytes, 0, byteCount);
    long[] values = new long[n];
    scratch.littleEndianBuffer(byteCount).asLongBuffer().get(values);
    return values;
  }

  public static void writeIntArray(DataOutputStream out, int[] values, Scratch scratch)
      throws IOException {
    writeIntArray(out, values, values.length, scratch);
  }

  public static void writeIntArray(DataOutputStream out, int[] values, int length, Scratch scratch)
      throws IOException {
    int n = normalizedLength(values, length, "int");
    out.writeInt(n);
    if (n == 0) {
      return;
    }
    int byteCount = multiplyExact(n, Integer.BYTES, "int array");
    scratch.littleEndianBuffer(byteCount).asIntBuffer().put(values, 0, n);
    out.write(scratch.bytes, 0, byteCount);
  }

  public static void writeShortArray(DataOutputStream out, short[] values, Scratch scratch)
      throws IOException {
    int n = normalizedLength(values, values.length, "short");
    out.writeInt(n);
    if (n == 0) {
      return;
    }
    int byteCount = multiplyExact(n, Short.BYTES, "short array");
    scratch.littleEndianBuffer(byteCount).asShortBuffer().put(values);
    out.write(scratch.bytes, 0, byteCount);
  }

  /** テンソルごとにまとめたホスト側の連続バッファの連続行ビューを、通常のshort 配列と同じ保存データ形式で書く。 */
  static void writeInputCategories(
      DataOutputStream out, DecisionHostBatch.RowSlice rows, Scratch scratch) throws IOException {
    int length = rows.inputCategoricalElementCount();
    out.writeInt(length);
    int byteCount = multiplyExact(length, Short.BYTES, "categorical input row");
    rows.copyInputCategoriesTo(scratch.littleEndianBuffer(byteCount).asShortBuffer());
    out.write(scratch.bytes, 0, byteCount);
  }

  /** テンソルごとにまとめたホスト側の連続バッファの連続行ビューを、通常のfloat 配列と同じ保存データ形式で書く。 */
  static void writeInputNumerics(
      DataOutputStream out, DecisionHostBatch.RowSlice rows, Scratch scratch) throws IOException {
    int length = rows.inputNumericElementCount();
    out.writeInt(length);
    int byteCount = multiplyExact(length, Float.BYTES, "numeric input row");
    rows.copyInputNumericsTo(scratch.littleEndianBuffer(byteCount).asFloatBuffer());
    out.write(scratch.bytes, 0, byteCount);
  }

  public static int[] readIntArray(DataInputStream in, Scratch scratch) throws IOException {
    int n = in.readInt();
    if (n < 0) {
      throw new IOException("Negative int array length: " + n);
    }
    if (n == 0) {
      return new int[0];
    }
    int byteCount = multiplyExact(n, Integer.BYTES, "int array");
    byte[] bytes = scratch.bytes(byteCount);
    in.readFully(bytes, 0, byteCount);
    int[] values = new int[n];
    scratch.littleEndianBuffer(byteCount).asIntBuffer().get(values);
    return values;
  }

  /** 配列を生成せず、保存データ上のint配列を読み飛ばす。 */
  static void skipIntArray(DataInputStream in) throws IOException {
    skipArray(in, Integer.BYTES, "int array");
  }

  public static short[] readShortArray(DataInputStream in, Scratch scratch) throws IOException {
    int n = in.readInt();
    if (n < 0) {
      throw new IOException("Negative short array length: " + n);
    }
    if (n == 0) {
      return new short[0];
    }
    int byteCount = multiplyExact(n, Short.BYTES, "short array");
    byte[] bytes = scratch.bytes(byteCount);
    in.readFully(bytes, 0, byteCount);
    short[] values = new short[n];
    scratch.littleEndianBuffer(byteCount).asShortBuffer().get(values);
    return values;
  }

  /** 配列を生成せず、再利用一時バッファ上のリトルエンディアンの short ビューとして読む。次の符号化・復号処理読込までだけ有効。 */
  static ShortBuffer readShortBuffer(DataInputStream in, int expectedLength, Scratch scratch)
      throws IOException {
    int length = readExactLength(in, expectedLength, "short array");
    int byteCount = multiplyExact(length, Short.BYTES, "short array");
    byte[] bytes = scratch.bytes(byteCount);
    in.readFully(bytes, 0, byteCount);
    return scratch.littleEndianBuffer(byteCount).asShortBuffer();
  }

  private static int normalizedLength(Object values, int length, String label) {
    if (length < 0) {
      throw new IllegalArgumentException(label + " array length must be non-negative");
    }
    Objects.requireNonNull(values, label + " array");
    int actual =
        values instanceof float[] floats
            ? floats.length
            : values instanceof int[] ints
                ? ints.length
                : values instanceof short[] shorts
                    ? shorts.length
                    : values instanceof long[] longs ? longs.length : -1;
    if (length > actual) {
      throw new IllegalArgumentException(
          label + " array length " + length + " exceeds actual length " + actual);
    }
    return length;
  }

  private static int readExactLength(DataInputStream in, int expectedLength, String label)
      throws IOException {
    if (expectedLength < 0) {
      throw new IllegalArgumentException("expectedLength must be non-negative");
    }
    int actualLength = in.readInt();
    if (actualLength != expectedLength) {
      throw new IOException(
          label + " length mismatch: expected=" + expectedLength + " actual=" + actualLength);
    }
    return actualLength;
  }

  private static void skipArray(DataInputStream in, int elementBytes, String label)
      throws IOException {
    int length = in.readInt();
    if (length < 0) {
      throw new IOException("Negative " + label + " length: " + length);
    }
    in.skipNBytes(multiplyExact(length, elementBytes, label));
  }

  private static int multiplyExact(int length, int bytes, String label) throws IOException {
    try {
      return Math.multiplyExact(length, bytes);
    } catch (ArithmeticException e) {
      throw new IOException("Encoded " + label + " is too large: length=" + length, e);
    }
  }
}

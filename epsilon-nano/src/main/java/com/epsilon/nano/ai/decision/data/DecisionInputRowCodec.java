package com.epsilon.nano.ai.decision.data;

import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.input.DecisionTrainingSlabWriter;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/** Nanoの観測一行を保存・復元する共通形式。教師や学習目的に依存しない。 */
public final class DecisionInputRowCodec {
  private static final byte[] HEADER = header();
  private static final DecisionInputLayout.Tensor[] TENSORS = DecisionInputLayout.Tensor.values();
  private static final int[] NO_PADDING = new int[0];

  private DecisionInputRowCodec() {}

  private static byte[] header() {
    try {
      var bytes = new ByteArrayOutputStream();
      try (var out = new DataOutputStream(bytes)) {
        out.writeInt(DecisionInputSchema.VERSION);
        out.writeUTF(DecisionInputSchema.fingerprint());
      }
      return bytes.toByteArray();
    } catch (IOException impossible) {
      throw new ExceptionInInitializerError(impossible);
    }
  }

  /** 一行の保存に必要なバイト数を返す。固定形状の領域そのものは確保しない。 */
  public static int encodedSize(DecisionHostBatch.RowSlice input) {
    if (input.size() != 1 || input.hasTrainingTargets()) {
      throw new IllegalArgumentException("Decision input codec requires one inference row");
    }
    return Math.addExact(
        HEADER.length + 4 * Integer.BYTES,
        Math.addExact(
            Math.multiplyExact(input.inputCategoricalElementCount(), Short.BYTES),
            Math.multiplyExact(input.inputNumericElementCount(), Float.BYTES)));
  }

  /** 元の保存形式を保ちながら、未使用遷移のゼロを記憶領域へ確保せずに一行を所有する。 */
  public static EncodedRow encode(DecisionHostBatch.RowSlice input) {
    int size = encodedSize(input);
    if (!input.hasCompactTransitions()) {
      // 固定長入力に既にある値は、未使用セルも含めてそのビット列を保存する。
      byte[] bytes = new byte[size];
      writeZeroed(ByteBuffer.wrap(bytes), input);
      return new EncodedRow(bytes, size, NO_PADDING);
    }
    int actions = input.bucket().legalActionCapacity();
    int transitions = input.bucket().actionTransitionCapacity();
    int[] counts = new int[actions];
    int omitted = 0;
    int gaps = 0;
    int previousEnd = -1;
    for (int action = 0; action < actions; action++) {
      int count = input.actionTransitionCount(action);
      counts[action] = count;
      omitted += transitions - count;
      if (count < transitions) {
        int start = action * transitions + count;
        if (start != previousEnd) gaps++;
        previousEnd = (action + 1) * transitions;
      }
    }
    int categoryWidth = 0;
    int numericWidth = 0;
    int transitionTensors = 0;
    for (var tensor : TENSORS) {
      if (!tensor.isTransition()) continue;
      transitionTensors++;
      if (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL)
        categoryWidth += tensor.elementsPerTransition();
      else numericWidth += tensor.elementsPerTransition();
    }
    int categories = input.inputCategoricalElementCount();
    int numerics = input.inputNumericElementCount();
    int compactCategories = categories - omitted * categoryWidth;
    int compactNumerics = numerics - omitted * numericWidth;
    byte[] bytes =
        new byte
            [HEADER.length
                + 4 * Integer.BYTES
                + compactCategories * Short.BYTES
                + compactNumerics * Float.BYTES];
    var output = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
    output.put(HEADER).putInt(actions).putInt(transitions).putInt(categories);
    var categoryOutput = output.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
    categoryOutput.limit(compactCategories);
    output.position(output.position() + compactCategories * Short.BYTES);
    output.order(ByteOrder.BIG_ENDIAN).putInt(numerics);
    var numericOutput = output.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    numericOutput.limit(compactNumerics);
    input.copyCompactInputsTo(categoryOutput, numericOutput);

    int[] zeros = new int[gaps * transitionTensors * 2];
    int used = 0;
    int categoryOffset = HEADER.length + 3 * Integer.BYTES;
    int numericOffset = categoryOffset + categories * Short.BYTES + Integer.BYTES;
    for (var tensor : TENSORS) {
      boolean categorical = tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL;
      int offset = categorical ? categoryOffset : numericOffset;
      int elementBytes = categorical ? Short.BYTES : Float.BYTES;
      if (tensor.isTransition()) {
        int stride = tensor.elementsPerTransition() * elementBytes;
        for (int action = 0; action < actions; action++) {
          int count = counts[action];
          if (count == transitions) continue;
          int start = offset + (action * transitions + count) * stride;
          int length = (transitions - count) * stride;
          if (used > 0 && zeros[used - 2] + zeros[used - 1] == start) {
            zeros[used - 1] += length;
          } else {
            zeros[used++] = start;
            zeros[used++] = length;
          }
        }
      }
      int length = tensor.elementsPerRow(input.bucket()) * elementBytes;
      if (categorical) categoryOffset += length;
      else numericOffset += length;
    }
    return new EncodedRow(bytes, size, used == zeros.length ? zeros : Arrays.copyOf(zeros, used));
  }

  /** 保存時のゼロ領域を省いて所有する一行。入力バッチや再利用配列への参照は持たない。 */
  public static final class EncodedRow {
    private final byte[] bytes;
    private final int size;
    private final int[] zeros;

    private EncodedRow(byte[] bytes, int size, int[] zeros) {
      this.bytes = bytes;
      this.size = size;
      this.zeros = zeros;
    }

    /** 保存形式へ展開したバイト数を返す。 */
    public int size() {
      return size;
    }

    /** ゼロ領域の位置情報を含め、所有するプリミティブ配列のバイト数を返す。 */
    public int storageBytes() {
      return bytes.length + zeros.length * Integer.BYTES;
    }

    /** 所有データへ変更可能な参照を渡さず、保存形式を順次読み出す。 */
    public InputStream openStream() {
      return new RowInputStream();
    }

    /** 呼び出し側の配列へ元の保存形式を複製する。 */
    public void copyTo(byte[] destination, int offset) {
      Objects.checkFromIndexSize(offset, size, destination.length);
      new RowInputStream().read(destination, offset, size);
    }

    /** 呼び出し側が所有する作業領域を通して保存する。転送先には内部配列を渡さない。 */
    public void writeTo(OutputStream output, byte[] scratch) throws IOException {
      if (scratch.length == 0) throw new IllegalArgumentException("Empty row transfer buffer");
      var input = new RowInputStream();
      for (int count; (count = input.read(scratch, 0, scratch.length)) >= 0; )
        output.write(scratch, 0, count);
    }

    private final class RowInputStream extends InputStream {
      private int position;
      private int storedPosition;
      private int zeroIndex;

      private boolean inPadding() {
        while (zeroIndex < zeros.length && position == zeros[zeroIndex] + zeros[zeroIndex + 1])
          zeroIndex += 2;
        return zeroIndex < zeros.length && position >= zeros[zeroIndex];
      }

      @Override
      public int read() {
        if (position == size) return -1;
        boolean padding = inPadding();
        position++;
        return padding ? 0 : Byte.toUnsignedInt(bytes[storedPosition++]);
      }

      @Override
      public int read(byte[] target, int offset, int length) {
        Objects.checkFromIndexSize(offset, length, target.length);
        if (length == 0) return 0;
        int count = Math.min(length, size - position);
        if (count == 0) return -1;
        int remaining = count;
        while (remaining > 0) {
          boolean padding = inPadding();
          int end =
              zeroIndex == zeros.length
                  ? size
                  : zeros[zeroIndex] + (padding ? zeros[zeroIndex + 1] : 0);
          int part = Math.min(remaining, end - position);
          if (padding) Arrays.fill(target, offset, offset + part, (byte) 0);
          else {
            System.arraycopy(bytes, storedPosition, target, offset, part);
            storedPosition += part;
          }
          position += part;
          offset += part;
          remaining -= part;
        }
        return count;
      }

      @Override
      public long skip(long amount) {
        int count = (int) Math.max(0, Math.min(amount, size - position));
        int remaining = count;
        while (remaining > 0) {
          boolean padding = inPadding();
          int end =
              zeroIndex == zeros.length
                  ? size
                  : zeros[zeroIndex] + (padding ? zeros[zeroIndex + 1] : 0);
          int part = Math.min(remaining, end - position);
          if (!padding) storedPosition += part;
          position += part;
          remaining -= part;
        }
        return count;
      }

      @Override
      public int available() {
        return size - position;
      }
    }
  }

  /** ゼロ初期化済みの保存先へ直接書く。未使用遷移のゼロはそのまま残し、中間配列や再利用バッファを経由しない。 保存先の位置だけを進め、バイト順は変更しない。 */
  public static void writeZeroed(ByteBuffer destination, DecisionHostBatch.RowSlice input) {
    int bytes = encodedSize(input);
    if (destination.remaining() < bytes) throw new BufferOverflowException();
    var output = destination.duplicate().order(ByteOrder.BIG_ENDIAN);
    output.put(HEADER);
    output.putInt(input.bucket().legalActionCapacity());
    output.putInt(input.bucket().actionTransitionCapacity());
    int categories = input.inputCategoricalElementCount();
    int numerics = input.inputNumericElementCount();
    output.putInt(categories);
    var categoryOutput = output.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
    categoryOutput.limit(categories);
    output.position(output.position() + categories * Short.BYTES);
    output.order(ByteOrder.BIG_ENDIAN).putInt(numerics);
    var numericOutput = output.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    numericOutput.limit(numerics);
    input.copyInputsToZeroed(categoryOutput, numericOutput);
    destination.position(output.position() + numerics * Float.BYTES);
  }

  /** 既存trajectory形式と同じschema headerとlittle-endian連続バッファを書く。 */
  public static void write(
      DataOutputStream out,
      DecisionHostBatch.RowSlice input,
      EpsilonDecisionBinaryArrayCodec.Scratch scratch)
      throws IOException {
    if (input.size() != 1 || input.hasTrainingTargets()) {
      throw new IOException("Decision input codec requires one inference row");
    }
    out.write(HEADER);
    out.writeInt(input.bucket().legalActionCapacity());
    out.writeInt(input.bucket().actionTransitionCapacity());
    EpsilonDecisionBinaryArrayCodec.writeInputCategories(out, input, scratch);
    EpsilonDecisionBinaryArrayCodec.writeInputNumerics(out, input, scratch);
  }

  /** schemaから確定した長さだけを読み、復号した配列の所有権を一行バッチへ渡す。 */
  public static DecisionHostBatch read(
      DataInputStream in, EpsilonDecisionBinaryArrayCodec.Scratch scratch) throws IOException {
    DecisionBucket bucket = readHeader(in);
    DecisionInputLayout layout = new DecisionInputLayout(1, bucket);
    short[] categories = new short[layout.categoricalElementCount()];
    EpsilonDecisionBinaryArrayCodec.readShortBuffer(in, categories.length, scratch).get(categories);
    float[] numerics = new float[layout.numericElementCount()];
    EpsilonDecisionBinaryArrayCodec.readFloatBuffer(in, numerics.length, scratch).get(numerics);
    for (float value : numerics) {
      if (!Float.isFinite(value)) throw new IOException("Nonfinite Decision input feature");
    }
    try {
      return DecisionHostBatch.takeEncodedInferenceBatch(1, bucket, categories, numerics);
    } catch (IllegalArgumentException | IllegalStateException invalid) {
      throw new IOException("Invalid typed Decision input row", invalid);
    }
  }

  static DecisionBucket readHeader(DataInputStream in) throws IOException {
    int version = in.readInt();
    String fingerprint = in.readUTF();
    if (version != DecisionInputSchema.VERSION
        || !DecisionInputSchema.fingerprint().equals(fingerprint)) {
      throw new IOException(
          "Decision input schema mismatch: version=" + version + " fingerprint=" + fingerprint);
    }
    try {
      return new DecisionBucket(in.readInt(), in.readInt());
    } catch (IllegalArgumentException invalid) {
      throw new IOException("Invalid Decision input bucket", invalid);
    }
  }

  /** 学習用には中間のhost行を作らず、既存slab writerへ直接復号する。 */
  static void readInputs(
      DataInputStream in,
      DecisionBucket bucket,
      EpsilonDecisionBinaryArrayCodec.Scratch scratch,
      DecisionTrainingSlabWriter destination,
      int row)
      throws IOException {
    DecisionInputLayout layout = new DecisionInputLayout(1, bucket);
    destination.writeInputCategories(
        row,
        layout,
        EpsilonDecisionBinaryArrayCodec.readShortBuffer(
            in, layout.categoricalElementCount(), scratch));
    destination.writeInputNumerics(
        row,
        layout,
        EpsilonDecisionBinaryArrayCodec.readFloatBuffer(in, layout.numericElementCount(), scratch));
  }
}

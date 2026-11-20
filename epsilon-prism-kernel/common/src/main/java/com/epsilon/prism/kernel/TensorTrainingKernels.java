package com.epsilon.prism.kernel;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.util.EnumMap;
import java.util.Map;

/**
 * {@link TrainingKernels} と {@link RelationCodes} の GPU 実装（{@code relation_attention.cuh}・{@code rms_norm.cuh}・{@code relation_codes.cuh}）。
 * DJL の GPU の配列をコピーせずに直接読み書きする。DJL と同じストリームに積むので、前後の DJL の演算と順序がそろう。
 * Attention の内積・softmax・p·v とその勾配、RMSNorm の統計とその勾配は FP32 で計算し、出力と入力の勾配だけを BF16 に 1 回丸める。
 * 型は、q・kv・出力・RMSNorm の入出力とそれらの勾配が BF16、関係の属性が UINT8、ほかが FP32・INT32・INT16。配列は全て連続していて
 * （演算の結果か、その先頭からの切り出し）、同じ GPU にあること。
 *
 * <p>区分付き候補 Attention の逆伝播の作業領域は実装が持ち、次の逆伝播で使い回す（足りないときだけ確保し直す）。そのため、1 つの実装を
 * 同時に複数のスレッドから呼ばない。作業領域は {@link #close} で解放する。
 */
public abstract class TensorTrainingKernels
    implements TrainingKernels, RelationCodes, CandidateKernels {

  /** native に渡す要素の型。並びは native の TensorElement と同じ。 */
  private enum Element {
    INT16,
    INT32,
    BFLOAT16,
    FLOAT32,
    UINT8;

    static Element of(DataType type) {
      return switch (type) {
        case INT16 -> INT16;
        case INT32 -> INT32;
        case BFLOAT16 -> BFLOAT16;
        case FLOAT32 -> FLOAT32;
        case UINT8 -> UINT8;
        default ->
            throw new IllegalArgumentException(
                "GPU training kernels do not take " + type + " arrays");
      };
    }
  }

  /** 作業領域の各領域の先頭をそろえる要素の数（FP32・INT32 で 256 バイト）。 */
  private static final long SCRATCH_ALIGNMENT = 64;

  private final long stream;
  private final Device device;

  /** 作業領域を持つマネージャー。最初に作業領域が要るときに作る。 */
  private NDManager scratchManager;

  /** 型ごとの作業領域。 */
  private final Map<DataType, NDArray> scratch = new EnumMap<>(DataType.class);

  protected TensorTrainingKernels(long stream, Device device) {
    this.stream = stream;
    this.device = device;
  }

  @Override
  public void codes(
      Layout layout,
      int rows,
      int events,
      int tokens,
      NDArray relation,
      NDArray rowColumn,
      NDArray tileRelations,
      NDArray codes) {
    relationCodes(
        device.getDeviceId(),
        stream,
        rows,
        events,
        tokens,
        layout.seats(),
        layout.tiles(),
        layout.rowWidth(),
        layout.rowFirstEvent(),
        layout.rowEvents(),
        layout.width(),
        layout.tile(),
        layout.owner(),
        layout.event(),
        layout.riichi(),
        layout.lastDiscard(),
        layout.link(),
        layout.safety(),
        layout.safeties(),
        address(
            relation,
            new Shape((long) rows * layout.fixed() + events, layout.width()),
            DataType.UINT8),
        address(rowColumn, new Shape(rows, layout.rowWidth()), DataType.INT32),
        address(tileRelations, new Shape((long) layout.tiles() * layout.tiles()), DataType.INT32),
        address(codes, new Shape(rows, tokens, tokens, WORDS), DataType.INT16));
  }

  @Override
  public void forward(
      Dimensions d,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      NDArray out,
      NDArray outFloat,
      NDArray logSumExp) {
    attentionForward(
        device.getDeviceId(),
        stream,
        d.rows(),
        d.queries(),
        d.keys(),
        d.codeQueries(),
        d.heads(),
        d.words(),
        d.segments(),
        d.entries(),
        d.scale(),
        d.causalFixedTokens(),
        address(query, queryShape(d), DataType.BFLOAT16),
        address(keyValue, keyValueShape(d), DataType.BFLOAT16),
        address(mask, new Shape(d.rows(), d.keys()), DataType.FLOAT32),
        address(codes, new Shape(d.rows(), d.codeQueries(), d.keys(), d.words()), DataType.INT16),
        address(table, new Shape(d.tableRows(), d.heads()), DataType.FLOAT32),
        address(out, queryShape(d), DataType.BFLOAT16),
        address(outFloat, queryShape(d), DataType.FLOAT32),
        address(logSumExp, new Shape(d.rows(), d.heads(), d.queries()), DataType.FLOAT32));
  }

  @Override
  public void backward(
      Dimensions d,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      NDArray outFloat,
      NDArray logSumExp,
      NDArray gradOut,
      NDArray gradQuery,
      NDArray gradKeyValue,
      NDArray gradTable) {
    attentionBackward(
        device.getDeviceId(),
        stream,
        d.rows(),
        d.queries(),
        d.keys(),
        d.codeQueries(),
        d.heads(),
        d.words(),
        d.segments(),
        d.entries(),
        d.scale(),
        d.causalFixedTokens(),
        address(query, queryShape(d), DataType.BFLOAT16),
        address(keyValue, keyValueShape(d), DataType.BFLOAT16),
        address(mask, new Shape(d.rows(), d.keys()), DataType.FLOAT32),
        address(codes, new Shape(d.rows(), d.codeQueries(), d.keys(), d.words()), DataType.INT16),
        address(table, new Shape(d.tableRows(), d.heads()), DataType.FLOAT32),
        address(outFloat, queryShape(d), DataType.FLOAT32),
        address(logSumExp, new Shape(d.rows(), d.heads(), d.queries()), DataType.FLOAT32),
        address(gradOut, queryShape(d), DataType.BFLOAT16),
        address(gradQuery, queryShape(d), DataType.BFLOAT16),
        address(gradKeyValue, keyValueShape(d), DataType.BFLOAT16),
        address(
            gradTable, new Shape((long) d.rows() * d.heads(), d.tableRows()), DataType.FLOAT32));
  }

  @Override
  public void forward(NormShape d, NDArray input, NDArray gain, NDArray out, NDArray inverseRms) {
    normForward(
        device.getDeviceId(),
        stream,
        d.rows(),
        d.width(),
        d.epsilon(),
        rowsAddress(input, d, DataType.BFLOAT16),
        address(gain, new Shape(d.width()), DataType.FLOAT32),
        rowsAddress(out, d, DataType.BFLOAT16),
        address(inverseRms, new Shape(d.rows()), DataType.FLOAT32));
  }

  @Override
  public void backward(
      NormShape d,
      NDArray input,
      NDArray gain,
      NDArray inverseRms,
      NDArray gradOut,
      NDArray gradInput,
      NDArray gradGain) {
    normBackward(
        device.getDeviceId(),
        stream,
        d.rows(),
        d.width(),
        d.epsilon(),
        rowsAddress(input, d, DataType.BFLOAT16),
        address(gain, new Shape(d.width()), DataType.FLOAT32),
        address(inverseRms, new Shape(d.rows()), DataType.FLOAT32),
        rowsAddress(gradOut, d, DataType.BFLOAT16),
        rowsAddress(gradInput, d, DataType.BFLOAT16),
        address(gradGain, new Shape(d.partials(), d.width()), DataType.FLOAT32));
  }

  private static Shape queryShape(Dimensions d) {
    return new Shape(d.rows(), d.queries(), d.width());
  }

  @Override
  public CandidateKernels candidateKernels() {
    return this;
  }

  /** FP32 のゲートを BF16 の牌の射影に掛け、牌番号の順に足す。 */
  @Override
  public void count(NDArray projected, NDArray gates, NDArray rows, NDArray ids, NDArray output) {
    int batch = Math.toIntExact(projected.getShape().get(0));
    int tiles = Math.toIntExact(projected.getShape().get(1));
    int width = Math.toIntExact(projected.getShape().get(2));
    int transitions = Math.toIntExact(rows.getShape().size());
    int codes = Math.toIntExact(gates.getShape().get(0));
    countForward(
        device.getDeviceId(),
        stream,
        batch,
        transitions,
        tiles,
        codes,
        width,
        address(projected, new Shape(batch, tiles, width), DataType.BFLOAT16),
        address(gates, new Shape(codes, width), DataType.FLOAT32),
        address(rows, new Shape(transitions), DataType.INT32),
        address(ids, new Shape(transitions, tiles), DataType.INT32),
        address(output, new Shape(transitions, width), DataType.BFLOAT16));
  }

  /** 牌の勾配を FP32 で累積して BF16 で書き、ゲートの勾配を FP32 で返す。 */
  @Override
  public void countGradient(
      NDArray projected,
      NDArray gates,
      NDArray rows,
      NDArray ids,
      NDArray gradient,
      NDArray gradProjected,
      NDArray gradGate) {
    int batch = Math.toIntExact(projected.getShape().get(0));
    int tiles = Math.toIntExact(projected.getShape().get(1));
    int width = Math.toIntExact(projected.getShape().get(2));
    int transitions = Math.toIntExact(rows.getShape().size());
    int codes = Math.toIntExact(gates.getShape().get(0));
    countBackward(
        device.getDeviceId(),
        stream,
        batch,
        transitions,
        tiles,
        codes,
        width,
        address(projected, new Shape(batch, tiles, width), DataType.BFLOAT16),
        address(gates, new Shape(codes, width), DataType.FLOAT32),
        address(rows, new Shape(transitions), DataType.INT32),
        address(ids, new Shape(transitions, tiles), DataType.INT32),
        address(gradient, new Shape(transitions, width), DataType.BFLOAT16),
        address(gradProjected, new Shape(batch, tiles, width), DataType.BFLOAT16),
        address(gradGate, new Shape(codes, width), DataType.FLOAT32));
  }

  @Override
  public void candidateAttention(CandidateInput input, NDArray output, NDArray probabilities) {
    int[] d = input.dimensions();
    segmentAttentionForward(
        device.getDeviceId(),
        stream,
        d,
        input.scale(),
        candidateAddresses(input, d),
        address(output, input.query().getShape(), DataType.FLOAT32),
        address(probabilities, input.probabilities(), DataType.FLOAT32));
  }

  @Override
  public void candidateAttentionGradient(
      CandidateInput input,
      NDArray probabilities,
      NDArray upstream,
      NDArray gradQuery,
      NDArray gradMemory,
      NDArray gradBase,
      NDArray gradWeights) {
    int[] d = input.dimensions();
    int buckets = d[4] + input.layout().tableRows();
    long[] floats =
        scratch(
            gradQuery.getManager(),
            DataType.FLOAT32,
            input.probabilities().size(),
            (long) buckets * CollectionCapabilities.limit("TableGradientShards") * d[5]);
    long[] ints =
        scratch(
            gradQuery.getManager(),
            DataType.INT32,
            buckets,
            buckets + 1L,
            (long) d[0] * input.layout().codeWidth());
    long[] gradients = {
      address(probabilities, input.probabilities(), DataType.FLOAT32),
      address(upstream, input.query().getShape(), DataType.FLOAT32),
      address(gradQuery, input.query().getShape(), DataType.FLOAT32),
      address(gradMemory, input.memory().getShape(), DataType.FLOAT32),
      address(gradBase, input.base().getShape(), DataType.FLOAT32),
      address(gradWeights, input.weights().getShape(), DataType.FLOAT32),
      floats[0],
      floats[1],
      ints[0],
      ints[1],
      ints[2]
    };
    long[] inputs = candidateAddresses(input, d);
    segmentAttentionBackward(device.getDeviceId(), stream, d, input.scale(), inputs, gradients);
  }

  /**
   * 型ごとの作業領域を、指定した要素数の領域に先頭から順に分け、各領域のアドレスを返す。前の逆伝播の作業領域が足りれば使い回し、足りなければ
   * 確保し直す。作業領域は確保したときのストリームで使うので、前の演算が読み終える前に次の演算を積んでも、読み書きの順はそのストリームの順になる。
   *
   * @param arrays 呼び出し元の配列のマネージャー（作業領域を同じエンジンで作る）
   * @param type 要素の型
   * @param lengths 領域ごとの要素の数
   * @return 領域ごとの先頭のアドレス
   */
  private long[] scratch(NDManager arrays, DataType type, long... lengths) {
    long[] offsets = new long[lengths.length];
    long total = 0;
    for (int i = 0; i < lengths.length; i++) {
      offsets[i] = total;
      total += Math.ceilDiv(lengths[i], SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT;
    }
    NDArray buffer = scratch.get(type);
    if (buffer == null || buffer.size() < total) {
      if (scratchManager == null) scratchManager = arrays.getEngine().newBaseManager(device);
      if (buffer != null) buffer.close();
      buffer = scratchManager.create(new Shape(total), type);
      scratch.put(type, buffer);
    }
    long base = deviceAddress(buffer, type);
    long[] addresses = new long[lengths.length];
    for (int i = 0; i < lengths.length; i++) {
      addresses[i] = base + offsets[i] * type.getNumOfBytes();
    }
    return addresses;
  }

  /** 作業領域を解放する。提出した演算が終わった後に呼ぶ。 */
  @Override
  public void close() {
    scratch.clear();
    if (scratchManager != null) {
      scratchManager.close();
      scratchManager = null;
    }
  }

  private long[] candidateAddresses(CandidateInput a, int[] d) {
    return new long[] {
      address(a.query(), new Shape(d[0], d[7], 16), DataType.FLOAT32),
      address(a.memory(), new Shape(d[6], d[1], d[5]), DataType.FLOAT32),
      address(a.base(), new Shape(d[4], d[5]), DataType.FLOAT32),
      address(a.weights(), new Shape(a.layout().tableRows(), d[5]), DataType.FLOAT32),
      address(a.codes(), new Shape(d[0], a.layout().codeWidth()), DataType.UINT8),
      address(a.groups(), new Shape(d[0]), DataType.INT32)
    };
  }

  private static Shape keyValueShape(Dimensions d) {
    return new Shape(d.rows(), d.keys(), 2L * d.width());
  }

  /** 配列の形・型・デバイスを照合し、データが置かれた GPU のアドレスを返す。 */
  private long address(NDArray array, Shape shape, DataType type) {
    if (!array.getShape().equals(shape)) {
      throw new IllegalArgumentException(
          "GPU training kernels expect "
              + type
              + shape
              + ", got "
              + array.getDataType()
              + array.getShape());
    }
    return deviceAddress(array, type);
  }

  /** RMSNorm の行の配列のアドレスを返す。最後の軸の長さが幅に等しく、要素の数が行数 × 幅であること。 */
  private long rowsAddress(NDArray array, NormShape d, DataType type) {
    Shape shape = array.getShape();
    if (shape.dimension() == 0
        || shape.get(shape.dimension() - 1) != d.width()
        || shape.size() != (long) d.rows() * d.width()) {
      throw new IllegalArgumentException(
          "GPU RMSNorm expects " + type + " rows of " + d + ", got " + array.getDataType() + shape);
    }
    return deviceAddress(array, type);
  }

  /** 配列の型・デバイスを照合し、データが置かれた GPU のアドレスを返す。 */
  private long deviceAddress(NDArray array, DataType type) {
    if (array.getDataType() != type) {
      throw new IllegalArgumentException(
          "GPU training kernels expect "
              + type
              + ", got "
              + array.getDataType()
              + array.getShape());
    }
    if (!device.equals(array.getDevice()))
      throw new IllegalArgumentException("Training tensor is on a different device");
    return tensorAddress(contiguousHandle(array), Element.of(type).ordinal(), device.getDeviceId());
  }

  protected abstract long contiguousHandle(NDArray array);

  protected abstract void relationCodes(
      int device,
      long stream,
      int rows,
      int events,
      int tokens,
      int seats,
      int tiles,
      int rowWidth,
      int rowFirstEvent,
      int rowEvents,
      int width,
      int tile,
      int owner,
      int event,
      int riichi,
      int lastDiscard,
      int link,
      int safety,
      int safeties,
      long relation,
      long rowColumn,
      long tileRelation,
      long codes);

  protected abstract void attentionForward(
      int device,
      long stream,
      int rows,
      int queries,
      int keys,
      int codeQueries,
      int heads,
      int words,
      int segments,
      int entries,
      float scale,
      int causalFixedTokens,
      long query,
      long keyValue,
      long mask,
      long codes,
      long table,
      long out,
      long outFloat,
      long logSumExp);

  protected abstract void attentionBackward(
      int device,
      long stream,
      int rows,
      int queries,
      int keys,
      int codeQueries,
      int heads,
      int words,
      int segments,
      int entries,
      float scale,
      int causalFixedTokens,
      long query,
      long keyValue,
      long mask,
      long codes,
      long table,
      long outFloat,
      long logSumExp,
      long gradOut,
      long gradQuery,
      long gradKeyValue,
      long gradTable);

  protected abstract void normForward(
      int device,
      long stream,
      int rows,
      int width,
      float epsilon,
      long input,
      long gain,
      long out,
      long inverseRms);

  protected abstract void normBackward(
      int device,
      long stream,
      int rows,
      int width,
      float epsilon,
      long input,
      long gain,
      long inverseRms,
      long gradOut,
      long gradInput,
      long gradGain);

  protected abstract void countForward(
      int device,
      long stream,
      int rows,
      int transitions,
      int tiles,
      int codes,
      int width,
      long projected,
      long gates,
      long rowIndex,
      long ids,
      long output);

  protected abstract void countBackward(
      int device,
      long stream,
      int rows,
      int transitions,
      int tiles,
      int codes,
      int width,
      long projected,
      long gates,
      long rowIndex,
      long ids,
      long gradOutput,
      long gradProjected,
      long gradGate);

  protected abstract void segmentAttentionForward(
      int device,
      long stream,
      int[] dimensions,
      float scale,
      long[] inputs,
      long output,
      long probabilities);

  protected abstract void segmentAttentionBackward(
      int device, long stream, int[] dimensions, float scale, long[] inputs, long[] gradients);

  protected abstract long tensorAddress(long handle, int element, int gpu);
}

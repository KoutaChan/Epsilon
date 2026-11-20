package com.epsilon.prism.hip;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtStream;
import java.util.Objects;

/**
 * {@link TrainingKernels} と {@link RelationCodes} の GPU 実装（{@code t1_attention.hip}・{@code t2_norm.hip}・{@code t0_relation.hip}）。
 * DJL の GPU の配列をコピーせずに直接読み書きする。DJL と同じストリームに積むので、前後の DJL の演算と順序がそろう。
 * Attention の内積・softmax・p·v とその勾配、RMSNorm の統計とその勾配は FP32 で計算し、出力と入力の勾配だけを BF16 に 1 回丸める。
 * 型は、q・kv・出力・RMSNorm の入出力とそれらの勾配が BF16、関係の属性が UINT8、ほかが FP32・INT32・INT16。配列は全て連続していて
 * （演算の結果か、その先頭からの切り出し）、同じ GPU にあること。
 */
public final class HipAttention implements TrainingKernels, RelationCodes, CandidateKernels {

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
                "HIP training kernels do not take " + type + " arrays");
      };
    }
  }

  private final long stream;
  private final Device device;

  private HipAttention(long stream, Device device) {
    this.stream = stream;
    this.device = device;
  }

  /**
   * 演算を GPU のストリームに積む実装を作る。
   *
   * @param library 読み込み済みの libprism_hip.so
   * @param stream 演算を積むストリーム（呼び出し元の DJL の演算と同じもの）
   * @return 作った実装
   * @throws IllegalArgumentException ストリームが GPU のものでないとき
   */
  public static HipAttention on(HipLibrary library, PtStream stream) {
    Objects.requireNonNull(library, "library");
    if (!stream.getDevice().isGpu()) {
      throw new IllegalArgumentException("HIP attention needs a GPU stream");
    }
    library.requireArchitecture(stream.getDevice().getDeviceId());
    return new HipAttention(stream.getId(), stream.getDevice());
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
    PrismHip.relationCodes(
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
    PrismHip.attentionForward(
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
    PrismHip.attentionBackward(
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
    PrismHip.normForward(
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
    PrismHip.normBackward(
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

  /** V4 candidate counting across tiles. The gate is FP32 and the two projections use BF16. */
  @Override
  public void count(NDArray projected, NDArray gates, NDArray rows, NDArray ids, NDArray output) {
    int batch = Math.toIntExact(projected.getShape().get(0));
    int tiles = Math.toIntExact(projected.getShape().get(1));
    int width = Math.toIntExact(projected.getShape().get(2));
    int transitions = Math.toIntExact(rows.getShape().size());
    int codes = Math.toIntExact(gates.getShape().get(0));
    PrismHip.countForward(
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

  /** VJP of the count reduction. Accumulate in FP32 and write the projected-tile gradient in BF16. */
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
    PrismHip.countBackward(
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
    PrismHip.segmentAttentionForward(
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
      NDArray gradWeights,
      Runnable beforeLaunch) {
    int[] d = input.dimensions();
    int buckets = d[4] + input.layout().tableRows();
    var manager = gradQuery.getManager();
    NDArray score = manager.create(input.probabilities(), DataType.FLOAT32);
    NDArray partial = manager.create(new Shape(buckets, 32, d[5]), DataType.FLOAT32);
    NDArray cursor = manager.create(new Shape(buckets), DataType.INT32);
    NDArray offsets = manager.create(new Shape(buckets + 1), DataType.INT32);
    long codeCount = (long) d[0] * input.layout().codeWidth();
    NDArray occurrences = manager.create(new Shape(codeCount), DataType.INT32);
    long[] gradients = {
      address(probabilities, input.probabilities(), DataType.FLOAT32),
      address(upstream, input.query().getShape(), DataType.FLOAT32),
      address(gradQuery, input.query().getShape(), DataType.FLOAT32),
      address(gradMemory, input.memory().getShape(), DataType.FLOAT32),
      address(gradBase, input.base().getShape(), DataType.FLOAT32),
      address(gradWeights, input.weights().getShape(), DataType.FLOAT32),
      address(score, score.getShape(), DataType.FLOAT32),
      address(partial, partial.getShape(), DataType.FLOAT32),
      address(cursor, cursor.getShape(), DataType.INT32),
      address(offsets, offsets.getShape(), DataType.INT32),
      address(occurrences, occurrences.getShape(), DataType.INT32)
    };
    long[] inputs = candidateAddresses(input, d);
    if (beforeLaunch != null) {
      beforeLaunch.run();
    }
    PrismHip.segmentAttentionBackward(stream, d, input.scale(), inputs, gradients);
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
          "HIP training kernels expect "
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
          "HIP RMSNorm expects " + type + " rows of " + d + ", got " + array.getDataType() + shape);
    }
    return deviceAddress(array, type);
  }

  /** 配列の型・デバイスを照合し、データが置かれた GPU のアドレスを返す。 */
  private long deviceAddress(NDArray array, DataType type) {
    if (array.getDataType() != type) {
      throw new IllegalArgumentException(
          "HIP training kernels expect "
              + type
              + ", got "
              + array.getDataType()
              + array.getShape());
    }
    if (!device.equals(array.getDevice()) || !(array instanceof PtNDArray tensor)) {
      throw new IllegalArgumentException("HIP training arrays must be PyTorch arrays on " + device);
    }
    return PrismHip.tensorAddress(
        tensor.getHandle(), Element.of(type).ordinal(), device.getDeviceId());
  }
}

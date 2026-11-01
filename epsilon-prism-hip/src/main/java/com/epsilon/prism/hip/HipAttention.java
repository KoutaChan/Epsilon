package com.epsilon.prism.hip;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtStream;
import java.util.Objects;

/**
 * {@link TrainingKernels} と {@link RelationCodes} の GPU の演算（{@code t1_attention.hip}・{@code t2_norm.hip}・{@code t0_relation.hip}）。
 * DJL の GPU の配列を写さずに読み書きし、ストリームの順で前後の DJL の演算とそろう。注意の内積・softmax・p·v と勾配、RMSNorm の統計と
 * 勾配は FP32 で、出力と入力の勾配は BF16 に 1 回丸める。q・kv・出力・RMSNorm の入出力とそれらの勾配は BF16、関係の属性は UINT8、ほかは
 * FP32・INT32・INT16 で、配列は全て連続（演算の結果か、その先頭からの切り出し）で同じ GPU にあること。
 */
public final class HipAttention implements TrainingKernels, RelationCodes {

  /** native に渡す要素の型（native の TensorElement と同じ順）。 */
  private enum Element {
    BOOL,
    INT16,
    INT32,
    BFLOAT16,
    FLOAT32,
    UINT8;

    static Element of(DataType type) {
      return switch (type) {
        case BOOLEAN -> BOOL;
        case INT16 -> INT16;
        case INT32 -> INT32;
        case BFLOAT16 -> BFLOAT16;
        case FLOAT32 -> FLOAT32;
        case UINT8 -> UINT8;
        default -> throw new IllegalArgumentException("HIP training kernels do not take " + type + " arrays");
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
   * GPU のストリームに積む演算を作る。
   *
   * @param library 読み込み済みの libprism_hip.so
   * @param stream 積むストリーム（呼び手の DJL の演算と同じストリーム）
   * @return 演算
   * @throws IllegalArgumentException ストリームが GPU のものでないとき
   */
  public static HipAttention on(HipLibrary library, PtStream stream) {
    Objects.requireNonNull(library, "library");
    if (!stream.getDevice().isGpu()) {
      throw new IllegalArgumentException("HIP attention needs a GPU stream");
    }
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
        address(relation, new Shape((long) rows * layout.fixed() + events, layout.width()), DataType.UINT8),
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
        address(gradTable, new Shape((long) d.rows() * d.heads(), d.tableRows()), DataType.FLOAT32));
  }

  @Override
  public void forward(
      NormShape d, NDArray input, NDArray gain, NDArray out, NDArray inverseRms) {
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

  private static Shape keyValueShape(Dimensions d) {
    return new Shape(d.rows(), d.keys(), 2L * d.width());
  }

  /** 形・型・デバイスを照らして、配列のデータのデバイスのアドレスを返す。 */
  private long address(NDArray array, Shape shape, DataType type) {
    if (!array.getShape().equals(shape)) {
      throw new IllegalArgumentException(
          "HIP training kernels expect " + type + shape + ", got " + array.getDataType() + array.getShape());
    }
    return deviceAddress(array, type);
  }

  /** RMSNorm の行の配列（最後の軸が width で、要素が rows·width）のアドレスを返す。 */
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

  /** 型・デバイスを照らして、配列のデータのデバイスのアドレスを返す。 */
  private long deviceAddress(NDArray array, DataType type) {
    if (array.getDataType() != type) {
      throw new IllegalArgumentException(
          "HIP training kernels expect " + type + ", got " + array.getDataType() + array.getShape());
    }
    if (!device.equals(array.getDevice()) || !(array instanceof PtNDArray tensor)) {
      throw new IllegalArgumentException("HIP training arrays must be PyTorch arrays on " + device);
    }
    return PrismHip.tensorAddress(tensor.getHandle(), Element.of(type).ordinal(), device.getDeviceId());
  }
}

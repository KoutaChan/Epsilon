package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.EventField;
import com.epsilon.prism.input.InputSchema.RoundField;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.TileField;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.TokenLayout;
import java.util.Arrays;
import java.util.List;

/**
 * トークンの欄を埋め込みの入力に並べる。
 *
 * <p>2 値の欄（{@link FieldSpec.Coding#FLAG}）は埋め込まず 0/1 の数値にする。ほかの欄は {@link TokenLayout} の名前空間で表を引く。
 * 表の行の数は {@link TokenLayout#tableSize()} のままにして、書き手・HIP と同じ番地を使う（2 値の欄の名前空間の行は引かない）。並びは
 * 埋め込む欄の列の順の [埋め込み × d]、2 値の欄の列の順の旗、呼び出し側が渡す数値の順。
 *
 * <p>列の添字は順伝播ごとに {@link BatchIndex} が 1 本の転送に載せて渡す（{@link Columns}）。
 */
final class FieldPack {

  static final FieldPack ROUND = of(InputSchema.ROUND, RoundField.class);
  static final FieldPack SEAT = of(InputSchema.SEAT, SeatField.class);
  static final FieldPack TILE = of(InputSchema.TILE, TileField.class);
  static final FieldPack EVENT = of(InputSchema.EVENT, EventField.class);
  static final FieldPack TRANSITION = of(InputSchema.TRANSITION, TransitionField.class);
  static final List<FieldPack> ALL = List.of(ROUND, SEAT, TILE, EVENT, TRANSITION);

  /**
   * デバイスの上の列の添字。
   *
   * @param embedded 埋め込む列 [k]（INT64）
   * @param offsets 埋め込む列の名前空間の先頭 [1, k]（INT32）
   * @param flags 2 値の列 [m]（INT64）
   */
  record Columns(NDArray embedded, NDArray offsets, NDArray flags) {}

  private final long[] embedded;
  private final int[] offsets;
  private final long[] flags;
  private final int tableRows;

  private FieldPack(long[] embedded, int[] offsets, long[] flags, int tableRows) {
    this.embedded = embedded;
    this.offsets = offsets;
    this.flags = flags;
    this.tableRows = tableRows;
  }

  private static <E extends Enum<E> & InputSchema.BoundedField> FieldPack of(
      TokenLayout<E> layout, Class<E> type) {
    int[] namespaces = layout.columnOffsets();
    long[] embedded = new long[layout.width()];
    long[] flags = new long[layout.width()];
    int embeddedCount = 0;
    int flagCount = 0;
    for (E field : type.getEnumConstants()) {
      FieldSpec spec = field.spec();
      for (int repeat = 0; repeat < spec.repeats(); repeat++) {
        int column = layout.column(field, repeat);
        if (spec.coding() == FieldSpec.Coding.FLAG) {
          flags[flagCount++] = column;
        } else {
          embedded[embeddedCount++] = column;
        }
      }
    }
    if (embeddedCount == 0 || flagCount == 0) {
      throw new IllegalStateException(type.getSimpleName() + " needs embedded and flag columns");
    }
    long[] embeddedColumns = Arrays.copyOf(embedded, embeddedCount);
    int[] offsets = new int[embeddedCount];
    for (int i = 0; i < embeddedCount; i++) {
      offsets[i] = namespaces[(int) embeddedColumns[i]];
    }
    return new FieldPack(
        embeddedColumns, offsets, Arrays.copyOf(flags, flagCount), layout.tableSize());
  }

  /** 表の行の数。 */
  int tableRows() {
    return tableRows;
  }

  /** 埋め込む列（転送に載せる値。変えない）。 */
  long[] embeddedColumns() {
    return embedded;
  }

  /** 埋め込む列の名前空間の先頭（転送に載せる値。変えない）。 */
  int[] namespaceOffsets() {
    return offsets;
  }

  /** 2 値の列（転送に載せる値。変えない）。 */
  long[] flagColumns() {
    return flags;
  }

  /**
   * 並べた入力の幅を返す。
   *
   * @param dimension 埋め込みの次元
   * @param scalars 呼び出し側が足す数値の列の数
   * @return 幅
   */
  int inputWidth(int dimension, int scalars) {
    return embedded.length * dimension + flags.length + scalars;
  }

  /**
   * [N, 欄] の値から [N, 埋め込み × d + 旗 + 数値] を作る。
   *
   * @param category 欄の値 [N, 欄]（UINT8）
   * @param columns この並べ方の列の添字（{@link BatchIndex#columns}）
   * @param table 埋め込みの表 [表の行, d]（計算の型）
   * @param scalars 足す数値 [N, k]（計算の型。0 個でもよい）
   * @return 並べた入力（計算の型）
   */
  NDArray pack(NDArray category, Columns columns, NDArray table, NDArray... scalars) {
    NDArray values = category.toType(DataType.INT32, false);
    NDArray ids = values.get(new NDIndex(":, {}", columns.embedded()));
    NDList features = new NDList(scalars.length + 1);
    features.add(
        values.get(new NDIndex(":, {}", columns.flags())).toType(table.getDataType(), false));
    features.addAll(new NDList(scalars));
    NDArray dense = features.size() == 1 ? features.get(0) : NDArrays.concat(features, 1);
    return NDArrays.embeddingFeaturePack(ids, columns.offsets(), table, dense);
  }
}

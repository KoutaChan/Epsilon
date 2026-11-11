package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.TokenLayout;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * トークンのフィールドを、埋め込みの入力に並べる。
 *
 * <p>2 値のフィールド（{@link FieldSpec.Coding#FLAG}）は埋め込まず、0/1 の数値のまま使う。ほかのフィールドは {@link TokenLayout} の
 * 名前空間で表を引く。表の行の数は {@link TokenLayout#tableSize()} のままにして、ライターや HIP と同じ行番号を使う（2 値のフィールドの
 * 名前空間の行は引かない）。並べる順は、埋め込むフィールドを列の順に並べた [埋め込み × d]、2 値のフィールドを列の順に並べたフラグ、
 * 呼び出し側が渡す数値の順。
 *
 * <p>列の添字は、順伝播ごとに {@link BatchIndex} が 1 回の転送にまとめて渡す（{@link Columns}）。並べ方は入力の形式ごとに 1 回だけ作る
 * （{@link #of(InputFormat)}）。
 */
final class FieldPack {

  /** 入力の形式ごとの並べ方。 */
  private static final Map<InputFormat, Packs> PACKS = new ConcurrentHashMap<>();

  /**
   * 1 つの入力の形式で使うフィールドの並べ方。トークンの型ごとに 1 つと、遷移に 1 つを持つ。
   *
   * @param round 局
   * @param seat 席
   * @param tile 牌
   * @param event イベント
   * @param transition 遷移
   */
  record Packs(FieldPack round, FieldPack seat, FieldPack tile, FieldPack event, FieldPack transition) {

    /** 全ての並べ方（宣言の順）。 */
    List<FieldPack> all() {
      return List.of(round, seat, tile, event, transition);
    }

    /** トークンの型に対応する並べ方。 */
    FieldPack of(TokenType type) {
      return switch (type) {
        case ROUND -> round;
        case SEAT -> seat;
        case TILE -> tile;
        case EVENT -> event;
      };
    }
  }

  /**
   * 入力の形式に対応する並べ方を返す。最初に呼んだときに作り、後はそれを返す。
   *
   * @param format 入力の形式
   * @return 並べ方
   */
  static Packs of(InputFormat format) {
    return PACKS.computeIfAbsent(
        format,
        f ->
            new Packs(
                of(f.round(), "round"),
                of(f.seat(), "seat"),
                of(f.tile(), "tile"),
                of(f.event(), "event"),
                of(f.transition(), "transition")));
  }

  /**
   * デバイスに載せた列の添字。
   *
   * @param embedded 埋め込む列 [k]（INT64）
   * @param offsets 埋め込む列ごとの名前空間の先頭 [1, k]（INT32）
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

  private static FieldPack of(TokenLayout<?> layout, String token) {
    int[] namespaces = layout.columnOffsets();
    long[] embedded = new long[layout.width()];
    long[] flags = new long[layout.width()];
    int embeddedCount = 0;
    int flagCount = 0;
    for (int column = 0; column < layout.width(); column++) {
      if (layout.coding(column) == FieldSpec.Coding.FLAG) {
        flags[flagCount++] = column;
      } else {
        embedded[embeddedCount++] = column;
      }
    }
    if (embeddedCount == 0 || flagCount == 0) {
      throw new IllegalStateException(token + " needs embedded and flag columns");
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

  /** 埋め込む列。転送に載せる値で、書き換えてはならない。 */
  long[] embeddedColumns() {
    return embedded;
  }

  /** 埋め込む列ごとの名前空間の先頭。転送に載せる値で、書き換えてはならない。 */
  int[] namespaceOffsets() {
    return offsets;
  }

  /** 2 値の列。転送に載せる値で、書き換えてはならない。 */
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
   * [N, フィールド] の値から、[N, 埋め込み × d + フラグ + 数値] の入力を作る。
   *
   * @param category フィールドの値 [N, フィールド]（UINT8）
   * @param columns この並べ方の列の添字（{@link BatchIndex#columns}）
   * @param table 埋め込みの表 [表の行, d]（計算の型）
   * @param scalars 後ろに足す数値 [N, k]（計算の型。1 つも渡さなくてもよい）
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

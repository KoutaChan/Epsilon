package com.epsilon.classic.input;

import com.epsilon.classic.input.Buckets.Bucket;
import java.util.Arrays;
import java.util.Objects;

/**
 * 行をまとめたバッチの入力。入力の配置と行の大きさの区分は、作るときに 1 つずつ決める。整数のテンソルは 1 つの short の配列に、
 * 実数のテンソルは 1 つの float の配列にまとめる。どちらの配列もテンソルを宣言の順に続けて置き、各テンソルの中では行の順に 1 行ずつ要素を並べる。
 * 使う側が作って使い回し、{@link ClassicTokenizer} が行ごとに書き込む。同時に使えるのは 1 つのスレッドだけ。
 */
public final class ClassicInputs {

  private static final Tensor[] TENSORS = Tensor.values();

  private final ClassicSchema schema;
  private final Bucket bucket;
  private final int rows;
  private final short[] categorical;
  private final float[] numeric;

  /** テンソルごと（ordinal で引く）に、配列の中でそのテンソルが始まる位置。 */
  private final int[] starts = new int[Tensor.values().length];

  /** テンソルごと（ordinal で引く）の、1 行の要素数。 */
  private final int[] rowWidths = new int[Tensor.values().length];

  /**
   * 指定した入力の配置と区分で、行のバッチを確保する。
   *
   * @param schema アーキテクチャごとの入力の配置
   * @param bucket 行の大きさの区分
   * @param rows 行数（1 以上）
   * @throws IllegalArgumentException 行数が 1 未満のとき
   */
  public ClassicInputs(ClassicSchema schema, Bucket bucket, int rows) {
    this.schema = Objects.requireNonNull(schema, "schema");
    this.bucket = Objects.requireNonNull(bucket, "bucket");
    if (rows < 1) {
      throw new IllegalArgumentException("Rows must be positive: " + rows);
    }
    this.rows = rows;
    int categoricalSize = 0;
    int numericSize = 0;
    for (Tensor tensor : Tensor.values()) {
      int width = schema.rowWidth(tensor, bucket);
      rowWidths[tensor.ordinal()] = width;
      int size = Math.multiplyExact(width, rows);
      if (tensor.slab() == Tensor.Slab.CATEGORICAL) {
        starts[tensor.ordinal()] = categoricalSize;
        categoricalSize = Math.addExact(categoricalSize, size);
      } else {
        starts[tensor.ordinal()] = numericSize;
        numericSize = Math.addExact(numericSize, size);
      }
    }
    categorical = new short[categoricalSize];
    numeric = new float[numericSize];
  }

  public ClassicSchema schema() {
    return schema;
  }

  public Bucket bucket() {
    return bucket;
  }

  public int rows() {
    return rows;
  }

  /** 整数のテンソルをまとめた配列。 */
  public short[] categorical() {
    return categorical;
  }

  /** 実数のテンソルをまとめた配列。 */
  public float[] numeric() {
    return numeric;
  }

  /**
   * テンソルの指定した行が、配列のどこから始まるかを返す。
   *
   * @param tensor 位置を求めるテンソル
   * @param row 行の番号（0 始まり）
   * @return 整数のテンソルなら {@link #categorical()}、実数のテンソルなら {@link #numeric()} の中の位置
   */
  public int start(Tensor tensor, int row) {
    return starts[tensor.ordinal()] + Objects.checkIndex(row, rows) * rowWidths[tensor.ordinal()];
  }

  /** テンソルの 1 行の要素数。このアーキテクチャに無いテンソルは 0。 */
  public int rowWidth(Tensor tensor) {
    return rowWidths[tensor.ordinal()];
  }

  /** 指定した行を、全てのテンソルで 0（パディング）にする。 */
  void clear(int row) {
    for (Tensor tensor : TENSORS) {
      int start = start(tensor, row);
      int end = start + rowWidths[tensor.ordinal()];
      if (tensor.slab() == Tensor.Slab.CATEGORICAL) {
        Arrays.fill(categorical, start, end, (short) 0);
      } else {
        Arrays.fill(numeric, start, end, 0f);
      }
    }
  }
}

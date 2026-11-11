package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicInputs;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.Tensor;
import java.util.Arrays;
import java.util.Objects;

/**
 * ネットワークへ渡す行のバッチ。{@link ClassicInputs} の 2 本の配列（整数は INT16、実数は FLOAT32）を、ページ固定のバッファ
 * （{@link PinnedUploads}）からデバイスへ 1 回ずつコピーする。アーキテクチャが持つテンソルは、先頭から使う行の数だけを切り出した view にする。
 * view の形は単位ごとに決まり、行なら [行, 幅]、合法手なら [行, 合法手, 幅]、遷移なら [行, 合法手, 遷移, 幅]。
 *
 * <p>コピーの仕方は 2 つある。呼び出し元のスレッドのバッファから同期でコピーする（{@link #send}）か、スロットのバッファに先に書いておき
 * （{@link #write}）、後で転送のストリームに非同期のコピーを積む（{@link #enqueue}）。
 *
 * <p>配列は作ったマネージャーが持ち、順伝播の出力もこのマネージャーに付く。バッチを使い終えたらマネージャーごと閉じる。
 */
public final class ClassicBatch {

  private final ClassicSchema schema;
  private final Bucket bucket;
  private final int rows;
  private final NDManager manager;
  private final NDArray[] tensors = new NDArray[Tensor.values().length];

  /**
   * デバイスへコピーした 2 本の配列から、アーキテクチャが持つテンソルの view を作る。
   *
   * @param manager 配列を持つマネージャー
   * @param inputs 書いた入力
   * @param rows 使う行の数
   * @param categorical デバイスへコピーした整数の配列
   * @param numeric デバイスへコピーした実数の配列
   */
  private ClassicBatch(NDManager manager, ClassicInputs inputs, int rows, NDArray categorical, NDArray numeric) {
    schema = inputs.schema();
    bucket = inputs.bucket();
    this.rows = rows;
    this.manager = manager;
    for (Tensor tensor : Tensor.values()) {
      int width = inputs.rowWidth(tensor);
      if (width == 0) {
        continue;
      }
      NDArray slab = tensor.slab() == Tensor.Slab.CATEGORICAL ? categorical : numeric;
      int start = inputs.start(tensor, 0);
      tensors[tensor.ordinal()] =
          slab.get(new NDIndex().addSliceDim(start, start + (long) rows * width))
              .reshape(shape(tensor, schema, bucket, rows));
    }
  }

  /**
   * 書いた入力の 2 本の配列を、バッファからデバイスへ同期でコピーし、先頭から使う行の数だけのバッチにする。
   *
   * @param manager 配列を持つマネージャー（デバイスもこのマネージャーのもの）
   * @param uploads コピーに使うバッファ（呼び出し元のスレッドのもの）
   * @param inputs 書いた入力
   * @param rows 使う行の数（1 以上で入力の行の数以下）
   * @return 行のバッチ
   */
  public static ClassicBatch send(NDManager manager, PinnedUploads uploads, ClassicInputs inputs, int rows) {
    Objects.requireNonNull(manager, "manager");
    requireRows(inputs, rows);
    short[] categorical = inputs.categorical();
    float[] numeric = inputs.numeric();
    NDArray categoricalSlab =
        uploads.send(
            manager, DataType.INT16, new Shape(categorical.length), buffer -> buffer.asShortBuffer().put(categorical));
    NDArray numericSlab =
        uploads.send(
            manager, DataType.FLOAT32, new Shape(numeric.length), buffer -> buffer.asFloatBuffer().put(numeric));
    return new ClassicBatch(manager, inputs, rows, categoricalSlab, numericSlab);
  }

  /**
   * 書いた入力の 2 本の配列を、非同期のコピーに備えてバッファに書く。バッファは、前に積んだコピーが終わるのを待ってから書かせる。コピーは
   * {@link #enqueue} が積む。
   *
   * @param region スロットのバッファ
   * @param inputs 書いた入力
   */
  public static void write(PinnedUploads region, ClassicInputs inputs) {
    short[] categorical = inputs.categorical();
    float[] numeric = inputs.numeric();
    region.write(DataType.INT16, categorical.length).asShortBuffer().put(categorical);
    region.write(DataType.FLOAT32, numeric.length).asFloatBuffer().put(numeric);
  }

  /**
   * バッファに書いた入力のコピーをストリームに積み（ホストはコピーを待たない）、先頭から使う行の数だけのバッチを返す。バッチを読む
   * ストリームは、読む前に {@link Upload#receive} を呼ぶ。
   *
   * @param manager 配列を持つマネージャー（デバイスもこのマネージャーのもの。計算を終えるまで閉じない）
   * @param region {@link #write} で入力を書いたバッファ
   * @param inputs 書いた入力
   * @param rows 使う行の数（1 以上で入力の行の数以下）
   * @param stream コピーを積むストリーム
   * @return 積んだコピー
   */
  public static Upload enqueue(
      NDManager manager, PinnedUploads region, ClassicInputs inputs, int rows, PtStream stream) {
    Objects.requireNonNull(manager, "manager");
    requireRows(inputs, rows);
    PinnedUploads.Transfer transfer = region.enqueue(manager, Objects.requireNonNull(stream, "stream"));
    ClassicBatch batch =
        new ClassicBatch(manager, inputs, rows, transfer.array(DataType.INT16), transfer.array(DataType.FLOAT32));
    return new Upload(batch, transfer);
  }

  /**
   * アーキテクチャが持つテンソルについて、初期化に使う形を返す。行・合法手・遷移の軸は −1 にし、テンソルを宣言した順に並べる。
   *
   * @param schema アーキテクチャの入力の配置
   * @return アーキテクチャが持つテンソルの数だけの形
   */
  static Shape[] initializationShapes(ClassicSchema schema) {
    return Arrays.stream(Tensor.values())
        .filter(tensor -> tensor.in(schema.architecture()))
        .map(tensor -> unitShape(tensor, schema.width(tensor), -1, -1, -1))
        .toArray(Shape[]::new);
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

  /** 配列を持つマネージャー。 */
  public NDManager manager() {
    return manager;
  }

  /**
   * テンソルの view を返す。
   *
   * @throws IllegalArgumentException アーキテクチャがそのテンソルを持たないとき
   */
  public NDArray tensor(Tensor tensor) {
    NDArray array = tensors[tensor.ordinal()];
    if (array == null) {
      throw new IllegalArgumentException(schema.architecture() + " has no " + tensor);
    }
    return array;
  }

  private static void requireRows(ClassicInputs inputs, int rows) {
    if (rows < 1 || rows > inputs.rows()) {
      throw new IllegalArgumentException("Rows must be in [1, " + inputs.rows() + "]: " + rows);
    }
  }

  private static Shape shape(Tensor tensor, ClassicSchema schema, Bucket bucket, int rows) {
    return unitShape(tensor, schema.width(tensor), rows, bucket.actions(), bucket.transitions());
  }

  private static Shape unitShape(Tensor tensor, int width, long rows, long actions, long transitions) {
    return switch (tensor.axis()) {
      case ROW -> new Shape(rows, width);
      case ACTION -> new Shape(rows, actions, width);
      case TRANSITION -> new Shape(rows, actions, transitions, width);
    };
  }

  /** 積んだコピー（{@link #enqueue}）。バッチは、{@link #receive} を呼んだストリームの上で、呼んだ後にだけ読む。 */
  public static final class Upload {

    private final ClassicBatch batch;
    private final PinnedUploads.Transfer transfer;

    private Upload(ClassicBatch batch, PinnedUploads.Transfer transfer) {
      this.batch = batch;
      this.transfer = transfer;
    }

    /**
     * 呼び出し元のスレッドが今使っているストリームに、コピーの完了を待つ処理を積む（ホストは待たない）。コピーした配列は、そのストリームが
     * 使うものとして記録し、バッチを返す。
     *
     * @return 行のバッチ
     */
    public ClassicBatch receive() {
      transfer.waitOnStream();
      return batch;
    }
  }
}

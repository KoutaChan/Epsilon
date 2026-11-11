package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Element;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * ネットワークに渡す入力の 1 バッチ。{@link Tensor} の領域ごとの NDArray（どれも同じマネージャー・同じデバイス）と、行と手のインデックスの
 * ホスト側の写しを持つ。インデックス（{@link BatchIndex}）は次のどちらか一方で用意する。
 * その場で作ってバッファ（{@link PinnedUploads}。入力を作ったスレッドのもの）から送るか、前もって作って入力と一緒に送っておく
 * （{@link StagedBatch}）。
 *
 * <p>並び方の契約（入力のライター・HostBatch・HIP と共通）:
 *
 * <ul>
 *   <li>行の軸（局のフィールドと数値、行のインデックス）は行の順に並ぶ。席は行ごとに相対席の順で S 個、牌のトークンは行ごとに正準の色の順で
 *       T 個並ぶ（S と T は入力の形式が決める席と牌のトークンの数）。
 *   <li>イベントは行ごとに公開された順に続く。行 r のイベントは {@link RowColumn#FIRST_EVENT} 番目から {@link RowColumn#EVENTS} 個。
 *   <li>関係の属性（{@link Axis#TOKEN}）はライター（BatchPlan）と同じく、型ごとの領域を局 R 個・席 S·R 個・牌 T·R 個・イベントの順に続ける。
 *       行 r の局は r 番目、席は R + S·r 番目から、牌は (1 + S)·R + T·r 番目から、イベントは (1 + S + T)·R + FIRST_EVENT(r) 番目から始まる。
 *   <li>遷移と手は行ごとに連続して並ぶ。手のインデックスの {@link ActionColumn#ROW}・{@link ActionColumn#FIRST_TRANSITION}・
 *       {@link ActionColumn#BASE}・{@link ActionColumn#REPRESENTATIVE} はバッチの中の通し番号（無ければ −1）。行の手はラベルの合法手の順に
 *       並び、出力 s の列 k は行の k 番目の手を指す。
 *   <li>遷移のキーのコードは {@link InputFormat.Keys} の順に 1 + S + T 列ある。
 * </ul>
 *
 * <p>数値の領域（BF16）は scale で割ってから丸めた値を持ち、厳密な領域（FP32）は ExactField の scale で割った値を持つ。
 *
 * @param format 入力の形式
 * @param tensors 領域ごとの配列。形は [軸の個数, {@link InputFormat#width}]
 * @param rows 行の数
 * @param events イベントのトークンの数
 * @param transitions 遷移の数
 * @param actions 手の数
 * @param rowColumns 行のインデックスのホスト側の写し（[行, {@link RowColumn}] を 1 列に並べたもの）
 * @param actionColumns 手のインデックスのホスト側の写し（[手, {@link ActionColumn}] を 1 列に並べたもの）
 * @param uploads インデックス（{@link BatchIndex}）をその場で作って送るバッファ（前もって送ったインデックスを持つなら null）
 * @param staged 前もって作って送ったインデックス（その場で作るなら null）
 */
public record NetworkInputs(
    InputFormat format,
    Map<Tensor, NDArray> tensors,
    int rows,
    int events,
    int transitions,
    int actions,
    int[] rowColumns,
    int[] actionColumns,
    PinnedUploads uploads,
    BatchIndex staged)
    implements HostColumns {

  static {
    // HostBatch はリトルエンディアンで書き、ネットワークの入口はネイティブのバイト順で領域を読む
    if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
      throw new IllegalStateException("PRISM host batches need a little-endian machine");
    }
  }

  /** 行のインデックスの列の数。 */
  static final int ROW_WIDTH = RowColumn.values().length;

  /** 手のインデックスの列の数。 */
  static final int ACTION_WIDTH = ActionColumn.values().length;

  /**
   * 数と形を確かめる。
   *
   * @throws IllegalArgumentException 行が 0 以下か、イベントの数が負か、遷移か手が行より少ないとき。領域が欠けているか、形が数と
   *     合わないとき。バッファと前もって送ったインデックスの両方を持つか、どちらも持たないとき
   */
  public NetworkInputs {
    if (rows <= 0 || events < 0 || transitions < rows || actions < rows) {
      throw new IllegalArgumentException(
          "invalid batch counts: rows=" + rows + " transitions=" + transitions + " actions=" + actions);
    }
    Objects.requireNonNull(format, "format");
    EnumMap<Tensor, NDArray> copy = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      NDArray array = Objects.requireNonNull(tensors.get(tensor), tensor.name());
      Shape expected =
          new Shape(count(format, tensor.axis(), rows, events, transitions, actions), format.width(tensor));
      if (!array.getShape().equals(expected)) {
        throw new IllegalArgumentException(tensor + " has shape " + array.getShape() + ", expected " + expected);
      }
      copy.put(tensor, array);
    }
    tensors = Collections.unmodifiableMap(copy);
    if (rowColumns.length != rows * ROW_WIDTH || actionColumns.length != actions * ACTION_WIDTH) {
      throw new IllegalArgumentException("row or action columns do not match the counts");
    }
    if ((uploads == null) == (staged == null)) {
      throw new IllegalArgumentException("the inputs need either uploads or a staged index");
    }
  }

  /**
   * 入力のライターが書いたバッチから作る。領域を切り出して
   * {@link #of(NDManager, PinnedUploads, InputFormat, int, int, int, int, Map)} に渡す。CPU では配列がバッチのメモリをそのまま指すので、
   * 配列を閉じるまでバッチを使い回さないこと。
   *
   * @param manager 配列を作るマネージャー（デバイスもこれで決まる）
   * @param uploads インデックスと損失の入力を送るバッファ（呼び出し元のスレッドのもの）
   * @param batch 計画を確定したバッチ
   * @return 入力
   */
  public static NetworkInputs of(NDManager manager, PinnedUploads uploads, HostBatch batch) {
    BatchPlan plan = batch.plan();
    InputFormat format = plan.format();
    Map<Tensor, ByteBuffer> sections = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      int bytes = plan.count(tensor.axis()) * format.width(tensor) * tensor.element().bytes();
      sections.put(
          tensor, batch.blob().slice(plan.offset(tensor), bytes).order(ByteOrder.LITTLE_ENDIAN));
    }
    return of(
        manager,
        uploads,
        format,
        plan.rows(),
        plan.count(Axis.EVENT),
        plan.count(Axis.TRANSITION),
        plan.count(Axis.ACTION),
        sections);
  }

  /**
   * ホスト側の領域（転送用の符号化のまま、ネイティブのバイト順）から作る。HostBatch の blob から領域を切り出して渡すときに使う。
   *
   * @param manager 配列を作るマネージャー（デバイスもこれで決まる）
   * @param uploads インデックスと損失の入力を送るバッファ（呼び出し元のスレッドのもの）
   * @param format 入力の形式
   * @param rows 行の数
   * @param events イベントのトークンの数
   * @param transitions 遷移の数
   * @param actions 手の数
   * @param sections 領域ごとのバイト列（位置 0 から形の分だけ読む）
   * @return 入力
   */
  public static NetworkInputs of(
      NDManager manager,
      PinnedUploads uploads,
      InputFormat format,
      int rows,
      int events,
      int transitions,
      int actions,
      Map<Tensor, ByteBuffer> sections) {
    EnumMap<Tensor, NDArray> arrays = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      ByteBuffer section = Objects.requireNonNull(sections.get(tensor), tensor.name());
      Shape shape =
          new Shape(count(format, tensor.axis(), rows, events, transitions, actions), format.width(tensor));
      arrays.put(tensor, manager.create(section.duplicate(), shape, type(tensor.element())));
    }
    return new NetworkInputs(
        format,
        arrays,
        rows,
        events,
        transitions,
        actions,
        ints(sections.get(Tensor.ROW_COLUMN), rows * ROW_WIDTH),
        ints(sections.get(Tensor.ACTION_COLUMN), actions * ACTION_WIDTH),
        uploads,
        null);
  }

  /**
   * 軸の個数を返す（{@link BatchPlan#count(InputFormat, Axis, int, int, int, int)} と同じ）。
   *
   * @param format 入力の形式
   * @param axis 軸
   * @param rows 行の数
   * @param events イベントの数
   * @param transitions 遷移の数
   * @param actions 手の数
   * @return 個数
   */
  public static int count(InputFormat format, Axis axis, int rows, int events, int transitions, int actions) {
    return BatchPlan.count(format, axis, rows, events, transitions, actions);
  }

  /**
   * 順伝播で使うインデックスを返す。前もって送ったインデックスがあれば、それを返す。そのインデックスは、順伝播の種類と同じ C の印と
   * 補助のヘッドの教師で作ってあること。無ければ、その場で作って送る。
   *
   * @param pass 順伝播の種類
   * @return インデックス
   * @throws IllegalArgumentException 前もって送ったインデックスが、別の C の印か補助のヘッドの教師で作ったものであるとき
   */
  BatchIndex index(Pass pass) {
    if (staged == null) {
      return BatchIndex.send(this, pass);
    }
    if (!staged.builtFor(pass)) {
      throw new IllegalArgumentException("the staged index was built for another pass");
    }
    return staged;
  }

  public NDArray get(Tensor tensor) {
    return tensors.get(tensor);
  }

  @Override
  public int row(int row, RowColumn column) {
    return rowColumns[row * ROW_WIDTH + column.ordinal()];
  }

  @Override
  public int action(int action, ActionColumn column) {
    return actionColumns[action * ACTION_WIDTH + column.ordinal()];
  }

  /** 配列のマネージャー。作業用のマネージャーとしても使う。 */
  NDManager manager() {
    return tensors.get(Tensor.ROW_COLUMN).getManager();
  }

  private static DataType type(Element element) {
    return switch (element) {
      case U8 -> DataType.UINT8;
      case BF16 -> DataType.BFLOAT16;
      case FP32 -> DataType.FLOAT32;
      case I32 -> DataType.INT32;
    };
  }

  static int[] ints(ByteBuffer section, int count) {
    int[] values = new int[count];
    section.duplicate().order(ByteOrder.nativeOrder()).asIntBuffer().get(values);
    return values;
  }
}

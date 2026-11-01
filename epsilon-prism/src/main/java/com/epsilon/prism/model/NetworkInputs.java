package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.FrontierBatch;
import com.epsilon.prism.input.FrontierField;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.Element;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * 網の入力の 1 塊。{@link Tensor} の区画ごとの NDArray（同じ管理元・同じデバイス）と、前線の腕なら前線の区画、行・手の索引のホストの写し、
 * そこから作る索引と損失の入力をデバイスへ送る置き場（{@link Uploads}。入力を作ったスレッドの物）を持つ。
 *
 * <p>並びの約束（Tokenizer・HostBatch・HIP と共有する）:
 *
 * <ul>
 *   <li>行の軸（局の欄と数値、行の索引）は行の順。席は行ごとに相対席の順の 4 個、牌は行ごとに正準の色の番号 0〜33 の順の 34 個。
 *   <li>出来事は行ごとに公開の順に続け、行 r の出来事は {@link RowColumn#FIRST_EVENT} から {@link RowColumn#EVENTS} 個。
 *   <li>関係の属性（{@link Axis#TOKEN}）は書き手（BatchPlan）と同じく、型ごとの区画を局 R・席 4R・牌 34R・出来事の順に続ける。行 r の
 *       局は r、席は R + 4r から、牌は 5R + 34r から、出来事は 39R + FIRST_EVENT(r) から。
 *   <li>遷移と手は行ごとに連続。手の索引の {@link ActionColumn#ROW}・{@link ActionColumn#FIRST_TRANSITION}・{@link ActionColumn#BASE}・
 *       {@link ActionColumn#REPRESENTATIVE} は塊の中の通し番号（なしは −1）。行の手はラベルの合法手の順に並べ、出力 s の列 k は行の k
 *       番目の手。
 *   <li>遷移の鍵の符号は {@link InputSchema.KeyBlock} の順の 39 列。
 * </ul>
 *
 * <p>数値の区画（BF16）は scale で割ってから丸めた値、厳密な区画（FP32）は ExactField の scale で割った値。前線の区画は遷移の区画と同じ
 * 並びの {@link FrontierField} の列（BF16）で、前線の腕の網（{@link PrismShape#frontier()}）だけが読む。
 *
 * @param tensors 区画ごとの配列。形は [軸の個数, {@link Tensor#width()}]
 * @param frontier 前線の区画 [遷移, {@link FrontierField#WIDTH}]（前線のない入力は null）
 * @param rows 行の数
 * @param events 出来事のトークンの数
 * @param transitions 遷移の数
 * @param actions 手の数
 * @param rowColumns 行の索引のホストの写し（[行, {@link RowColumn}] を平らに）
 * @param actionColumns 手の索引のホストの写し（[手, {@link ActionColumn}] を平らに）
 * @param uploads 索引（{@link BatchIndex}）と損失の入力（{@link LossInputs}）を送る置き場
 */
public record NetworkInputs(
    Map<Tensor, NDArray> tensors,
    NDArray frontier,
    int rows,
    int events,
    int transitions,
    int actions,
    int[] rowColumns,
    int[] actionColumns,
    Uploads uploads) {

  static {
    // HostBatch はリトルエンディアン、網の入口はネイティブの順で区画を読む
    if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
      throw new IllegalStateException("PRISM host batches need a little-endian machine");
    }
  }

  /** 行の索引の列の数。 */
  static final int ROW_WIDTH = RowColumn.values().length;

  /** 手の索引の列の数。 */
  static final int ACTION_WIDTH = ActionColumn.values().length;

  /**
   * 形を確かめる。
   *
   * @throws IllegalArgumentException 区画が欠けるか、形が数と合わないとき
   */
  public NetworkInputs {
    if (rows <= 0 || events < 0 || transitions < rows || actions < rows) {
      throw new IllegalArgumentException(
          "invalid batch counts: rows=" + rows + " transitions=" + transitions + " actions=" + actions);
    }
    EnumMap<Tensor, NDArray> copy = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      NDArray array = Objects.requireNonNull(tensors.get(tensor), tensor.name());
      Shape expected = new Shape(count(tensor.axis(), rows, events, transitions, actions), tensor.width());
      if (!array.getShape().equals(expected)) {
        throw new IllegalArgumentException(tensor + " has shape " + array.getShape() + ", expected " + expected);
      }
      copy.put(tensor, array);
    }
    tensors = Collections.unmodifiableMap(copy);
    if (frontier != null && !frontier.getShape().equals(new Shape(transitions, FrontierField.WIDTH))) {
      throw new IllegalArgumentException(
          "frontier has shape " + frontier.getShape() + " for " + transitions + " transitions");
    }
    if (rowColumns.length != rows * ROW_WIDTH || actionColumns.length != actions * ACTION_WIDTH) {
      throw new IllegalArgumentException("row or action columns do not match the counts");
    }
    Objects.requireNonNull(uploads, "uploads");
  }

  /**
   * Tokenizer が書いた塊から、前線のない入力を作る（{@link #of(NDManager, Uploads, HostBatch, FrontierBatch)} の前線なし）。
   *
   * @param manager 配列を作る管理元（デバイスもここで決まる）
   * @param uploads 索引と損失の入力を送る置き場（呼び手のスレッドの物）
   * @param batch 計画を封じた塊
   * @return 入力
   */
  public static NetworkInputs of(NDManager manager, Uploads uploads, HostBatch batch) {
    return of(manager, uploads, batch, null);
  }

  /**
   * Tokenizer が書いた塊（と前線の腕なら前線の書き手が書いた塊）から作る。区画を切り出して
   * {@link #of(NDManager, Uploads, int, int, int, int, Map, ByteBuffer)} に渡す。CPU では配列が塊の領域を指すので、配列を閉じるまで
   * 塊を使い回さない。
   *
   * @param manager 配列を作る管理元（デバイスもここで決まる）
   * @param uploads 索引と損失の入力を送る置き場（呼び手のスレッドの物）
   * @param batch 計画を封じた塊
   * @param frontier 前線の塊（前線のない入力は null）
   * @return 入力
   */
  public static NetworkInputs of(
      NDManager manager, Uploads uploads, HostBatch batch, FrontierBatch frontier) {
    BatchPlan plan = batch.plan();
    Map<Tensor, ByteBuffer> sections = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      int bytes = plan.count(tensor.axis()) * tensor.width() * tensor.element().bytes();
      sections.put(
          tensor, batch.blob().slice(plan.offset(tensor), bytes).order(ByteOrder.LITTLE_ENDIAN));
    }
    return of(
        manager,
        uploads,
        plan.rows(),
        plan.count(Axis.EVENT),
        plan.count(Axis.TRANSITION),
        plan.count(Axis.ACTION),
        sections,
        frontier == null
            ? null
            : frontier.blob().slice(0, frontier.bytes()).order(ByteOrder.LITTLE_ENDIAN));
  }

  /**
   * ホストの区画（転送の符号のまま、ネイティブのバイト順）から、前線のない入力を作る。HostBatch の blob の区画を切り出して渡す口。
   *
   * @param manager 配列を作る管理元（デバイスもここで決まる）
   * @param uploads 索引と損失の入力を送る置き場（呼び手のスレッドの物）
   * @param rows 行の数
   * @param events 出来事のトークンの数
   * @param transitions 遷移の数
   * @param actions 手の数
   * @param sections 区画ごとのバイト列（位置 0 から形の分だけ読む）
   * @return 入力
   */
  public static NetworkInputs of(
      NDManager manager,
      Uploads uploads,
      int rows,
      int events,
      int transitions,
      int actions,
      Map<Tensor, ByteBuffer> sections) {
    return of(manager, uploads, rows, events, transitions, actions, sections, null);
  }

  /**
   * ホストの区画と前線の区画から作る。前線の腕では、同じ塊の FrontierBatch の blob の先頭を渡す。
   *
   * @param manager 配列を作る管理元（デバイスもここで決まる）
   * @param uploads 索引と損失の入力を送る置き場（呼び手のスレッドの物）
   * @param rows 行の数
   * @param events 出来事のトークンの数
   * @param transitions 遷移の数
   * @param actions 手の数
   * @param sections 区画ごとのバイト列（位置 0 から形の分だけ読む）
   * @param frontier 前線の区画（位置 0 から [遷移, {@link FrontierField#WIDTH}] の BF16 を読む。前線のない入力は null）
   * @return 入力
   */
  public static NetworkInputs of(
      NDManager manager,
      Uploads uploads,
      int rows,
      int events,
      int transitions,
      int actions,
      Map<Tensor, ByteBuffer> sections,
      ByteBuffer frontier) {
    EnumMap<Tensor, NDArray> arrays = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      ByteBuffer section = Objects.requireNonNull(sections.get(tensor), tensor.name());
      Shape shape = new Shape(count(tensor.axis(), rows, events, transitions, actions), tensor.width());
      arrays.put(tensor, manager.create(section.duplicate(), shape, type(tensor.element())));
    }
    NDArray frontierArray =
        frontier == null
            ? null
            : manager.create(
                frontier.duplicate(),
                new Shape(transitions, FrontierField.WIDTH),
                type(FrontierField.ELEMENT));
    return new NetworkInputs(
        arrays,
        frontierArray,
        rows,
        events,
        transitions,
        actions,
        ints(sections.get(Tensor.ROW_COLUMN), rows * ROW_WIDTH),
        ints(sections.get(Tensor.ACTION_COLUMN), actions * ACTION_WIDTH),
        uploads);
  }

  /**
   * 軸の個数を返す。
   *
   * @param axis 軸
   * @param rows 行の数
   * @param events 出来事の数
   * @param transitions 遷移の数
   * @param actions 手の数
   * @return 個数
   */
  public static int count(Axis axis, int rows, int events, int transitions, int actions) {
    return switch (axis) {
      case ROW -> rows;
      case SEAT -> rows * TokenType.SEAT.perRow();
      case TILE -> rows * TokenType.TILE.perRow();
      case EVENT -> events;
      case TOKEN -> rows * TokenType.FIXED_TOKENS + events;
      case TRANSITION -> transitions;
      case ACTION -> actions;
    };
  }

  public NDArray get(Tensor tensor) {
    return tensors.get(tensor);
  }

  public int row(int row, RowColumn column) {
    return rowColumns[row * ROW_WIDTH + column.ordinal()];
  }

  public int action(int action, ActionColumn column) {
    return actionColumns[action * ACTION_WIDTH + column.ordinal()];
  }

  /** 配列の管理元（作業の管理元にする）。 */
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

  private static int[] ints(ByteBuffer section, int count) {
    int[] values = new int[count];
    section.duplicate().order(ByteOrder.nativeOrder()).asIntBuffer().get(values);
    return values;
  }
}

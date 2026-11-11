package com.epsilon.prism.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import java.util.Objects;

/**
 * 1 つのバッチについて、行の位置と、blob の中の各テンソルの領域の位置を決める。行の位置は、判断の記録にあるイベント・遷移・合法手の数の
 * 累積和で決める。領域は {@link Tensor} の順に並べ、先頭を {@link #TENSOR_ALIGNMENT} B にそろえて、実際の数の分だけ詰める。そのため使う部分は
 * blob の先頭から切れ目なく続き、送るのは {@link #bytes} B だけで済む。
 *
 * <p>トークンの軸（{@link Axis#TOKEN}）には、局・席・牌・イベントのトークンの領域が {@link Axis} の宣言の順に続く。数は局が R 個、席が S·R 個、
 * 牌が T·R 個、イベントが N_E 個（S と T は入力の形式が決める行ごとの席と牌のトークンの数）。局・席・牌のトークンの位置は行の番号で決まり、
 * イベントのトークンの位置は {@link #firstEvent} で決まる。
 *
 * <p>{@link #clear} の後に行ごとに {@link #add} を呼び、最後に {@link #seal} で確定する。確定した後は位置を読むだけで、行は足せない。
 * 受付のスレッドが作り、符号化のスレッドは確定した後に読むだけにする。
 */
public final class BatchPlan {

  /** バッチの中で、各テンソルの領域の先頭をそろえるバイト数。HostBatch と HIP 側の表はこの値を使う。 */
  public static final int TENSOR_ALIGNMENT = 256;

  private static final Tensor[] TENSORS = Tensor.values();
  private static final Axis[] AXES = Axis.values();
  private static final int OPEN = -1;

  private final InputFormat format;
  private final int[] rowBytes;
  private final Capacity capacity;
  private final int[] firstEvent;
  private final int[] firstTransition;
  private final int[] firstAction;
  private final int[] offsets = new int[TENSORS.length];
  private int rows;
  private int bytes = OPEN;

  /**
   * 1 つのバッチの上限。各領域の大きさの上限はここから決まり、{@link HostBatch} はこの大きさの blob を持つ。どんな行でも 1 行は必ず入るように、
   * ルールで決まる 1 行の上限より小さい値は受け付けない。
   *
   * @param rows 行の数
   * @param events イベントのトークンの数（全ての行の和）
   * @param transitions 遷移の数（全ての行の和）
   * @param actions 合法手の数（全ての行の和）
   */
  public record Capacity(int rows, int events, int transitions, int actions) {

    /** 推論での既定の行の上限。 */
    private static final int ROWS = 4096;

    /** 推論での既定のトークンの上限（局・席・牌・イベントのトークンの和）。 */
    private static final int TOKENS = 393_216;

    /**
     * 上限を確かめる。
     *
     * @throws IllegalArgumentException 上限が小さく、1 行が入りきらないことがあるとき
     */
    public Capacity {
      if (rows < 1
          || events < Limits.MAX_EVENT_TOKENS
          || transitions < Limits.MAX_TRANSITIONS
          || actions < Limits.MAX_LEGAL_ACTIONS) {
        throw new IllegalArgumentException(
            "capacity cannot hold one row: "
                + rows
                + " rows, "
                + events
                + " events, "
                + transitions
                + " transitions, "
                + actions
                + " actions");
      }
    }

    /**
     * 推論での既定の上限（行 4096、トークン 393,216、遷移 98,304、合法手 131,072）を返す。イベントの上限は、トークンの上限から局・席・牌の
     * トークンの分を引いた数にする。
     *
     * @param format 入力の形式
     * @return 上限
     */
    public static Capacity standard(InputFormat format) {
      return new Capacity(ROWS, TOKENS - format.fixedTokens() * ROWS, 98_304, 131_072);
    }
  }

  /**
   * 空の計画を作る。
   *
   * @param format 入力の形式（各テンソルの幅と、行ごとの席と牌のトークンの数）
   * @param capacity 1 つのバッチの上限
   */
  public BatchPlan(InputFormat format, Capacity capacity) {
    this.format = Objects.requireNonNull(format, "format");
    this.capacity = Objects.requireNonNull(capacity, "capacity");
    rowBytes = rowBytes(format);
    firstEvent = new int[capacity.rows() + 1];
    firstTransition = new int[capacity.rows() + 1];
    firstAction = new int[capacity.rows() + 1];
  }

  /**
   * 行・イベント・遷移・合法手を上限まで詰めたときの blob の大きさを返す。{@link HostBatch} の blob はこれ以上の大きさにする。
   *
   * @param format 入力の形式
   * @param capacity 1 つのバッチの上限
   * @return バイト数
   */
  public static int maxBytes(InputFormat format, Capacity capacity) {
    return layout(
        format,
        rowBytes(format),
        capacity.rows(),
        capacity.events(),
        capacity.transitions(),
        capacity.actions(),
        null);
  }

  /**
   * 入力の形式を返す。
   *
   * @return 形式
   */
  public InputFormat format() {
    return format;
  }

  /**
   * 1 つのバッチの上限を返す。
   *
   * @return 上限
   */
  public Capacity capacity() {
    return capacity;
  }

  /** 全ての行を捨てて、行を足せる状態に戻す。 */
  public void clear() {
    rows = 0;
    bytes = OPEN;
  }

  /**
   * 行を足す。
   *
   * @param o 行の判断の記録（ヘッダーにあるイベント・遷移・合法手の数だけを読む）
   * @return 行の番号。上限を超えるなら −1（行は足さないので、呼び出し元はその行を次のバッチに回す）
   * @throws IllegalStateException 確定した後のとき
   */
  public int add(Observation o) {
    requireOpen();
    int events = o.publicEvents();
    int transitions = o.transitions();
    int actions = o.legalActions();
    if (rows == capacity.rows()
        || firstEvent[rows] + events > capacity.events()
        || firstTransition[rows] + transitions > capacity.transitions()
        || firstAction[rows] + actions > capacity.actions()) {
      return -1;
    }
    firstEvent[rows + 1] = firstEvent[rows] + events;
    firstTransition[rows + 1] = firstTransition[rows] + transitions;
    firstAction[rows + 1] = firstAction[rows] + actions;
    return rows++;
  }

  /**
   * 行を足し終えて確定し、各テンソルの領域の位置を決める。
   *
   * @return 使う blob のバイト数
   * @throws IllegalStateException 確定した後のとき、または行が 1 つもないとき
   */
  public int seal() {
    requireOpen();
    if (rows == 0) {
      throw new IllegalStateException("batch plan has no rows");
    }
    bytes =
        layout(
            format, rowBytes, rows, firstEvent[rows], firstTransition[rows], firstAction[rows], offsets);
    return bytes;
  }

  /**
   * 確定したかを返す。
   *
   * @return 確定したなら true
   */
  public boolean sealed() {
    return bytes != OPEN;
  }

  /**
   * 使う blob のバイト数を返す。
   *
   * @return バイト数
   * @throws IllegalStateException 確定する前のとき
   */
  public int bytes() {
    requireSealed();
    return bytes;
  }

  /**
   * テンソルの領域の先頭を返す。
   *
   * @param tensor テンソル
   * @return blob の先頭からのバイトの位置
   * @throws IllegalStateException 確定する前のとき
   */
  public int offset(Tensor tensor) {
    requireSealed();
    return offsets[tensor.ordinal()];
  }

  /**
   * 行の数を返す。
   *
   * @return 行の数
   */
  public int rows() {
    return rows;
  }

  /**
   * 軸の要素の数を返す。
   *
   * @param axis 軸
   * @return 要素の数
   */
  public int count(Axis axis) {
    return count(format, axis, rows, firstEvent[rows], firstTransition[rows], firstAction[rows]);
  }

  /**
   * トークンの軸の中で、その種類のトークンの領域が始まる位置を返す。
   *
   * @param axis トークンの種類を表す軸（{@link Axis#ROW} は局、ほかは {@link Axis#SEAT}・{@link Axis#TILE}・{@link Axis#EVENT}）
   * @return トークンの軸での位置
   * @throws IllegalArgumentException トークンの種類を表す軸でないとき
   */
  public int tokenStart(Axis axis) {
    if (axis.ordinal() >= Axis.TOKEN.ordinal()) {
      throw new IllegalArgumentException("not a token type: " + axis);
    }
    int start = 0;
    for (int before = 0; before < axis.ordinal(); before++) {
      start += count(AXES[before]);
    }
    return start;
  }

  /**
   * 行の最初のイベントのトークンの番号を返す。
   *
   * @param row 行の番号
   * @return イベントの軸での位置
   */
  public int firstEvent(int row) {
    return firstEvent[Objects.checkIndex(row, rows)];
  }

  /**
   * 行のイベントのトークンの数を返す。
   *
   * @param row 行の番号
   * @return イベントの数
   */
  public int events(int row) {
    return firstEvent[Objects.checkIndex(row, rows) + 1] - firstEvent[row];
  }

  /**
   * 行の最初の遷移の番号を返す。
   *
   * @param row 行の番号
   * @return 遷移の軸での位置
   */
  public int firstTransition(int row) {
    return firstTransition[Objects.checkIndex(row, rows)];
  }

  /**
   * 行の遷移の数を返す。
   *
   * @param row 行の番号
   * @return 遷移の数
   */
  public int transitions(int row) {
    return firstTransition[Objects.checkIndex(row, rows) + 1] - firstTransition[row];
  }

  /**
   * 行の最初の合法手の番号を返す。
   *
   * @param row 行の番号
   * @return 合法手の軸での位置
   */
  public int firstAction(int row) {
    return firstAction[Objects.checkIndex(row, rows)];
  }

  /**
   * 行の合法手の数を返す。
   *
   * @param row 行の番号
   * @return 合法手の数
   */
  public int actions(int row) {
    return firstAction[Objects.checkIndex(row, rows) + 1] - firstAction[row];
  }

  /** 各テンソルの領域を宣言の順に、先頭をそろえて詰め、末尾の位置を返す。{@code offsets} が null でなければ、各領域の先頭をそこに書く。 */
  private static int layout(
      InputFormat format, int[] rowBytes, int rows, int events, int transitions, int actions, int[] offsets) {
    int cursor = 0;
    for (Tensor tensor : TENSORS) {
      cursor = align(cursor);
      if (offsets != null) {
        offsets[tensor.ordinal()] = cursor;
      }
      int elements = count(format, tensor.axis(), rows, events, transitions, actions);
      cursor = Math.addExact(cursor, Math.multiplyExact(elements, rowBytes[tensor.ordinal()]));
    }
    return cursor;
  }

  /**
   * 軸の要素の数を返す。
   *
   * @param format 入力の形式
   * @param axis 軸
   * @param rows 行の数
   * @param events イベントのトークンの数
   * @param transitions 遷移の数
   * @param actions 合法手の数
   * @return 要素の数
   */
  public static int count(
      InputFormat format, Axis axis, int rows, int events, int transitions, int actions) {
    return switch (axis) {
      case ROW -> rows;
      case SEAT -> format.seats() * rows;
      case TILE -> format.tiles() * rows;
      case EVENT -> events;
      case TOKEN -> format.fixedTokens() * rows + events;
      case TRANSITION -> transitions;
      case ACTION -> actions;
    };
  }

  private static int align(int offset) {
    int alignment = TENSOR_ALIGNMENT;
    return Math.addExact(offset, alignment - 1) / alignment * alignment;
  }

  /** テンソルごとの、軸の要素 1 つあたりのバイト数。 */
  private static int[] rowBytes(InputFormat format) {
    int[] bytes = new int[TENSORS.length];
    for (Tensor tensor : TENSORS) {
      bytes[tensor.ordinal()] = format.width(tensor) * tensor.element().bytes();
    }
    return bytes;
  }

  private void requireOpen() {
    if (bytes != OPEN) {
      throw new IllegalStateException("batch plan is sealed");
    }
  }

  private void requireSealed() {
    if (bytes == OPEN) {
      throw new IllegalStateException("batch plan is not sealed");
    }
  }
}

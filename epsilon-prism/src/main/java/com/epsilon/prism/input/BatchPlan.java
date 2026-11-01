package com.epsilon.prism.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.observe.Observation;
import java.util.Objects;

/**
 * 1 塊の行の位置と、blob の中の区画の位置を決める。行は記録 O の形の数（出来事・遷移・手）の前置和で並べ、区画は {@link Tensor} の順に
 * {@link InputSchema#TENSOR_ALIGNMENT} B にそろえて、実際の数で詰める。だから使う部分は blob の先頭から続き、送るのは {@link #bytes} B だけになる。
 *
 * <p>軸 {@link Axis#TOKEN} は、局・席・牌・出来事のトークンの区画を {@link Axis} の宣言の順に続けたもの（局 R 個、席 4R 個、牌 34R 個、出来事
 * N_E 個）。行 r の局・席・牌のトークンの位置は行番号で決まり、出来事のトークンの位置は {@link #firstEvent} で決まる。
 *
 * <p>使い方は {@link #clear} → 行ごとに {@link #add} → {@link #seal}。封じた後は位置を読むだけで、行を足せない。受付のスレッドが作り、符号化の
 * スレッドは封じた後に読むだけにする。
 */
public final class BatchPlan {

  /** 行ごとの局・席・牌のトークンの数。 */
  public static final int FIXED_TOKENS = 1 + GameState.NUM_PLAYERS + Tile.NUM_TILE_TYPES;

  private static final Tensor[] TENSORS = Tensor.values();
  private static final Axis[] AXES = Axis.values();
  private static final int[] ROW_BYTES = rowBytes();
  private static final int OPEN = -1;

  private final Capacity capacity;
  private final int[] firstEvent;
  private final int[] firstTransition;
  private final int[] firstAction;
  private final int[] offsets = new int[TENSORS.length];
  private int rows;
  private int bytes = OPEN;

  /**
   * 1 塊の上限。区画の大きさの上限はここから決まり、{@link HostBatch} はこの大きさの blob を持つ。1 行は必ず入るように、規則の上限より
   * 小さい値は受け付けない。
   *
   * @param rows 行の数
   * @param events 出来事トークンの数（全ての行の和）
   * @param transitions 遷移の数（全ての行の和）
   * @param actions 合法手の数（全ての行の和）
   */
  public record Capacity(int rows, int events, int transitions, int actions) {

    /** 既定の上限（行 4096、トークン 393,216、遷移 98,304、手 131,072）。出来事はトークンから局・席・牌の分を引いた数。 */
    public static final Capacity DEFAULT =
        new Capacity(4096, 393_216 - FIXED_TOKENS * 4096, 98_304, 131_072);

    /**
     * 上限を確かめる。
     *
     * @throws IllegalArgumentException 1 行が入らない上限のとき
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
  }

  /**
   * 空の計画を作る。
   *
   * @param capacity 1 塊の上限
   */
  public BatchPlan(Capacity capacity) {
    this.capacity = Objects.requireNonNull(capacity, "capacity");
    firstEvent = new int[capacity.rows() + 1];
    firstTransition = new int[capacity.rows() + 1];
    firstAction = new int[capacity.rows() + 1];
  }

  /**
   * 上限の行・出来事・遷移・手で詰めたときの blob の大きさを返す。{@link HostBatch} の blob はこれ以上にする。
   *
   * @param capacity 1 塊の上限
   * @return バイト数
   */
  public static int maxBytes(Capacity capacity) {
    return layout(
        capacity.rows(), capacity.events(), capacity.transitions(), capacity.actions(), null);
  }

  /**
   * 1 塊の上限を返す。
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
   * @param o 行の記録 O（頭の形の数だけを読む）
   * @return 行の番号。上限を超えるなら −1（行は足さず、呼び手は次の塊に回す）
   * @throws IllegalStateException 封じた後のとき
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
   * 行を足し終え、区画の位置を決める。
   *
   * @return 使う blob のバイト数
   * @throws IllegalStateException 封じた後か、行がないとき
   */
  public int seal() {
    requireOpen();
    if (rows == 0) {
      throw new IllegalStateException("batch plan has no rows");
    }
    bytes = layout(rows, firstEvent[rows], firstTransition[rows], firstAction[rows], offsets);
    return bytes;
  }

  /**
   * 封じたかを返す。
   *
   * @return 封じたなら true
   */
  public boolean sealed() {
    return bytes != OPEN;
  }

  /**
   * 使う blob のバイト数を返す。
   *
   * @return バイト数
   * @throws IllegalStateException 封じる前のとき
   */
  public int bytes() {
    requireSealed();
    return bytes;
  }

  /**
   * 区画の先頭を返す。
   *
   * @param tensor テンソル
   * @return blob の中のバイトの位置
   * @throws IllegalStateException 封じる前のとき
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
    return count(axis, rows, firstEvent[rows], firstTransition[rows], firstAction[rows]);
  }

  /**
   * トークンの軸の中で、型ごとの区画の先頭を返す。
   *
   * @param axis トークンの型の軸（{@link Axis#ROW} は局、{@link Axis#SEAT}・{@link Axis#TILE}・{@link Axis#EVENT}）
   * @return トークンの軸の位置
   * @throws IllegalArgumentException トークンの型の軸でないとき
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
   * 行の最初の出来事トークンの番号を返す。
   *
   * @param row 行の番号
   * @return 出来事の軸の位置
   */
  public int firstEvent(int row) {
    return firstEvent[Objects.checkIndex(row, rows)];
  }

  /**
   * 行の出来事トークンの数を返す。
   *
   * @param row 行の番号
   * @return 出来事の数
   */
  public int events(int row) {
    return firstEvent[Objects.checkIndex(row, rows) + 1] - firstEvent[row];
  }

  /**
   * 行の最初の遷移の番号を返す。
   *
   * @param row 行の番号
   * @return 遷移の軸の位置
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
   * @return 手の軸の位置
   */
  public int firstAction(int row) {
    return firstAction[Objects.checkIndex(row, rows)];
  }

  /**
   * 行の合法手の数を返す。
   *
   * @param row 行の番号
   * @return 手の数
   */
  public int actions(int row) {
    return firstAction[Objects.checkIndex(row, rows) + 1] - firstAction[row];
  }

  /** 区画を宣言の順にそろえて詰め、終わりの位置を返す。offsets が null でなければ区画の先頭を書く。 */
  private static int layout(int rows, int events, int transitions, int actions, int[] offsets) {
    int cursor = 0;
    for (Tensor tensor : TENSORS) {
      cursor = align(cursor);
      if (offsets != null) {
        offsets[tensor.ordinal()] = cursor;
      }
      int elements = count(tensor.axis(), rows, events, transitions, actions);
      cursor = Math.addExact(cursor, Math.multiplyExact(elements, ROW_BYTES[tensor.ordinal()]));
    }
    return cursor;
  }

  private static int count(Axis axis, int rows, int events, int transitions, int actions) {
    return switch (axis) {
      case ROW -> rows;
      case SEAT -> GameState.NUM_PLAYERS * rows;
      case TILE -> Tile.NUM_TILE_TYPES * rows;
      case EVENT -> events;
      case TOKEN -> FIXED_TOKENS * rows + events;
      case TRANSITION -> transitions;
      case ACTION -> actions;
    };
  }

  private static int align(int offset) {
    int alignment = InputSchema.TENSOR_ALIGNMENT;
    return Math.addExact(offset, alignment - 1) / alignment * alignment;
  }

  /** テンソルの軸 1 つあたりのバイト数（準備のときに一度だけ作る）。 */
  private static int[] rowBytes() {
    int[] bytes = new int[TENSORS.length];
    for (Tensor tensor : TENSORS) {
      bytes[tensor.ordinal()] = tensor.width() * tensor.element().bytes();
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

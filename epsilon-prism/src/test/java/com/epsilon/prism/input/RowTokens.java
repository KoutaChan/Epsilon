package com.epsilon.prism.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.engine.GoldenHash;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.observe.Observation;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * 網の入力の 1 行のハッシュ。塊の中の位置に依らない形にする（複数の行の塊と 1 行の塊で同じ行は同じ値）。
 *
 * <p>行の形: {@link Tensor} の順に、行の要素（局 1・席 4・牌 34・出来事・遷移・手。トークンの軸は局・席・牌・出来事のトークンをこの順につないだ
 * もの）のバイトを混ぜる。索引の列（{@link Tensor#ACTION_COLUMN}・{@link Tensor#ROW_COLUMN}）は塊の中の絶対の番号なので、行の最初の番号を
 * 引いて行の中の番号にする（{@link ActionColumn#BASE} の −1 はそのまま）。最後に手ごとの p0 の特徴（送らないホストの配列）を混ぜる。
 *
 * <p>道具はスレッド安全でない（スレッドごとに 1 つ）。{@link #row} は塊を読むだけ。
 */
public final class RowTokens {
  /** 1 行だけの塊の上限。 */
  private static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);

  /** トークンの軸をつなぐ型の順（{@link BatchPlan} の区画の順）。 */
  private static final Axis[] TOKEN_TYPES = {Axis.ROW, Axis.SEAT, Axis.TILE, Axis.EVENT};

  private final Observation observation = new Observation();
  private final Tokenizer tokenizer = new Tokenizer();
  private final InputWriter writer = new InputWriter();
  private final HostBatch single =
      new HostBatch(ONE_ROW, ByteBuffer.allocate(BatchPlan.maxBytes(ONE_ROW)));

  /**
   * O を 1 行の塊に {@link Tokenizer} で書いた入力のハッシュ。
   *
   * @param words O を持つ配列
   * @param base O の先頭
   * @return {@link #row} と同じ形のハッシュ
   */
  public long tokenize(long[] words, int base) {
    BatchPlan plan = single.plan();
    plan.clear();
    if (plan.add(observation.bind(words, base)) != 0) {
      throw new IllegalStateException("one observation does not fit a one-row batch");
    }
    plan.seal();
    tokenizer.write(observation, writer.bind(single, 0));
    return row(single, 0);
  }

  /**
   * 封じて書き終えた塊の 1 行のハッシュ（行の中の位置に直した形）。
   *
   * @param batch 塊
   * @param row 行の番号
   * @return ハッシュ
   */
  public static long row(HostBatch batch, int row) {
    BatchPlan plan = batch.plan();
    ByteBuffer blob = batch.blob().duplicate().order(ByteOrder.LITTLE_ENDIAN);
    GoldenHash hash = new GoldenHash();
    for (Tensor tensor : Tensor.values()) {
      int stride = tensor.width() * tensor.element().bytes();
      int offset = plan.offset(tensor);
      switch (tensor) {
        case ACTION_COLUMN -> actionColumns(hash, blob, offset, plan, row);
        case ROW_COLUMN -> rowColumns(hash, blob, offset, plan, row);
        default -> {
          if (tensor.axis() == Axis.TOKEN) {
            for (Axis type : TOKEN_TYPES) {
              bytes(hash, blob, offset + (plan.tokenStart(type) + first(plan, type, row)) * stride,
                  count(plan, type, row) * stride);
            }
          } else {
            bytes(hash, blob, offset + first(plan, tensor.axis(), row) * stride,
                count(plan, tensor.axis(), row) * stride);
          }
        }
      }
    }
    int firstAction = plan.firstAction(row);
    for (int action = 0; action < plan.actions(row); action++) {
      hash.add(batch.priorShanten(firstAction + action)).add(batch.priorUkeire(firstAction + action));
    }
    return hash.value();
  }

  /** 軸の上の、行の最初の要素の番号。 */
  private static int first(BatchPlan plan, Axis axis, int row) {
    return switch (axis) {
      case ROW -> row;
      case SEAT -> GameState.NUM_PLAYERS * row;
      case TILE -> Tile.NUM_TILE_TYPES * row;
      case EVENT -> plan.firstEvent(row);
      case TRANSITION -> plan.firstTransition(row);
      case ACTION -> plan.firstAction(row);
      case TOKEN -> throw new IllegalArgumentException("the token axis has no single range per row");
    };
  }

  /** 軸の上の、行の要素の数。 */
  private static int count(BatchPlan plan, Axis axis, int row) {
    return switch (axis) {
      case ROW -> 1;
      case SEAT -> GameState.NUM_PLAYERS;
      case TILE -> Tile.NUM_TILE_TYPES;
      case EVENT -> plan.events(row);
      case TRANSITION -> plan.transitions(row);
      case ACTION -> plan.actions(row);
      case TOKEN -> throw new IllegalArgumentException("the token axis has no single range per row");
    };
  }

  /** 手の索引の列（行・遷移・手の番号を行の中の番号に直す）。 */
  private static void actionColumns(GoldenHash hash, ByteBuffer blob, int offset, BatchPlan plan, int row) {
    int width = Tensor.ACTION_COLUMN.width();
    int firstAction = plan.firstAction(row);
    for (int action = firstAction; action < firstAction + plan.actions(row); action++) {
      for (ActionColumn column : ActionColumn.values()) {
        int value = blob.getInt(offset + (action * width + column.ordinal()) * Integer.BYTES);
        hash.add(switch (column) {
          case ROW -> value - row;
          case FIRST_TRANSITION -> value - plan.firstTransition(row);
          case BASE -> value < 0 ? value : value - firstAction;
          case REPRESENTATIVE -> value - firstAction;
          case TRANSITIONS, ROLE -> value;
        });
      }
    }
  }

  /** 行の索引の列（出来事・手・遷移の最初の番号を 0 に直す）。 */
  private static void rowColumns(GoldenHash hash, ByteBuffer blob, int offset, BatchPlan plan, int row) {
    int width = Tensor.ROW_COLUMN.width();
    for (RowColumn column : RowColumn.values()) {
      int value = blob.getInt(offset + (row * width + column.ordinal()) * Integer.BYTES);
      hash.add(switch (column) {
        case FIRST_EVENT -> value - plan.firstEvent(row);
        case FIRST_ACTION -> value - plan.firstAction(row);
        case FIRST_TRANSITION -> value - plan.firstTransition(row);
        case EVENTS, ACTIONS, TRANSITIONS, DECISION_KIND -> value;
      });
    }
  }

  /** 区画の範囲のバイト（長さと、8 バイトずつの語、残りの 1 バイトずつ）。 */
  private static void bytes(GoldenHash hash, ByteBuffer blob, int from, int length) {
    hash.add(length);
    int words = length / Long.BYTES;
    for (int word = 0; word < words; word++) hash.add(blob.getLong(from + word * Long.BYTES));
    for (int at = words * Long.BYTES; at < length; at++) hash.add(blob.get(from + at));
  }
}

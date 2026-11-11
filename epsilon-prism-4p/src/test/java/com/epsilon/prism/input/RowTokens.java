package com.epsilon.prism.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism4p.input.InputSchema;
import com.epsilon.prism4p.input.InputWriter;
import com.epsilon.prism4p.input.Tokenizer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * ネットワークの入力の 1 行から作るハッシュ。バッチの中の位置に依らない形にするので、複数の行のバッチでも 1 行のバッチでも、同じ行は
 * 同じ値になる。
 *
 * <p>混ぜ方: {@link Tensor} の順に、その行の要素のバイトを混ぜる。要素は局 1、席 4、牌 34、イベント、遷移、手で、トークンの軸は局・席・牌・
 * イベントのトークンをこの順につないだもの。インデックスの列（{@link Tensor#ACTION_COLUMN}・{@link Tensor#ROW_COLUMN}）はバッチの中の
 * 絶対の番号なので、行の最初の番号を引いて行の中の番号に直す（{@link ActionColumn#BASE} の −1 はそのまま）。最後に、手ごとの p0 の特徴
 * （GPU に送らないホストの配列）を混ぜる。
 *
 * <p>このヘルパーはスレッドセーフでないので、スレッドごとに 1 つ使う。{@link #row} はバッチを読むだけ。
 */
public final class RowTokens {
  /** 1 行だけのバッチの容量。 */
  private static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);

  private static final int SEATS = TENHOU_4P.seats();

  /** トークンの軸につなぐ型の順。{@link BatchPlan} の中の領域の順と同じ。 */
  private static final Axis[] TOKEN_TYPES = {Axis.ROW, Axis.SEAT, Axis.TILE, Axis.EVENT};

  private final Observation observation = new Observation(TENHOU_4P);
  private final Tokenizer tokenizer = new Tokenizer();
  private final InputWriter writer = new InputWriter();
  private final HostBatch single =
      new HostBatch(InputSchema.FORMAT, ONE_ROW, ByteBuffer.allocate(BatchPlan.maxBytes(InputSchema.FORMAT, ONE_ROW)));

  /**
   * O を {@link Tokenizer} で 1 行のバッチに書き、その入力のハッシュを返す。
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
   * 確定して書き終えたバッチの 1 行から、行の中の位置に直した形でハッシュを作る。
   *
   * @param batch バッチ
   * @param row 行の番号
   * @return ハッシュ
   */
  public static long row(HostBatch batch, int row) {
    BatchPlan plan = batch.plan();
    ByteBuffer blob = batch.blob().duplicate().order(ByteOrder.LITTLE_ENDIAN);
    Mix hash = new Mix();
    for (Tensor tensor : Tensor.values()) {
      int stride = InputSchema.FORMAT.width(tensor) * tensor.element().bytes();
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

  /** その軸で、行の最初の要素が何番目か。 */
  private static int first(BatchPlan plan, Axis axis, int row) {
    return switch (axis) {
      case ROW -> row;
      case SEAT -> SEATS * row;
      case TILE -> Tiles.KINDS * row;
      case EVENT -> plan.firstEvent(row);
      case TRANSITION -> plan.firstTransition(row);
      case ACTION -> plan.firstAction(row);
      case TOKEN -> throw new IllegalArgumentException("the token axis has no single range per row");
    };
  }

  /** その軸での、行の要素の数。 */
  private static int count(BatchPlan plan, Axis axis, int row) {
    return switch (axis) {
      case ROW -> 1;
      case SEAT -> SEATS;
      case TILE -> Tiles.KINDS;
      case EVENT -> plan.events(row);
      case TRANSITION -> plan.transitions(row);
      case ACTION -> plan.actions(row);
      case TOKEN -> throw new IllegalArgumentException("the token axis has no single range per row");
    };
  }

  /** 手のインデックスの列を混ぜる。行・遷移・手の番号は、行の中の番号に直す。 */
  private static void actionColumns(Mix hash, ByteBuffer blob, int offset, BatchPlan plan, int row) {
    int width = InputSchema.FORMAT.width(Tensor.ACTION_COLUMN);
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

  /** 行のインデックスの列を混ぜる。イベント・手・遷移の最初の番号は、0 からに直す。 */
  private static void rowColumns(Mix hash, ByteBuffer blob, int offset, BatchPlan plan, int row) {
    int width = InputSchema.FORMAT.width(Tensor.ROW_COLUMN);
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

  /** 領域の範囲のバイトを混ぜる。長さ、8 バイトずつのワード、残りの 1 バイトずつの順に混ぜる。 */
  private static void bytes(Mix hash, ByteBuffer blob, int from, int length) {
    hash.add(length);
    int words = length / Long.BYTES;
    for (int word = 0; word < words; word++) hash.add(blob.getLong(from + word * Long.BYTES));
    for (int at = words * Long.BYTES; at < length; at++) hash.add(blob.get(from + at));
  }

  /** テスト用の 64 bit のハッシュ。値を順に混ぜ、順序も結果に効くようにする（h ← fmix64(h + v·K)、h の初期値は 0 でない定数）。 */
  private static final class Mix {
    private static final long K = 0x9E37_79B9_7F4A_7C15L;
    private long state = 0x6A09_E667_F3BC_C908L;

    long value() {
      return state;
    }

    Mix add(long value) {
      long h = state + value * K;
      h ^= h >>> 33;
      h *= 0xFF51_AFD7_ED55_8CCDL;
      h ^= h >>> 33;
      h *= 0xC4CE_B9FE_1A85_EC53L;
      h ^= h >>> 33;
      state = h;
      return this;
    }

    Mix add(int value) {
      return add((long) value);
    }

    Mix add(float value) {
      return add(Float.floatToRawIntBits(value));
    }
  }
}

package com.epsilon.prism.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.Bfloat16;
import com.epsilon.prism.codec.UnsignedBytes;
import com.epsilon.prism.input.FieldSpec.Coding;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.BoundedField;
import com.epsilon.prism.input.InputSchema.DiscardExact;
import com.epsilon.prism.input.InputSchema.EventField;
import com.epsilon.prism.input.InputSchema.EventScalar;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.RelationAttribute;
import com.epsilon.prism.input.InputSchema.RiichiExact;
import com.epsilon.prism.input.InputSchema.RoundField;
import com.epsilon.prism.input.InputSchema.RoundScalar;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.SeatScalar;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.input.InputSchema.TileField;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.InputSchema.TransitionScalar;
import com.epsilon.prism.input.InputSchema.WinExact;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * 1 行の網の入力を {@link HostBatch} に書く型つきの書き手。{@link #bind} で塊の中の行に付け替えて使い回し、書くときに割り当てをしない。符号化の
 * スレッドに 1 つ持つ（{@link Tokenizer} と対）。
 *
 * <p>全てのテンソルの行の分を下書きに書き、{@link #commit} で塊の区画へまとめて写す。写す前に、行ごとに 1 回、鍵の符号を区画の符号の範囲で
 * 確かめる。下書きは付け替えのときに行の分を 0 にするので、書かない要素は 0 になり、使い回した塊に前の値は残らない。途中で例外になった行は塊に
 * 何も残さない。
 *
 * <p>カテゴリの欄の口は書き方（{@link Coding}）ごとに分け、口の中で欄の書き方と値の enum の型を照らして {@link Ids} を通す。語彙は書く口で
 * 確かめる（{@link TokenLayout#requireValid} と同じ「値 &lt; 語彙」）。2 値・番号・enum の口は作りから語彙を超えないので、数え上げの口だけが
 * 欄の語彙と比べる。下書きにはこの口からしか書かないので、写す前に下書きを語彙で見直さない。数値は欄の
 * {@link InputSchema.ScalarField#scale} で、厳密な欄は {@link InputSchema.ExactField#scale} で割ってから書く。席・牌・出来事・遷移・手の番号は
 * 行の中の番号で、塊の中の位置への変換はこの書き手が {@link BatchPlan} から行う。
 */
public final class InputWriter {

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int TILES = Tile.NUM_TILE_TYPES;
  private static final int ROUND_SCALARS = Tensor.ROUND_SCALAR.width();
  private static final int SEAT_SCALARS = Tensor.SEAT_SCALAR.width();
  private static final int EVENT_SCALARS = Tensor.EVENT_SCALAR.width();
  private static final int TRANSITION_SCALARS = Tensor.TRANSITION_SCALAR.width();
  private static final int DISCARD_EXACTS = Tensor.DISCARD_EXACT.width();
  private static final int ACTION_EXACTS = Tensor.ACTION_EXACT.width();
  private static final int ACTION_COLUMNS = Tensor.ACTION_COLUMN.width();
  private static final int ROW_COLUMNS = Tensor.ROW_COLUMN.width();
  private static final int KEYS = KeyBlock.KEY_COUNT;
  private static final long ALL_KEYS = (1L << KEYS) - 1;
  private static final int NONE = -1;
  private static final Axis[] AXES = Axis.values();

  /** 関係の属性の下書きの中で、型ごとのトークンの先頭（局・席・牌・出来事の順。出来事の前の型は数が決まっている）。 */
  private static final int[] RELATION_START = relationStarts();

  /** 鍵の符号の下書きの添字ごとの、区画の最初の符号（遷移の数だけ鍵の順を繰り返す）。 */
  private static final byte[] CODE_FLOOR = codeBounds(false);

  /** 同じく、区画の最後の符号。 */
  private static final byte[] CODE_CEILING = codeBounds(true);

  private final TokenWriter<RoundField> round =
      new TokenWriter<>(InputSchema.ROUND, RoundField.class, 1);
  private final TokenWriter<SeatField> seat =
      new TokenWriter<>(InputSchema.SEAT, SeatField.class, SEATS);
  private final TokenWriter<TileField> tile =
      new TokenWriter<>(InputSchema.TILE, TileField.class, TILES);
  private final TokenWriter<EventField> event =
      new TokenWriter<>(InputSchema.EVENT, EventField.class, Limits.MAX_EVENT_TOKENS);
  private final TokenWriter<TransitionField> transition =
      new TokenWriter<>(InputSchema.TRANSITION, TransitionField.class, Limits.MAX_TRANSITIONS);
  private final TokenWriter<RelationAttribute> relation =
      new TokenWriter<>(
          InputSchema.RELATION,
          RelationAttribute.class,
          BatchPlan.FIXED_TOKENS + Limits.MAX_EVENT_TOKENS);
  private final short[] roundScalars = new short[ROUND_SCALARS];
  private final short[] seatScalars = new short[SEATS * SEAT_SCALARS];
  private final short[] eventScalars = new short[Limits.MAX_EVENT_TOKENS * EVENT_SCALARS];
  private final short[] transitionScalars = new short[Limits.MAX_TRANSITIONS * TRANSITION_SCALARS];
  private final byte[] codes = new byte[Limits.MAX_TRANSITIONS * KEYS];
  private final long[] keysWritten = new long[Limits.MAX_TRANSITIONS];
  private final float[] discardExact = new float[Limits.MAX_TRANSITIONS * DISCARD_EXACTS];
  private final float[] actionExact = new float[Limits.MAX_LEGAL_ACTIONS * ACTION_EXACTS];
  private final int[] actionColumns = new int[Limits.MAX_LEGAL_ACTIONS * ACTION_COLUMNS];
  private final int[] rowColumns = new int[ROW_COLUMNS];
  private final float[] priorShanten = new float[Limits.MAX_LEGAL_ACTIONS];
  private final float[] priorUkeire = new float[Limits.MAX_LEGAL_ACTIONS];

  private HostBatch batch;
  private ByteBuffer bytes;
  private ShortBuffer shorts;
  private FloatBuffer floats;
  private IntBuffer ints;
  private BatchPlan plan;
  private int row = NONE;
  private int events;
  private int transitions;
  private int actions;
  private long actionsWritten;

  /**
   * 塊の中の行に付け替え、行の分の下書きを 0 にする。
   *
   * @param batch 書く塊（計画を封じたもの）
   * @param row 塊の中の行の番号
   * @return この書き手
   * @throws IllegalStateException 塊の計画を封じていないとき
   */
  public InputWriter bind(HostBatch batch, int row) {
    BatchPlan target = batch.plan();
    if (!target.sealed()) {
      throw new IllegalStateException("batch plan must be sealed before writing rows");
    }
    if (batch != this.batch) {
      // 塊ごとに 1 回だけ、このスレッドの複製を作る（絶対位置の書き込みだけを使う）。
      this.batch = batch;
      bytes = batch.blob().duplicate().clear().order(ByteOrder.LITTLE_ENDIAN);
      shorts = bytes.asShortBuffer();
      floats = bytes.asFloatBuffer();
      ints = bytes.asIntBuffer();
    }
    plan = target;
    this.row = Objects.checkIndex(row, target.rows());
    events = target.events(row);
    transitions = target.transitions(row);
    actions = target.actions(row);
    round.open(1);
    seat.open(SEATS);
    tile.open(TILES);
    event.open(events);
    transition.open(transitions);
    relation.open(BatchPlan.FIXED_TOKENS + events);
    Arrays.fill(roundScalars, (short) 0);
    Arrays.fill(seatScalars, (short) 0);
    Arrays.fill(eventScalars, 0, events * EVENT_SCALARS, (short) 0);
    Arrays.fill(transitionScalars, 0, transitions * TRANSITION_SCALARS, (short) 0);
    Arrays.fill(keysWritten, 0, transitions, 0L);
    Arrays.fill(discardExact, 0, transitions * DISCARD_EXACTS, 0f);
    Arrays.fill(actionExact, 0, actions * ACTION_EXACTS, 0f);
    Arrays.fill(priorShanten, 0, actions, 0f);
    Arrays.fill(priorUkeire, 0, actions, 0f);
    actionsWritten = 0L;
    return this;
  }

  /**
   * 局のトークンの欄の口を返す。
   *
   * @return 局のトークンに付けた口
   */
  public TokenWriter<RoundField> round() {
    return round.at(0);
  }

  /**
   * 席のトークンの欄の口を返す。
   *
   * @param relativeSeat 観測席から見た相対席
   * @return その席のトークンに付けた口
   */
  public TokenWriter<SeatField> seat(int relativeSeat) {
    return seat.at(relativeSeat);
  }

  /**
   * 牌のトークンの欄の口を返す。
   *
   * @param canonicalTile 正準の色の順の牌（トークンの番号）
   * @return その牌のトークンに付けた口
   */
  public TokenWriter<TileField> tile(int canonicalTile) {
    return tile.at(canonicalTile);
  }

  /**
   * 出来事のトークンの欄の口を返す。
   *
   * @param position 公開の出来事の順の番号
   * @return その出来事のトークンに付けた口
   */
  public TokenWriter<EventField> event(int position) {
    return event.at(position);
  }

  /**
   * 遷移の欄の口を返す。
   *
   * @param transition 行の中の遷移の番号
   * @return その遷移に付けた口
   */
  public TokenWriter<TransitionField> transition(int transition) {
    return this.transition.at(transition);
  }

  /**
   * トークンの関係の属性の口を返す。
   *
   * @param type トークンの型（{@link Axis#ROW} は局、{@link Axis#SEAT}・{@link Axis#TILE}・{@link Axis#EVENT}）
   * @param index その型の中のトークンの番号
   * @return そのトークンに付けた口
   * @throws IllegalArgumentException トークンの型でないとき
   */
  public TokenWriter<RelationAttribute> relation(Axis type, int index) {
    int count = rowTokens(type);
    return relation.at(RELATION_START[type.ordinal()] + Objects.checkIndex(index, count));
  }

  /**
   * 局の数値を書く。
   *
   * @param field 欄
   * @param value 正規化する前の値
   */
  public void scalar(RoundScalar field, float value) {
    roundScalars[field.ordinal()] = Bfloat16.fromFloat(value / field.scale());
  }

  /**
   * 席の数値を書く。
   *
   * @param relativeSeat 観測席から見た相対席
   * @param field 欄
   * @param value 正規化する前の値
   */
  public void scalar(int relativeSeat, SeatScalar field, float value) {
    seatScalars[Objects.checkIndex(relativeSeat, SEATS) * SEAT_SCALARS + field.ordinal()] =
        Bfloat16.fromFloat(value / field.scale());
  }

  /**
   * 出来事の数値を書く。
   *
   * @param position 公開の出来事の順の番号
   * @param field 欄
   * @param value 正規化する前の値
   */
  public void scalar(int position, EventScalar field, float value) {
    eventScalars[Objects.checkIndex(position, events) * EVENT_SCALARS + field.ordinal()] =
        Bfloat16.fromFloat(value / field.scale());
  }

  /**
   * 遷移の数値を書く。
   *
   * @param transition 行の中の遷移の番号
   * @param field 欄
   * @param value 正規化する前の値
   */
  public void scalar(int transition, TransitionScalar field, float value) {
    transitionScalars[
            Objects.checkIndex(transition, transitions) * TRANSITION_SCALARS + field.ordinal()] =
        Bfloat16.fromFloat(value / field.scale());
  }

  /**
   * B の道の厳密な値を書く。
   *
   * @param transition 行の中の遷移の番号
   * @param field 欄
   * @param value 正規化する前の値（有限）
   */
  public void exact(int transition, DiscardExact field, float value) {
    discardExact[Objects.checkIndex(transition, transitions) * DISCARD_EXACTS + field.ordinal()] =
        finite(value / field.scale());
  }

  /**
   * RON・TSUMO の門の道を書く。列は手の門の道の先頭から {@link WinExact} の順。
   *
   * @param action 行の中の手の番号
   * @param field 欄
   * @param value 正規化する前の値（有限）
   */
  public void exact(int action, WinExact field, float value) {
    actionExact[Objects.checkIndex(action, actions) * ACTION_EXACTS + field.ordinal()] =
        finite(value / field.scale());
  }

  /**
   * RIICHI の門の道を書く。列は手の門の道の先頭から {@link RiichiExact} の順。
   *
   * @param action 行の中の手の番号
   * @param field 欄
   * @param value 正規化する前の値（有限）
   */
  public void exact(int action, RiichiExact field, float value) {
    actionExact[Objects.checkIndex(action, actions) * ACTION_EXACTS + field.ordinal()] =
        finite(value / field.scale());
  }

  /**
   * 遷移の 1 つの区画の全ての鍵の符号を書く。{@link #commit} は、全ての遷移の全ての鍵が書かれたことと、符号が区画の符号の範囲にあることを
   * 確かめる。
   *
   * @param transition 行の中の遷移の番号
   * @param block 鍵の区画
   * @param blockCodes 区画の鍵の順の符号（{@link KeyBlock} の符号の作り方で作ったもの。長さは区画の鍵の数）
   * @throws IllegalArgumentException 長さが区画の鍵の数でないとき
   */
  public void codes(int transition, KeyBlock block, byte[] blockCodes) {
    int keys = block.keys();
    if (blockCodes.length != keys) {
      throw new IllegalArgumentException(block + " has " + keys + " keys: " + blockCodes.length);
    }
    System.arraycopy(
        blockCodes,
        0,
        codes,
        Objects.checkIndex(transition, transitions) * KEYS + block.firstKey(),
        keys);
    keysWritten[transition] |= ((1L << keys) - 1) << block.firstKey();
  }

  /**
   * 手の索引の列を書く。番号は行の中の番号で、塊の中の番号にして書く。{@link #commit} は全ての手が書かれたことを確かめる。
   *
   * @param action 行の中の手の番号
   * @param role 価値グラフでの役割
   * @param firstTransition 最初の遷移の行の中の番号（代表でない双子は代表の最初の遷移）
   * @param transitionCount 遷移の数（代表でない双子は 0）
   * @param base 基準の手の行の中の番号（なしは −1）
   * @param representative 双子の代表の手の行の中の番号（代表は自分）
   */
  public void action(
      int action,
      ActionRole role,
      int firstTransition,
      int transitionCount,
      int base,
      int representative) {
    int at = Objects.checkIndex(action, actions) * ACTION_COLUMNS;
    Objects.checkIndex(firstTransition, transitions);
    Objects.checkFromIndexSize(firstTransition, transitionCount, transitions);
    actionColumns[at + ActionColumn.ROW.ordinal()] = row;
    actionColumns[at + ActionColumn.FIRST_TRANSITION.ordinal()] =
        plan.firstTransition(row) + firstTransition;
    actionColumns[at + ActionColumn.TRANSITIONS.ordinal()] = transitionCount;
    actionColumns[at + ActionColumn.ROLE.ordinal()] = role.ordinal();
    actionColumns[at + ActionColumn.BASE.ordinal()] =
        base == NONE ? NONE : plan.firstAction(row) + Objects.checkIndex(base, actions);
    actionColumns[at + ActionColumn.REPRESENTATIVE.ordinal()] =
        plan.firstAction(row) + Objects.checkIndex(representative, actions);
    actionsWritten |= 1L << action;
  }

  /**
   * 手の規則事前 p0 の特徴を書く（ホストだけの配列。打牌でない手は書かない）。
   *
   * @param action 行の中の手の番号
   * @param shanten 打牌の後の向聴（p0 が読む float の値。{@link PriorFeatures}）
   * @param ukeire 打牌の後の受け入れの生きた枚数（同上）
   */
  public void prior(int action, float shanten, float ukeire) {
    priorShanten[Objects.checkIndex(action, actions)] = shanten;
    priorUkeire[action] = ukeire;
  }

  /**
   * 行の索引を書き、鍵の符号を区画の範囲で確かめてから、下書きを塊の区画へ写す。
   *
   * @param kind 判断の種類
   * @throws IllegalStateException 付け替えていないか、書いていないか範囲の外の鍵の符号・書いていない手があるとき
   */
  public void commit(EngineDecisionKind kind) {
    if (row == NONE) {
      throw new IllegalStateException("writer is not bound to a row");
    }
    for (int index = 0; index < transitions; index++) {
      if (keysWritten[index] != ALL_KEYS) {
        throw new IllegalStateException("transition " + index + " lacks key codes");
      }
    }
    if (actionsWritten != (1L << actions) - 1) {
      throw new IllegalStateException("actions lack index columns: " + actionsWritten);
    }
    int codeCount = transitions * KEYS;
    if (!UnsignedBytes.allAtMost(codes, CODE_CEILING, codeCount)
        || !UnsignedBytes.allAtMost(CODE_FLOOR, codes, codeCount)) {
      throw codeOutOfRange(codeCount);
    }
    rowColumns[RowColumn.FIRST_EVENT.ordinal()] = plan.firstEvent(row);
    rowColumns[RowColumn.EVENTS.ordinal()] = events;
    rowColumns[RowColumn.FIRST_ACTION.ordinal()] = plan.firstAction(row);
    rowColumns[RowColumn.ACTIONS.ordinal()] = actions;
    rowColumns[RowColumn.FIRST_TRANSITION.ordinal()] = plan.firstTransition(row);
    rowColumns[RowColumn.TRANSITIONS.ordinal()] = transitions;
    rowColumns[RowColumn.DECISION_KIND.ordinal()] = kind.ordinal();

    int firstEvent = plan.firstEvent(row);
    int firstTransition = plan.firstTransition(row);
    int firstAction = plan.firstAction(row);
    round.copyTo(bytes, plan.offset(Tensor.ROUND_CATEGORY), row, 0, 1);
    seat.copyTo(bytes, plan.offset(Tensor.SEAT_CATEGORY), SEATS * row, 0, SEATS);
    tile.copyTo(bytes, plan.offset(Tensor.TILE_CATEGORY), TILES * row, 0, TILES);
    event.copyTo(bytes, plan.offset(Tensor.EVENT_CATEGORY), firstEvent, 0, events);
    transition.copyTo(
        bytes, plan.offset(Tensor.TRANSITION_CATEGORY), firstTransition, 0, transitions);
    int relations = plan.offset(Tensor.RELATION);
    int token = 0;
    for (int type = 0; type < Axis.TOKEN.ordinal(); type++) {
      Axis axis = AXES[type];
      int count = rowTokens(axis);
      relation.copyTo(bytes, relations, plan.tokenStart(axis) + rowTokenStart(axis), token, count);
      token += count;
    }
    bytes.put(
        plan.offset(Tensor.TRANSITION_CODE) + firstTransition * KEYS, codes, 0, transitions * KEYS);
    put(shorts, Tensor.ROUND_SCALAR, row * ROUND_SCALARS, roundScalars, ROUND_SCALARS);
    put(shorts, Tensor.SEAT_SCALAR, SEATS * row * SEAT_SCALARS, seatScalars, seatScalars.length);
    put(
        shorts,
        Tensor.EVENT_SCALAR,
        firstEvent * EVENT_SCALARS,
        eventScalars,
        events * EVENT_SCALARS);
    put(
        shorts,
        Tensor.TRANSITION_SCALAR,
        firstTransition * TRANSITION_SCALARS,
        transitionScalars,
        transitions * TRANSITION_SCALARS);
    floats.put(
        element(Tensor.DISCARD_EXACT) + firstTransition * DISCARD_EXACTS,
        discardExact,
        0,
        transitions * DISCARD_EXACTS);
    floats.put(
        element(Tensor.ACTION_EXACT) + firstAction * ACTION_EXACTS,
        actionExact,
        0,
        actions * ACTION_EXACTS);
    ints.put(
        element(Tensor.ACTION_COLUMN) + firstAction * ACTION_COLUMNS,
        actionColumns,
        0,
        actions * ACTION_COLUMNS);
    ints.put(element(Tensor.ROW_COLUMN) + row * ROW_COLUMNS, rowColumns, 0, ROW_COLUMNS);
    for (int action = 0; action < actions; action++) {
      batch.prior(firstAction + action, priorShanten[action], priorUkeire[action]);
    }
    row = NONE;
  }

  /** 行の中の型のトークンの数。 */
  private int rowTokens(Axis type) {
    return type == Axis.EVENT ? events : fixedTokens(type);
  }

  /** 行ごとに数が決まった型のトークンの数。 */
  private static int fixedTokens(Axis type) {
    return switch (type) {
      case ROW -> 1;
      case SEAT -> SEATS;
      case TILE -> TILES;
      case EVENT, TOKEN, TRANSITION, ACTION ->
          throw new IllegalArgumentException("not a fixed token type: " + type);
    };
  }

  private static int[] relationStarts() {
    int[] starts = new int[Axis.TOKEN.ordinal()];
    for (int type = 1; type < starts.length; type++) {
      starts[type] = starts[type - 1] + fixedTokens(AXES[type - 1]);
    }
    return starts;
  }

  private static byte[] codeBounds(boolean ceiling) {
    byte[] bounds = new byte[Limits.MAX_TRANSITIONS * KEYS];
    for (KeyBlock block : KeyBlock.values()) {
      int bound = block.firstCode() + (ceiling ? block.codes() - 1 : 0);
      for (int key = block.firstKey(); key < block.firstKey() + block.keys(); key++) {
        for (int at = key; at < bounds.length; at += KEYS) {
          bounds[at] = (byte) bound;
        }
      }
    }
    return bounds;
  }

  private IllegalStateException codeOutOfRange(int codeCount) {
    for (int at = 0; at < codeCount; at++) {
      int code = codes[at] & 0xFF;
      if (code < (CODE_FLOOR[at] & 0xFF) || code > (CODE_CEILING[at] & 0xFF)) {
        return new IllegalStateException(
            "transition " + at / KEYS + " key " + at % KEYS + " has a code outside its block: "
                + code);
      }
    }
    throw new AssertionError("no key code is out of range");
  }

  /** 型の区画の中で、この行の最初のトークンの位置。 */
  private int rowTokenStart(Axis type) {
    return type == Axis.EVENT ? plan.firstEvent(row) : rowTokens(type) * row;
  }

  /** 区画の先頭の要素の番号（要素の型のバイト数で割る）。 */
  private int element(Tensor tensor) {
    return plan.offset(tensor) / tensor.element().bytes();
  }

  private void put(ShortBuffer target, Tensor tensor, int at, short[] draft, int length) {
    target.put(element(tensor) + at, draft, 0, length);
  }

  private static float finite(float value) {
    if (!Float.isFinite(value)) {
      throw new IllegalArgumentException("exact value is not finite: " + value);
    }
    return value;
  }

  /**
   * 1 種類のトークンのカテゴリの欄の口。{@link InputWriter} の付け替えで行と番号に付き、同じ口を使い回す。
   *
   * <p>口は欄の書き方（{@link Coding}）ごとに分ける。繰り返しのある欄は番号つきの口で書く。欄の書き方・値の enum・列・語彙は作るときに欄の
   * enum から一度だけ引いておく。値が語彙の外なら口で例外にする（数え上げは欄の語彙と比べ、番号は {@link Ids#index}、enum は値の enum の型で
   * 語彙に収まる）。
   *
   * @param <E> 欄の enum
   */
  public static final class TokenWriter<E extends Enum<E> & BoundedField> {

    private final byte[] draft;
    private final int width;
    private final int capacity;
    private final Coding[] coding;
    private final Coding[] singleCoding;
    private final Class<?>[] values;
    private final int[] kinds;
    private final int[] firstColumn;
    private final int[] repeats;
    private int tokens;
    private int base;

    private TokenWriter(TokenLayout<E> layout, Class<E> type, int capacity) {
      this.capacity = capacity;
      width = layout.width();
      draft = new byte[capacity * width];
      E[] fields = type.getEnumConstants();
      coding = new Coding[fields.length];
      singleCoding = new Coding[fields.length];
      values = new Class<?>[fields.length];
      kinds = new int[fields.length];
      firstColumn = new int[fields.length];
      repeats = new int[fields.length];
      for (E field : fields) {
        FieldSpec spec = field.spec();
        int f = field.ordinal();
        coding[f] = spec.coding();
        singleCoding[f] = spec.repeats() == 1 ? spec.coding() : null;
        values[f] = spec.values();
        kinds[f] = spec.vocabulary() - 1;
        firstColumn[f] = layout.column(field, 0);
        repeats[f] = spec.repeats();
      }
    }

    /**
     * 2 値の欄を書く。
     *
     * @param field {@link Coding#FLAG} の欄
     * @param value 値
     */
    public void flag(E field, boolean value) {
      draft[base + column(field, Coding.FLAG)] = value ? (byte) 1 : (byte) 0;
    }

    /**
     * 数え上げの欄を書く。まとめる欄は呼び手が {@link Ids#saturated} を通す。
     *
     * @param field {@link Coding#COUNT} の欄
     * @param value 数
     * @throws IllegalStateException 数が欄の語彙の外のとき
     */
    public void count(E field, int value) {
      draft[base + column(field, Coding.COUNT)] = checkedCount(field, 0, value);
    }

    /**
     * 繰り返しのある数え上げの欄を書く。
     *
     * @param field {@link Coding#COUNT} の欄
     * @param index 繰り返しの番号
     * @param value 数
     * @throws IllegalStateException 数が欄の語彙の外のとき
     */
    public void count(E field, int index, int value) {
      draft[base + column(field, index, Coding.COUNT)] = checkedCount(field, index, value);
    }

    /**
     * 番号の欄を書く（{@link Ids#index} で番号 + 1、負はなし）。
     *
     * @param field {@link Coding#ID} の欄
     * @param value 0 始まりの番号（負はなし）
     */
    public void id(E field, int value) {
      int column = column(field, Coding.ID);
      draft[base + column] = (byte) Ids.index(value, kinds[field.ordinal()]);
    }

    /**
     * enum の値の欄を書く（{@link Ids#of}）。
     *
     * @param field {@link Coding#ENUM} の欄
     * @param value 欄の値の enum の値（null はなし）
     */
    public void value(E field, Enum<?> value) {
      draft[base + column(field, Coding.ENUM)] = (byte) Ids.of(checked(field, value));
    }

    /**
     * 繰り返しのある enum の値の欄を書く。
     *
     * @param field {@link Coding#ENUM} の欄
     * @param index 繰り返しの番号
     * @param value 欄の値の enum の値（null はなし）
     */
    public void value(E field, int index, Enum<?> value) {
      draft[base + column(field, index, Coding.ENUM)] = (byte) Ids.of(checked(field, value));
    }

    /** 行の分のトークンを 0 にして開く。 */
    private void open(int tokens) {
      if (tokens > capacity) {
        throw new IllegalArgumentException("too many tokens: " + tokens);
      }
      this.tokens = tokens;
      Arrays.fill(draft, 0, tokens * width, (byte) 0);
      base = 0;
    }

    private TokenWriter<E> at(int token) {
      base = Objects.checkIndex(token, tokens) * width;
      return this;
    }

    /** 繰り返しのない欄の列。書き方が違うか、繰り返しのある欄なら例外。 */
    private int column(E field, Coding expected) {
      int f = field.ordinal();
      if (singleCoding[f] != expected) {
        throw wrongPort(field, expected);
      }
      return firstColumn[f];
    }

    /** 繰り返しのある欄の index 番目の列。書き方が違えば例外。 */
    private int column(E field, int index, Coding expected) {
      int f = field.ordinal();
      if (coding[f] != expected) {
        throw wrongPort(field, expected);
      }
      return firstColumn[f] + Objects.checkIndex(index, repeats[f]);
    }

    private IllegalArgumentException wrongPort(E field, Coding expected) {
      return new IllegalArgumentException(
          field + " is a " + coding[field.ordinal()] + " field with " + repeats[field.ordinal()]
              + " repeats, written as " + expected);
    }

    private Enum<?> checked(E field, Enum<?> value) {
      if (value != null && value.getDeclaringClass() != values[field.ordinal()]) {
        throw new IllegalArgumentException(
            field + " does not take " + value.getDeclaringClass().getSimpleName());
      }
      return value;
    }

    /** 数え上げの値を欄の語彙で確かめる（{@link TokenLayout#requireValid} と同じ「値 &lt; 語彙」）。 */
    private byte checkedCount(E field, int index, int value) {
      if (value < 0 || value > kinds[field.ordinal()]) {
        throw new IllegalStateException(
            field.getDeclaringClass().getSimpleName()
                + "."
                + field.name()
                + "["
                + index
                + "] out of range: "
                + value);
      }
      return (byte) value;
    }

    /** 下書きの from 番目から count 個のトークンを、区画の at 番目のトークンの位置へ写す。 */
    private void copyTo(ByteBuffer target, int offset, int at, int from, int count) {
      target.put(offset + at * width, draft, from * width, count * width);
    }
  }
}

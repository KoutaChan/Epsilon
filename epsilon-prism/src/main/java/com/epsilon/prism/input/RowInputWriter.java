package com.epsilon.prism.input;

import com.epsilon.agent.observe.codec.Bfloat16;
import com.epsilon.agent.observe.codec.UnsignedBytes;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.FieldSpec.BoundedField;
import com.epsilon.prism.input.FieldSpec.Coding;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 1 行分のネットワークの入力を {@link HostBatch} に書く、型つきのライター。フィールドの型（局・席・牌・イベント・遷移のフィールドの enum と、
 * 関係の属性の enum）は型引数で決まる。ルールごとのモジュールは、フィールドの型だけを決めてこのクラスを継ぐ。{@link #bind} でバッチの別の行に
 * 付け替えて使い回し、書くときにメモリを確保しない。エンコードのスレッドごとに 1 つ持つ。
 *
 * <p>全てのテンソルの 1 行分を下書きに書き、{@link #commit} でバッチの領域へまとめて写す。写す前に行ごとに 1 回、キーのコードがキーの範囲に
 * 収まるかを確かめる。下書きは付け替えのときに行の分を 0 にするので、書かなかった要素は 0 になり、使い回したバッチに前の値は残らない。途中で
 * 例外になった行は、バッチに何も残さない。
 *
 * <p>カテゴリのフィールドを書くメソッドは、書き方（{@link Coding}）ごとに分ける。メソッドの中でフィールドの書き方と値の enum の型を照合してから、
 * {@link Ids} を通す。語彙は書くメソッドで確かめる（{@link TokenLayout#requireValid} と同じ「値 &lt; 語彙」）。2 値と番号のメソッドは作りの上で
 * 語彙を超えず、enum のメソッドは語彙に無い定数を拒む。そのため、フィールドの語彙と比べるのは数え上げのメソッドだけ。下書きにはこれらの
 * メソッドからしか書かないので、写す前に下書きを語彙で確かめ直さない。数値はフィールドの {@link InputFormat.Scalar#scale} で、厳密な
 * フィールドは {@link InputFormat.Exact#scale} で割ってから書く。席・牌・イベント・遷移・手の番号は行の中の番号で、バッチの中の位置には
 * このライターが {@link BatchPlan} を見て直す。
 *
 * @param <R> 局のフィールド
 * @param <S> 席のフィールド
 * @param <T> 牌のフィールド
 * @param <E> イベントのフィールド
 * @param <X> 遷移のフィールド
 * @param <L> 関係の属性
 */
public class RowInputWriter<
    R extends Enum<R> & BoundedField,
    S extends Enum<S> & BoundedField,
    T extends Enum<T> & BoundedField,
    E extends Enum<E> & BoundedField,
    X extends Enum<X> & BoundedField,
    L extends Enum<L> & BoundedField> {

  private static final int NONE = -1;
  private static final Axis[] AXES = Axis.values();

  private final int seats;
  private final int tiles;
  private final Vocabulary<EngineDecisionKind> decisionKinds;
  private final int roundScalarWidth;
  private final int seatScalarWidth;
  private final int eventScalarWidth;
  private final int transitionScalarWidth;
  private final int discardExactWidth;
  private final int actionExactWidth;
  private final int actionColumnWidth;
  private final int rowColumnWidth;
  private final int keys;
  private final long allKeys;

  /** 関係の属性の下書きでの、型ごとの最初のトークンの位置（局・席・牌・イベントの順。イベントより前の型はトークンの数が決まっている）。 */
  private final int[] relationStart;

  /** コードの下書きの位置ごとに、そのキーの範囲の最初のコード（キーの並びを遷移の数だけ繰り返す）。 */
  private final byte[] codeFloor;

  /** 同じく、キーの範囲の最後のコード。 */
  private final byte[] codeCeiling;

  private final TokenWriter<R> round;
  private final TokenWriter<S> seat;
  private final TokenWriter<T> tile;
  private final TokenWriter<E> event;
  private final TokenWriter<X> transition;
  private final TokenWriter<L> relation;
  private final short[] roundScalars;
  private final short[] seatScalars;
  private final short[] eventScalars;
  private final short[] transitionScalars;
  private final byte[] codes;
  private final long[] keysWritten = new long[Limits.MAX_TRANSITIONS];
  private final float[] discardExact;
  private final float[] actionExact;
  private final int[] actionColumns;
  private final int[] rowColumns;
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
   * 入力形式のフィールドの配置で、ライターを作る。
   *
   * @param format 入力形式
   * @param roundLayout 局のフィールドの配置（入力形式の {@link InputFormat#round()} と同じもの）
   * @param seatLayout 席のフィールドの配置
   * @param tileLayout 牌のフィールドの配置
   * @param eventLayout イベントのフィールドの配置
   * @param transitionLayout 遷移のフィールドの配置
   * @param relationLayout 関係の属性の配置
   * @param blocks キーの範囲（キーの順）
   * @throws IllegalArgumentException 配置が入力形式のものでないとき、キーが 64 個を超えるとき、または範囲のキーの数の合計が入力形式のキーの数と
   *     合わないとき
   */
  protected RowInputWriter(
      InputFormat format,
      TokenLayout<R> roundLayout,
      TokenLayout<S> seatLayout,
      TokenLayout<T> tileLayout,
      TokenLayout<E> eventLayout,
      TokenLayout<X> transitionLayout,
      TokenLayout<L> relationLayout,
      List<? extends InputFormat.KeyRange> blocks) {
    if (roundLayout != format.round()
        || seatLayout != format.seat()
        || tileLayout != format.tile()
        || eventLayout != format.event()
        || transitionLayout != format.transition()
        || relationLayout != format.relation()) {
      throw new IllegalArgumentException("the writer layouts are not the " + format.variant() + " input layouts");
    }
    seats = format.seats();
    tiles = format.tiles();
    decisionKinds = Vocabulary.of(EngineDecisionKind.class, format.variant());
    roundScalarWidth = format.width(Tensor.ROUND_SCALAR);
    seatScalarWidth = format.width(Tensor.SEAT_SCALAR);
    eventScalarWidth = format.width(Tensor.EVENT_SCALAR);
    transitionScalarWidth = format.width(Tensor.TRANSITION_SCALAR);
    discardExactWidth = format.width(Tensor.DISCARD_EXACT);
    actionExactWidth = format.width(Tensor.ACTION_EXACT);
    actionColumnWidth = format.width(Tensor.ACTION_COLUMN);
    rowColumnWidth = format.width(Tensor.ROW_COLUMN);
    keys = format.keys().count();
    if (keys > Long.SIZE) {
      throw new IllegalArgumentException("more than " + Long.SIZE + " keys: " + keys);
    }
    allKeys = keys == Long.SIZE ? -1L : (1L << keys) - 1;
    relationStart = relationStarts();
    codeFloor = codeBounds(blocks, false);
    codeCeiling = codeBounds(blocks, true);
    int maxEvents = format.maxEvents();
    round = new TokenWriter<>(roundLayout, 1);
    seat = new TokenWriter<>(seatLayout, seats);
    tile = new TokenWriter<>(tileLayout, tiles);
    event = new TokenWriter<>(eventLayout, maxEvents);
    transition = new TokenWriter<>(transitionLayout, Limits.MAX_TRANSITIONS);
    relation = new TokenWriter<>(relationLayout, format.fixedTokens() + maxEvents);
    roundScalars = new short[roundScalarWidth];
    seatScalars = new short[seats * seatScalarWidth];
    eventScalars = new short[maxEvents * eventScalarWidth];
    transitionScalars = new short[Limits.MAX_TRANSITIONS * transitionScalarWidth];
    codes = new byte[Limits.MAX_TRANSITIONS * keys];
    discardExact = new float[Limits.MAX_TRANSITIONS * discardExactWidth];
    actionExact = new float[Limits.MAX_LEGAL_ACTIONS * actionExactWidth];
    actionColumns = new int[Limits.MAX_LEGAL_ACTIONS * actionColumnWidth];
    rowColumns = new int[rowColumnWidth];
  }

  /**
   * バッチの中の行に付け替え、その行の分の下書きを 0 にする。
   *
   * @param batch 書き先のバッチ（計画を確定したもの）
   * @param row バッチの中の行の番号
   * @return このライター
   * @throws IllegalStateException バッチの計画が確定していないとき
   * @throws IllegalArgumentException 行のイベントの数が入力形式の上限を超えるとき
   */
  public RowInputWriter<R, S, T, E, X, L> bind(HostBatch batch, int row) {
    BatchPlan target = batch.plan();
    if (!target.sealed()) {
      throw new IllegalStateException("batch plan must be sealed before writing rows");
    }
    if (batch != this.batch) {
      // バッチが替わったときだけ、このスレッド用にバッファの複製を作る（書き込みは全て絶対位置で行う）。
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
    seat.open(seats);
    tile.open(tiles);
    event.open(events);
    transition.open(transitions);
    relation.open(seats + tiles + 1 + events);
    Arrays.fill(roundScalars, (short) 0);
    Arrays.fill(seatScalars, (short) 0);
    Arrays.fill(eventScalars, 0, events * eventScalarWidth, (short) 0);
    Arrays.fill(transitionScalars, 0, transitions * transitionScalarWidth, (short) 0);
    Arrays.fill(keysWritten, 0, transitions, 0L);
    Arrays.fill(discardExact, 0, transitions * discardExactWidth, 0f);
    Arrays.fill(actionExact, 0, actions * actionExactWidth, 0f);
    Arrays.fill(priorShanten, 0, actions, 0f);
    Arrays.fill(priorUkeire, 0, actions, 0f);
    actionsWritten = 0L;
    return this;
  }

  /**
   * 局のトークンのフィールドを書くライターを返す。
   *
   * @return 局のトークンに付けたライター
   */
  public TokenWriter<R> round() {
    return round.at(0);
  }

  /**
   * 席のトークンのフィールドを書くライターを返す。
   *
   * @param relativeSeat 観測者から見た相対席
   * @return その席のトークンに付けたライター
   */
  public TokenWriter<S> seat(int relativeSeat) {
    return seat.at(relativeSeat);
  }

  /**
   * 牌のトークンのフィールドを書くライターを返す。
   *
   * @param token 色を正規化した順での、牌のトークンの番号
   * @return その牌のトークンに付けたライター
   */
  public TokenWriter<T> tile(int token) {
    return tile.at(token);
  }

  /**
   * イベントのトークンのフィールドを書くライターを返す。
   *
   * @param position 公開イベントの順の番号
   * @return そのイベントのトークンに付けたライター
   */
  public TokenWriter<E> event(int position) {
    return event.at(position);
  }

  /**
   * 遷移のフィールドを書くライターを返す。
   *
   * @param transition 行の中の遷移の番号
   * @return その遷移に付けたライター
   */
  public TokenWriter<X> transition(int transition) {
    return this.transition.at(transition);
  }

  /**
   * トークンの関係の属性を書くライターを返す。
   *
   * @param type トークンの型（局は {@link Axis#ROW}、ほかは {@link Axis#SEAT}・{@link Axis#TILE}・{@link Axis#EVENT}）
   * @param index その型の中のトークンの番号
   * @return そのトークンに付けたライター
   * @throws IllegalArgumentException トークンの型でないとき
   */
  public TokenWriter<L> relation(Axis type, int index) {
    int count = rowTokens(type);
    return relation.at(relationStart[type.ordinal()] + Objects.checkIndex(index, count));
  }

  /**
   * 局の数値を書く。
   *
   * @param field 局の数値のフィールド
   * @param value 正規化する前の値
   * @throws IllegalArgumentException 局の数値のフィールドでないとき
   */
  public void scalar(InputFormat.Scalar field, float value) {
    if (field.tensor() != Tensor.ROUND_SCALAR) {
      throw new IllegalArgumentException(field + " is not a round scalar");
    }
    roundScalars[field.ordinal()] = Bfloat16.fromFloat(value / field.scale());
  }

  /**
   * 席・イベント・遷移の数値を書く。
   *
   * @param index 行の中の番号（フィールドの領域の軸に応じて、相対席・公開イベントの順の番号・遷移の番号のどれか）
   * @param field 席・イベント・遷移の数値のフィールド
   * @param value 正規化する前の値
   * @throws IllegalArgumentException 席・イベント・遷移の数値のフィールドでないとき
   */
  public void scalar(int index, InputFormat.Scalar field, float value) {
    short normalized = Bfloat16.fromFloat(value / field.scale());
    switch (field.tensor()) {
      case SEAT_SCALAR ->
          seatScalars[Objects.checkIndex(index, seats) * seatScalarWidth + field.ordinal()] = normalized;
      case EVENT_SCALAR ->
          eventScalars[Objects.checkIndex(index, events) * eventScalarWidth + field.ordinal()] = normalized;
      case TRANSITION_SCALAR ->
          transitionScalars[Objects.checkIndex(index, transitions) * transitionScalarWidth + field.ordinal()] =
              normalized;
      default -> throw new IllegalArgumentException(field + " is not a seat, event or transition scalar");
    }
  }

  /**
   * 厳密なフィールドを書く。打牌の値 B のヘッドが読む {@link Tensor#DISCARD_EXACT} は遷移ごと、ゲートのヘッドが読む
   * {@link Tensor#ACTION_EXACT} は手ごとに書く。列はフィールドの領域の中の番号。
   *
   * @param index 行の中の遷移の番号（{@link Tensor#DISCARD_EXACT}）か手の番号（{@link Tensor#ACTION_EXACT}）
   * @param field フィールド
   * @param value 正規化する前の値（有限）
   * @throws IllegalArgumentException 厳密なフィールドでないか、値が有限でないとき
   */
  public void exact(int index, InputFormat.Exact field, float value) {
    float normalized = finite(value / field.scale());
    switch (field.tensor()) {
      case DISCARD_EXACT ->
          discardExact[Objects.checkIndex(index, transitions) * discardExactWidth + field.ordinal()] = normalized;
      case ACTION_EXACT ->
          actionExact[Objects.checkIndex(index, actions) * actionExactWidth + field.ordinal()] = normalized;
      default -> throw new IllegalArgumentException(field + " is not an exact field");
    }
  }

  /**
   * 1 つの遷移について、1 つのキーの範囲に入る全てのキーのコードを書く。{@link #commit} は、全ての遷移で全てのキーが書かれたことと、コードがキーの
   * 範囲に収まることを確かめる。
   *
   * @param transition 行の中の遷移の番号
   * @param block キーの範囲
   * @param blockCodes 範囲のキーの順に並べたコード（長さは範囲のキーの数）
   * @throws IllegalArgumentException 長さが範囲のキーの数と違うとき
   */
  public void codes(int transition, InputFormat.KeyRange block, byte[] blockCodes) {
    int count = block.keys();
    if (blockCodes.length != count) {
      throw new IllegalArgumentException(block + " has " + count + " keys: " + blockCodes.length);
    }
    System.arraycopy(
        blockCodes, 0, codes, Objects.checkIndex(transition, transitions) * keys + block.firstKey(), count);
    keysWritten[transition] |= ((1L << count) - 1) << block.firstKey();
  }

  /**
   * 手のインデックスの列を書く。引数の番号は行の中の番号で、バッチの中の番号に直して書く。{@link #commit} は全ての手が書かれたことを
   * 確かめる。
   *
   * @param action 行の中の手の番号
   * @param role 価値グラフでの手の扱い
   * @param firstTransition 最初の遷移の、行の中の番号（同値な打牌の代表でない手では、代表の最初の遷移）
   * @param transitionCount 遷移の数（同値な打牌の代表でない手では 0）
   * @param base 基準の手の、行の中の番号（無ければ −1）
   * @param representative 同値な打牌の代表の、行の中の番号（代表なら自分自身）
   */
  public void action(
      int action,
      InputFormat.Role role,
      int firstTransition,
      int transitionCount,
      int base,
      int representative) {
    int at = Objects.checkIndex(action, actions) * actionColumnWidth;
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
   * 手について、ルールベースの事前分布 p0 が読む特徴を書く。書き先はホストだけに持つ配列。打牌でない手では呼ばず、値は 0 のままにする。
   *
   * @param action 行の中の手の番号
   * @param shanten 打牌の後の向聴（p0 が読む float の値。{@link PriorFeatures}）
   * @param ukeire 打牌の後の受け入れの残り枚数（同上）
   */
  public void prior(int action, float shanten, float ukeire) {
    priorShanten[Objects.checkIndex(action, actions)] = shanten;
    priorUkeire[action] = ukeire;
  }

  /**
   * 行のインデックスを書き、キーのコードが範囲に収まるかを確かめてから、下書きをバッチの領域へ写す。
   *
   * @param kind 判断の種類
   * @throws IllegalStateException 行に付け替えていないとき、書いていないか範囲の外にあるキーのコードがあるとき、または書いていない手が
   *     あるとき
   */
  public void commit(EngineDecisionKind kind) {
    if (row == NONE) {
      throw new IllegalStateException("writer is not bound to a row");
    }
    for (int index = 0; index < transitions; index++) {
      if (keysWritten[index] != allKeys) {
        throw new IllegalStateException("transition " + index + " lacks key codes");
      }
    }
    if (actionsWritten != (1L << actions) - 1) {
      throw new IllegalStateException("actions lack index columns: " + actionsWritten);
    }
    int codeCount = transitions * keys;
    if (!UnsignedBytes.allAtMost(codes, codeCeiling, codeCount)
        || !UnsignedBytes.allAtMost(codeFloor, codes, codeCount)) {
      throw codeOutOfRange(codeCount);
    }
    rowColumns[RowColumn.FIRST_EVENT.ordinal()] = plan.firstEvent(row);
    rowColumns[RowColumn.EVENTS.ordinal()] = events;
    rowColumns[RowColumn.FIRST_ACTION.ordinal()] = plan.firstAction(row);
    rowColumns[RowColumn.ACTIONS.ordinal()] = actions;
    rowColumns[RowColumn.FIRST_TRANSITION.ordinal()] = plan.firstTransition(row);
    rowColumns[RowColumn.TRANSITIONS.ordinal()] = transitions;
    rowColumns[RowColumn.DECISION_KIND.ordinal()] = decisionKinds.index(kind);

    int firstEvent = plan.firstEvent(row);
    int firstTransition = plan.firstTransition(row);
    int firstAction = plan.firstAction(row);
    round.copyTo(bytes, plan.offset(Tensor.ROUND_CATEGORY), row, 0, 1);
    seat.copyTo(bytes, plan.offset(Tensor.SEAT_CATEGORY), seats * row, 0, seats);
    tile.copyTo(bytes, plan.offset(Tensor.TILE_CATEGORY), tiles * row, 0, tiles);
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
        plan.offset(Tensor.TRANSITION_CODE) + firstTransition * keys, codes, 0, transitions * keys);
    put(shorts, Tensor.ROUND_SCALAR, row * roundScalarWidth, roundScalars, roundScalarWidth);
    put(shorts, Tensor.SEAT_SCALAR, seats * row * seatScalarWidth, seatScalars, seatScalars.length);
    put(
        shorts,
        Tensor.EVENT_SCALAR,
        firstEvent * eventScalarWidth,
        eventScalars,
        events * eventScalarWidth);
    put(
        shorts,
        Tensor.TRANSITION_SCALAR,
        firstTransition * transitionScalarWidth,
        transitionScalars,
        transitions * transitionScalarWidth);
    floats.put(
        element(Tensor.DISCARD_EXACT) + firstTransition * discardExactWidth,
        discardExact,
        0,
        transitions * discardExactWidth);
    floats.put(
        element(Tensor.ACTION_EXACT) + firstAction * actionExactWidth,
        actionExact,
        0,
        actions * actionExactWidth);
    ints.put(
        element(Tensor.ACTION_COLUMN) + firstAction * actionColumnWidth,
        actionColumns,
        0,
        actions * actionColumnWidth);
    ints.put(element(Tensor.ROW_COLUMN) + row * rowColumnWidth, rowColumns, 0, rowColumnWidth);
    for (int action = 0; action < actions; action++) {
      batch.prior(firstAction + action, priorShanten[action], priorUkeire[action]);
    }
    row = NONE;
  }

  /** 行の中の、その型のトークンの数。 */
  private int rowTokens(Axis type) {
    return type == Axis.EVENT ? events : fixedTokens(type);
  }

  /** 行ごとに数が決まった型のトークンの数。 */
  private int fixedTokens(Axis type) {
    return switch (type) {
      case ROW -> 1;
      case SEAT -> seats;
      case TILE -> tiles;
      case EVENT, TOKEN, TRANSITION, ACTION ->
          throw new IllegalArgumentException("not a fixed token type: " + type);
    };
  }

  private int[] relationStarts() {
    int[] starts = new int[Axis.TOKEN.ordinal()];
    for (int type = 1; type < starts.length; type++) {
      starts[type] = starts[type - 1] + fixedTokens(AXES[type - 1]);
    }
    return starts;
  }

  private byte[] codeBounds(List<? extends InputFormat.KeyRange> blocks, boolean ceiling) {
    byte[] bounds = new byte[Limits.MAX_TRANSITIONS * keys];
    int covered = 0;
    for (InputFormat.KeyRange block : blocks) {
      int bound = block.firstCode() + (ceiling ? block.codes() - 1 : 0);
      for (int key = block.firstKey(); key < block.firstKey() + block.keys(); key++) {
        for (int at = key; at < bounds.length; at += keys) {
          bounds[at] = (byte) bound;
        }
      }
      covered += block.keys();
    }
    if (covered != keys) {
      throw new IllegalArgumentException("key blocks cover " + covered + " of " + keys + " keys");
    }
    return bounds;
  }

  private IllegalStateException codeOutOfRange(int codeCount) {
    for (int at = 0; at < codeCount; at++) {
      int code = codes[at] & 0xFF;
      if (code < (codeFloor[at] & 0xFF) || code > (codeCeiling[at] & 0xFF)) {
        return new IllegalStateException(
            "transition " + at / keys + " key " + at % keys + " has a code outside its block: " + code);
      }
    }
    throw new AssertionError("no key code is out of range");
  }

  /** 型の領域の中での、この行の最初のトークンの位置。 */
  private int rowTokenStart(Axis type) {
    return type == Axis.EVENT ? plan.firstEvent(row) : rowTokens(type) * row;
  }

  /** 領域の先頭の要素の番号（バイトの位置を要素の型のバイト数で割る）。 */
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
   * 1 種類のトークンの、カテゴリのフィールドを書くライター。{@link RowInputWriter} の付け替えで行とトークンの番号が決まり、同じものを
   * 使い回す。
   *
   * <p>書くメソッドは、フィールドの書き方（{@link Coding}）ごとに分ける。繰り返しのあるフィールドは、繰り返しの番号を取るメソッドで書く。
   * フィールドの書き方・値の enum・列・語彙は、作るときにフィールドの enum から一度だけ引いておく。値が語彙の外なら、書くメソッドで例外に
   * する（数え上げはフィールドの語彙と比べ、番号は {@link Ids#index} が確かめ、enum は値の enum の型によって語彙に収まる）。
   *
   * @param <F> フィールドの enum
   */
  public static final class TokenWriter<F extends Enum<F> & BoundedField> {

    private final byte[] draft;
    private final int width;
    private final int capacity;
    private final Coding[] coding;
    private final Coding[] singleCoding;
    private final Vocabulary<?>[] values;
    private final int[] kinds;
    private final int[] firstColumn;
    private final int[] repeats;
    private int tokens;
    private int base;

    private TokenWriter(TokenLayout<F> layout, int capacity) {
      this.capacity = capacity;
      width = layout.width();
      draft = new byte[capacity * width];
      F[] fields = layout.type().getEnumConstants();
      coding = new Coding[fields.length];
      singleCoding = new Coding[fields.length];
      values = new Vocabulary<?>[fields.length];
      kinds = new int[fields.length];
      firstColumn = new int[fields.length];
      repeats = new int[fields.length];
      for (F field : fields) {
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
     * 2 値のフィールドを書く。
     *
     * @param field {@link Coding#FLAG} のフィールド
     * @param value 値
     */
    public void flag(F field, boolean value) {
      draft[base + column(field, Coding.FLAG)] = value ? (byte) 1 : (byte) 0;
    }

    /**
     * 数え上げのフィールドを書く。上限で頭打ちにするフィールドは、呼び出し元が先に {@link Ids#saturated} を通す。
     *
     * @param field {@link Coding#COUNT} のフィールド
     * @param value 数
     * @throws IllegalStateException 数がフィールドの語彙の外のとき
     */
    public void count(F field, int value) {
      draft[base + column(field, Coding.COUNT)] = checkedCount(field, 0, value);
    }

    /**
     * 繰り返しのある、数え上げのフィールドを書く。
     *
     * @param field {@link Coding#COUNT} のフィールド
     * @param index 繰り返しの番号
     * @param value 数
     * @throws IllegalStateException 数がフィールドの語彙の外のとき
     */
    public void count(F field, int index, int value) {
      draft[base + column(field, index, Coding.COUNT)] = checkedCount(field, index, value);
    }

    /**
     * 番号のフィールドを書く。{@link Ids#index} で番号 + 1 にして書き、負の番号は「無し」になる。
     *
     * @param field {@link Coding#ID} のフィールド
     * @param value 0 始まりの番号（負なら無し）
     */
    public void id(F field, int value) {
      int column = column(field, Coding.ID);
      draft[base + column] = (byte) Ids.index(value, kinds[field.ordinal()]);
    }

    /**
     * enum の値のフィールドを書く。{@link Ids#of} で語彙の中の位置 + 1 にして書く。
     *
     * @param field {@link Coding#ENUM} のフィールド
     * @param value フィールドの値の enum の定数（null なら無し）
     */
    public void value(F field, Enum<?> value) {
      draft[base + column(field, Coding.ENUM)] = (byte) Ids.of(values[field.ordinal()], checked(field, value));
    }

    /**
     * 繰り返しのある、enum の値のフィールドを書く。
     *
     * @param field {@link Coding#ENUM} のフィールド
     * @param index 繰り返しの番号
     * @param value フィールドの値の enum の定数（null なら無し）
     */
    public void value(F field, int index, Enum<?> value) {
      draft[base + column(field, index, Coding.ENUM)] =
          (byte) Ids.of(values[field.ordinal()], checked(field, value));
    }

    /** 新しい行のトークンの数を決め、その分の下書きを 0 にする。 */
    private void open(int tokens) {
      if (tokens > capacity) {
        throw new IllegalArgumentException("too many tokens: " + tokens);
      }
      this.tokens = tokens;
      Arrays.fill(draft, 0, tokens * width, (byte) 0);
      base = 0;
    }

    private TokenWriter<F> at(int token) {
      base = Objects.checkIndex(token, tokens) * width;
      return this;
    }

    /** 繰り返しのないフィールドの列。書き方が違うか、繰り返しのあるフィールドなら例外にする。 */
    private int column(F field, Coding expected) {
      int f = field.ordinal();
      if (singleCoding[f] != expected) {
        throw wrongPort(field, expected);
      }
      return firstColumn[f];
    }

    /** 繰り返しのあるフィールドの、{@code index} 番目の列。書き方が違えば例外にする。 */
    private int column(F field, int index, Coding expected) {
      int f = field.ordinal();
      if (coding[f] != expected) {
        throw wrongPort(field, expected);
      }
      return firstColumn[f] + Objects.checkIndex(index, repeats[f]);
    }

    private IllegalArgumentException wrongPort(F field, Coding expected) {
      return new IllegalArgumentException(
          field + " is a " + coding[field.ordinal()] + " field with " + repeats[field.ordinal()]
              + " repeats, written as " + expected);
    }

    private Enum<?> checked(F field, Enum<?> value) {
      if (value != null && value.getDeclaringClass() != values[field.ordinal()].type()) {
        throw new IllegalArgumentException(
            field + " does not take " + value.getDeclaringClass().getSimpleName());
      }
      return value;
    }

    /** 数え上げの値がフィールドの語彙に収まるかを確かめる（{@link TokenLayout#requireValid} と同じ「値 &lt; 語彙」）。 */
    private byte checkedCount(F field, int index, int value) {
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

    /** 下書きの {@code from} 番目から {@code count} 個のトークンを、領域の {@code at} 番目のトークンの位置へ写す。 */
    private void copyTo(ByteBuffer target, int offset, int at, int from, int count) {
      target.put(offset + at * width, draft, from * width, count * width);
    }
  }
}

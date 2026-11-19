package com.epsilon.prism.input;

import com.epsilon.agent.observe.EventRecords.Prefix;
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
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.EventField;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.RelationAttribute;
import com.epsilon.prism.input.InputSchema.RoundField;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.TileField;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.InputSchema.WinExact;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 入力形式の各フィールドを1行分の下書きに書く。{@link #bind} でバッチの行に結び付けて使い回す。
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
 */
public final class RowInputWriter {
  private final InputFormat schema;

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
  private final byte[][] initialCodes;
  private final int tileStart;
  private final byte[] handCodes;
  private final int handColumn;
  private int nextCodeRow;
  private int tileCodeStart;

  /** コードの下書きの位置ごとに、そのキーの範囲の最初のコード（キーの並びを遷移の数だけ繰り返す）。 */
  private final byte[] codeFloor;

  /** 同じく、キーの範囲の最後のコード。 */
  private final byte[] codeCeiling;

  private final TokenWriter<RoundField> round;
  private final TokenWriter<SeatField> seat;
  private final TokenWriter<TileField> tile;
  private final TokenWriter<EventField> event;
  private final TokenWriter<TransitionField> transition;
  private final TokenWriter<RelationAttribute> relation;
  private final InputColumn[] columns;
  private final InputSchema.RowScalar[] scalarColumns;
  private final short[] eventAgeValues;
  private final short[] roundScalars;
  private final short[] seatScalars;
  private final short[] eventScalars;
  private final short[] transitionScalars;
  private final byte[] codes;
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
  private final int transitionCodeWidth;
  private final CandidateWriter candidate;
  private int row = NONE;
  private int events;
  private int transitions;
  private int actions;
  private long actionsWritten;
  private int eventPrefix;
  private boolean published;

  /** 入力形式に合わせて各軸の下書きを確保する。 */
  public RowInputWriter(InputFormat format) {
    schema = format;
    transitionCodeWidth = format.width(Tensor.TRANSITION_CODE);
    candidate = format.has(InputFeature.CANDIDATE_SEGMENTS) ? new CandidateWriter(format) : null;
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
    tileStart = format.block(KeyBlock.TILE).firstKey();
    handCodes = new byte[tiles];
    handColumn = format.tile().column(TileField.HAND);
    initialCodes = new byte[seats + 1][tileStart];
    for (int source = -1; source < seats; source++) {
      byte[] row = initialCodes[source + 1];
      row[format.block(KeyBlock.ROUND).firstKey()] =
          (byte) format.block(KeyBlock.ROUND).firstCode();
      for (int seat = 0; seat < seats; seat++) {
        row[format.block(KeyBlock.SEAT).firstKey() + seat] =
            (byte) format.seatCode(seat, seat == source);
      }
    }
    codeFloor = codeBounds(false);
    codeCeiling = codeBounds(true);
    int maxEvents = format.maxEvents();
    round = new TokenWriter<>(format.round(), 1);
    seat = new TokenWriter<>(format.seat(), seats);
    tile = new TokenWriter<>(format.tile(), tiles);
    event = new TokenWriter<>(format.event(), maxEvents);
    transition = new TokenWriter<>(format.transition(), Limits.MAX_TRANSITIONS);
    relation = new TokenWriter<>(format.relation(), format.fixedTokens() + maxEvents);
    List<InputColumn> columns = new ArrayList<>();
    bindColumns(columns, format.round(), round);
    bindColumns(columns, format.tile(), tile);
    bindColumns(columns, format.event(), event);
    bindColumns(columns, format.relation(), relation);
    this.columns = columns.toArray(InputColumn[]::new);
    List<InputSchema.RowScalar> scalars = new ArrayList<>();
    scalars.addAll(Arrays.asList(InputSchema.RoundScalar.values()));
    scalars.addAll(Arrays.asList(InputSchema.SeatScalar.values()));
    scalars.addAll(Arrays.asList(InputSchema.EventScalar.values()));
    scalarColumns = scalars.toArray(InputSchema.RowScalar[]::new);
    eventAgeValues = new short[maxEvents];
    float ageScale = InputSchema.EventScalar.AGE.scale(format.variant());
    for (int age = 0; age < maxEvents; age++) {
      eventAgeValues[format.has(InputFeature.PUBLIC_ORDER_EVENTS) ? age : maxEvents - 1 - age] =
          Bfloat16.fromFloat(age / ageScale);
    }
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

  private <F extends Enum<F> & InputColumn.Field> void bindColumns(
      List<InputColumn> columns, TokenLayout<F> layout, TokenWriter<F> writer) {
    for (F field : layout.fields()) {
      columns.add(
          new InputColumn(
              field.source(schema),
              writer.draft,
              layout.column(field, 0),
              layout.width(),
              layout.spec(field)));
    }
  }

  void writeColumns(RowFacts facts) {
    for (InputColumn column : columns) column.write(facts);
    for (InputSchema.RowScalar column : scalarColumns) column.write(facts, this);
  }

  void eventAges(int count) {
    int start = published ? eventPrefix : 0;
    int first = schema.has(InputFeature.PUBLIC_ORDER_EVENTS) ? 0 : eventAgeValues.length - count;
    System.arraycopy(eventAgeValues, first + start, eventScalars, start, count - start);
  }

  /**
   * バッチの中の行に付け替え、その行の分の下書きを 0 にする。
   *
   * @param batch 書き先のバッチ（計画を確定したもの）
   * @param row バッチの中の行の番号
   * @return このライター
   * @throws IllegalStateException バッチの計画が確定していないとき
   * @throws IllegalArgumentException バッチの形式が異なるとき、または行のイベントの数が上限を超えるとき
   */
  public RowInputWriter bind(HostBatch batch, int row) {
    BatchPlan target = batch.plan();
    if (!target.sealed()) {
      throw new IllegalStateException("batch plan must be sealed before writing rows");
    }
    if (batch != this.batch) {
      if (target.format() != schema) {
        throw new IllegalArgumentException(
            "Writer format "
                + schema.fingerprint()
                + " does not match batch format "
                + target.format().fingerprint());
      }
      // バッチが替わったときだけ、このスレッド用にバッファの複製を作る（書き込みは全て絶対位置で行う）。
      this.batch = batch;
      bytes = batch.storage().duplicate().clear().order(ByteOrder.LITTLE_ENDIAN);
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
    nextCodeRow = 0;
    Arrays.fill(discardExact, 0, transitions * discardExactWidth, 0f);
    Arrays.fill(actionExact, 0, actions * actionExactWidth, 0f);
    Arrays.fill(priorShanten, 0, actions, 0f);
    Arrays.fill(priorUkeire, 0, actions, 0f);
    actionsWritten = 0L;
    eventPrefix = 0;
    published = false;
    batch.bindEvents(row, null);
    return this;
  }

  /**
   * 履歴の確定した公開の位置までの、イベントの行と関係のイベントの行を下書きに写す。付け替えの後、列を書く前に呼ぶ。
   *
   * @param positions 写す公開の位置の数（履歴の確定した位置の数と同じこと）
   */
  void restoreEvents(EventHistory history, int positions) {
    if (positions != history.positions()) {
      throw new IllegalStateException(
          "history holds " + history.positions() + " settled positions, not " + positions);
    }
    System.arraycopy(history.eventRows(), 0, event.draft, 0, positions * event.width);
    System.arraycopy(
        history.relationRows(),
        0,
        relation.draft,
        schema.fixedTokens() * relation.width,
        positions * relation.width);
    eventPrefix = positions;
  }

  /** 既に公開したトークンは CPU の下書きにも戻さず、新しい位置だけを同じ列のライターで符号化する。 */
  void writePublished(RowFacts facts, EventHistory.Bytes previous, Prefix prefix) {
    published = true;
    eventPrefix = facts.fromPosition;
    writeColumns(facts);
    batch.bindEvents(
        row,
        EventHistory.Bytes.publish(
            schema, previous, prefix, event.draft, eventScalars, relation.draft));
  }

  /** 列を書き終えた下書きの、確定した公開の位置 {@code to} までの行を履歴に写す。 */
  void saveEvents(EventHistory history, int to) {
    history.saveRows(
        event.draft,
        event.width,
        relation.draft,
        schema.fixedTokens() * relation.width,
        relation.width,
        to);
  }

  void prepareCodes(RowFacts facts) {
    for (int token = 0; token < tiles; token++) {
      int count = tile.draft[token * tile.width + handColumn];
      handCodes[token] = (byte) schema.tileCode(count, false, null, false);
    }
    if (candidate != null) candidate.prepare(facts, tile.draft, seat.draft, handCodes);
  }

  /**
   * 席のトークンのフィールドを書くライターを返す。
   *
   * @param relativeSeat 観測者から見た相対席
   * @return その席のトークンに付けたライター
   */
  public TokenWriter<SeatField> seat(int relativeSeat) {
    return seat.at(relativeSeat);
  }

  /**
   * 遷移のフィールドを書くライターを返す。
   *
   * @param transition 行の中の遷移の番号
   * @return その遷移に付けたライター
   */
  public TokenWriter<TransitionField> transition(int transition) {
    return this.transition.at(transition);
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
    roundScalars[field.ordinal()] = Bfloat16.fromFloat(value / field.scale(schema.variant()));
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
    short normalized = Bfloat16.fromFloat(value / field.scale(schema.variant()));
    switch (field.tensor()) {
      case SEAT_SCALAR ->
          seatScalars[Objects.checkIndex(index, seats) * seatScalarWidth + field.ordinal()] =
              normalized;
      case EVENT_SCALAR ->
          eventScalars[Objects.checkIndex(index, events) * eventScalarWidth + field.ordinal()] =
              normalized;
      case TRANSITION_SCALAR ->
          transitionScalars[
                  Objects.checkIndex(index, transitions) * transitionScalarWidth
                      + field.ordinal()] =
              normalized;
      default ->
          throw new IllegalArgumentException(field + " is not a seat, event or transition scalar");
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
          discardExact[
                  Objects.checkIndex(index, transitions) * discardExactWidth + field.ordinal()] =
              normalized;
      case ACTION_EXACT ->
          actionExact[
                  Objects.checkIndex(index, actions) * actionExactWidth
                      + (field instanceof WinExact win
                          ? schema.winFields().index(win)
                          : field.ordinal())] =
              normalized;
      default -> throw new IllegalArgumentException(field + " is not an exact field");
    }
  }

  /** 鳴き元の席に応じた局・席・牌の初期コードを次の遷移に書く。 */
  void startCodes(int transition, int source) {
    if (transition != nextCodeRow || transition >= transitions) {
      throw new IllegalStateException("Unexpected transition code row: " + transition);
    }
    int start = transition * keys;
    System.arraycopy(initialCodes[source + 1], 0, codes, start, tileStart);
    tileCodeStart = start + tileStart;
    System.arraycopy(handCodes, 0, codes, tileCodeStart, tiles);
    if (candidate != null) candidate.start(transition);
    nextCodeRow++;
  }

  void tileCode(int tile, int code) {
    codes[tileCodeStart + Objects.checkIndex(tile, tiles)] = (byte) code;
    if (candidate != null) candidate.mark(nextCodeRow - 1, tile, code);
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
      ActionRole role,
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
    actionColumns[at + ActionColumn.ROLE.ordinal()] = schema.roleIndex(role);
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
    if (nextCodeRow != transitions) {
      throw new IllegalStateException(
          "Missing transition code rows: " + nextCodeRow + " of " + transitions);
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
    int from = published ? eventPrefix : 0;
    event.copyTo(bytes, plan.offset(Tensor.EVENT_CATEGORY), firstEvent + from, from, events - from);
    transition.copyTo(
        bytes, plan.offset(Tensor.TRANSITION_CATEGORY), firstTransition, 0, transitions);
    int relations = plan.offset(Tensor.RELATION);
    int token = 0;
    for (int type = 0; type < Axis.TOKEN.ordinal(); type++) {
      Axis axis = AXES[type];
      int count = rowTokens(axis);
      int skip = axis == Axis.EVENT ? from : 0;
      relation.copyTo(
          bytes,
          relations,
          plan.tokenStart(axis) + rowTokenStart(axis) + skip,
          token + skip,
          count - skip);
      token += count;
    }
    int codeStart = plan.offset(Tensor.TRANSITION_CODE) + firstTransition * transitionCodeWidth;
    if (candidate == null) {
      bytes.put(codeStart, codes, 0, codeCount);
    } else {
      batch.writeBasicCodes(firstTransition, codes, codeCount);
      candidate.write(bytes, codeStart, codes, transitions);
    }
    put(shorts, Tensor.ROUND_SCALAR, row * roundScalarWidth, roundScalars, roundScalarWidth);
    put(shorts, Tensor.SEAT_SCALAR, seats * row * seatScalarWidth, seatScalars, seatScalars.length);
    shorts.put(
        element(Tensor.EVENT_SCALAR) + (firstEvent + from) * eventScalarWidth,
        eventScalars,
        from * eventScalarWidth,
        (events - from) * eventScalarWidth);
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
    batch.eventPrefix(row, eventPrefix);
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

  private byte[] codeBounds(boolean ceiling) {
    byte[] bounds = new byte[Limits.MAX_TRANSITIONS * keys];
    int covered = 0;
    for (KeyBlock kind : KeyBlock.values()) {
      var block = schema.block(kind);
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
            "transition "
                + at / keys
                + " key "
                + at % keys
                + " has a code outside its block: "
                + code);
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
      for (F field : layout.fields()) {
        FieldSpec spec = layout.spec(field);
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
      draft[base + column(field, Coding.ENUM)] =
          (byte) Ids.of(values[field.ordinal()], checked(field, value));
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
          field
              + " is a "
              + coding[field.ordinal()]
              + " field with "
              + repeats[field.ordinal()]
              + " repeats, written as "
              + expected);
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

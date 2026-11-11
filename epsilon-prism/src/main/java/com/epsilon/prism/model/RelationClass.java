package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.prism.hip.RelationCodes;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat.RelationRole;
import com.epsilon.prism.input.InputFormat;
import java.util.function.Supplier;

/**
 * 関係のバイアスのコード。トークンの組（クエリ i, キー j）ごとに 4 bit のコードを 7 セグメント分作り、下位のセグメントから 4 bit ずつ
 * INT16 の 2 ワードに詰める。{@code packedRelationScaledDotProductAttention} が読む形で、セグメント s のコード c はバイアスの表の
 * 行 16s + c を引く。整数の演算だけで作るので、CPU と GPU で同じ値になる。HIP の推論の K0 と、GPU の学習の自前のカーネル（{@link RelationCodes}、T0）
 * も同じ式で作る。T0 の結果は、この eager のコードとパディングの位置までビット単位で一致する。
 *
 * <p>属性の値は、入力形式が関係の属性に書く値のまま使う（0 はなし、番号は番号 + 1）。トークンの型は位置で決まる（{@link TokenType}）。
 *
 * <ol start="0">
 *   <li>型の組: 4·型(i) + 型(j)。
 *   <li>牌の関係の前半: 両方に牌があり、{@link MahjongTileRelation} のクラス c が 15 未満なら c + 1。
 *   <li>牌の関係の後半: 両方に牌があり、c が 15 以上なら c − 14。
 *   <li>席: 両方に持ち主があれば 1 + (持ち主(j) − 持ち主(i) + S) mod S、j だけにあれば S + 持ち主(j)、i だけにあれば 2S + 持ち主(i)。
 *       S は席の数で、持ち主は ID の値 1〜S。
 *   <li>立直: キーがイベントなら 1 + 前後(番号(j), 立直(j)) + 4·[番号(j) &gt; 最後の打牌(i)]。クエリがイベントでキーが席なら
 *       9 + 前後(番号(i), 立直(j))。前後は、立直が無ければ 0、立直の前・宣言・後で 1・2・3。最後の打牌の項は、i に最後の打牌があるときだけ
 *       足す。
 *   <li>時間: 両方がイベントなら 5 + 符号(δ)·段(|δ|)。δ = 番号(j) − 番号(i) で、段は |δ| が 0・1・2〜3・4〜7・8 以上のとき 0〜4。クエリが
 *       イベントでなくキーがイベントなら 10 + [d ≥ 5] + [d ≥ 9] + [d ≥ 17]。d = 番号(局) − 番号(j) は今からの距離で、直前のイベントが 1。
 *   <li>繋がり・安全度: 両方がイベントなら、i の繋がりが j のとき 1、j の繋がりが i のとき 2。席 → 牌で席が相手なら 3 + その相手から見た牌の
 *       安全度、牌 → 席で席が相手なら 7 + その相手から見た牌の安全度（安全度は ID の値 0〜3）。
 * </ol>
 *
 * <p>当てはまらないセグメントのコードは 0。パディングのトークン（行の局・席・牌とイベントより後ろの位置）は、属性が全て 0 のイベントとして
 * 同じ式で作り、Attention のマスクで外す。牌の関係の表は牌のトークンの組の表（T × T）で、属性の牌の値はトークンの番号 + 1。
 */
final class RelationClass {

  /** セグメントの数。 */
  static final int SEGMENTS = 7;

  /** 1 セグメントあたりの表の行数。 */
  static final int ENTRIES = 16;

  /** 関係のバイアスの表の行数。 */
  static final int ROWS = SEGMENTS * ENTRIES;

  /** 前半のセグメントに入る、牌の関係のクラスの数。 */
  private static final int FIRST_HALF = ENTRIES - 1;

  /** INT16 の 1 ワードが表せる値の数（2^16）。下位のワードの値が {@code WORD / 2} 以上なら、これを引いて 2 の補数に直す。 */
  private static final int WORD = 1 << 16;

  private static final int ROUND = TokenType.ROUND.ordinal();
  private static final int SEAT = TokenType.SEAT.ordinal();
  private static final int TILE = TokenType.TILE.ordinal();
  private static final int EVENT = TokenType.EVENT.ordinal();

  /** 自分の席を表す持ち主の ID（相対席 0 に 1 を足した値）。これより大きい ID は相手。 */
  private static final int SELF = 1;

  static {
    if (MahjongTileRelation.values().length > FIRST_HALF + ENTRIES - 1) {
      throw new IllegalStateException("tile relations do not fit two segments");
    }
    if (ROUND != 0) {
      throw new IllegalStateException("the round token must come first");
    }
  }

  private RelationClass() {}

  /** 自前のカーネルに渡す入力の並び。推論で WeightLayout が WeightPack のヘッダーに書く値と同じ。 */
  private static RelationCodes.Layout layout(InputFormat format) {
    return new RelationCodes.Layout(
        format.seats(),
        format.tiles(),
        NetworkInputs.ROW_WIDTH,
        RowColumn.FIRST_EVENT.ordinal(),
        RowColumn.EVENTS.ordinal(),
        format.relation().width(),
        format.relationColumn(RelationRole.TILE, 0),
        format.relationColumn(RelationRole.OWNER, 0),
        format.relationColumn(RelationRole.EVENT, 0),
        format.relationColumn(RelationRole.OWNER_RIICHI_EVENT, 0),
        format.relationColumn(RelationRole.OWNER_LAST_DISCARD_EVENT, 0),
        format.relationColumn(RelationRole.LINK, 0),
        format.relationColumn(RelationRole.SAFETY, 0),
        format.relationRepeats(RelationRole.SAFETY));
  }

  /**
   * コードを eager で作る（CPU、djl-rocm の Attention、推論で使う）。入力の関係の属性を [行, S] の位置へ散らしてから
   * {@link #codes(InputFormat, NDArray, NDArray, NDArray)} に渡す（パディングの位置の属性は 0）。
   *
   * @param inputs 入力
   * @param index 順伝播のインデックス
   * @return 詰めたコード [行, S, S, 2]（INT16。入力のマネージャーに付く）
   */
  static NDArray codes(NetworkInputs inputs, BatchIndex index) {
    int rows = inputs.rows();
    InputFormat format = inputs.format();
    NDArray attributes =
        NDArrays.scatterRows(
                inputs.get(Tensor.RELATION).toType(DataType.INT32, false),
                index.tokenSlots.array(),
                (long) rows * index.tokens)
            .reshape(rows, index.tokens, format.relation().width());
    return codes(format, attributes, index.tokenTypes.array(), index.tileRelations.array());
  }

  /**
   * コードを自前のカーネルで作る（GPU の学習で使う）。結果は {@link #codes(NetworkInputs, BatchIndex)} とビット単位で一致する。
   *
   * @param kernel コードを作るカーネル（学習のレーンの計算ストリームに積む）
   * @param inputs 入力
   * @param index 順伝播のインデックス
   * @return 詰めたコード [行, S, S, 2]（INT16。入力のマネージャーに付く）
   */
  static NDArray codes(RelationCodes kernel, NetworkInputs inputs, BatchIndex index) {
    Shape pairs = new Shape(inputs.rows(), index.tokens, index.tokens, RelationCodes.WORDS);
    NDArray codes = inputs.manager().create(pairs, DataType.INT16);
    kernel.codes(
        layout(inputs.format()),
        inputs.rows(),
        inputs.events(),
        index.tokens,
        inputs.get(Tensor.RELATION),
        inputs.get(Tensor.ROW_COLUMN),
        index.tileRelations.array(),
        codes);
    return codes;
  }

  /**
   * コードを作る。途中の配列は一時のマネージャーに付け、コードを作ったら捨てる（DJL はマネージャーを閉じるまで配列を持ち続ける）。
   *
   * @param format 入力形式（属性の列・席の数・牌のトークンの数を決める）
   * @param attributes 関係の属性 [行, S, 属性]（INT32）
   * @param tokenTypes 位置ごとのトークンの型 [S]（INT32）
   * @param tileRelations 牌の関係のクラス [T·T]（INT32）
   * @return 詰めたコード [行, S, S, 2]（INT16。属性のマネージャーに付く）
   */
  static NDArray codes(InputFormat format, NDArray attributes, NDArray tokenTypes, NDArray tileRelations) {
    NDManager owner = attributes.getManager();
    try (NDManager temporary = owner.newSubManager()) {
      attributes.tempAttach(temporary);
      tokenTypes.tempAttach(temporary);
      tileRelations.tempAttach(temporary);
      NDArray codes = new Pairs(format, attributes, tokenTypes, tileRelations).codes();
      codes.attach(owner);
      return codes;
    }
  }

  /**
   * 1 つのバッチのコードを、セグメントごとにワードへ足して作る。クエリの側 [行, S, 1] とキーの側 [行, 1, S] の属性、位置ごとの型、
   * 牌の関係の表を持つ。
   *
   * <p>セグメントの値 [行, S, S] はセグメントごとのマネージャーで作り、ワードに足したら、セグメントの中で作った配列と一緒に捨てる。
   * セグメントの中で作る [行, S, S] の配列は合わせて百を超える。コードができるまで全てを持つと、コードの数十倍のメモリをデバイスに置くことに
   * なる。セグメントごとに捨てれば、同時に持つのは下位と上位の 2 ワードと 1 セグメント分だけで済む。作ったばかりでほかから読まない
   * [行, S, S] の配列には、その場で書き換える演算を使う（値は新しい配列に書くときと同じになる）。
   */
  private static final class Pairs {
    private final InputFormat format;
    private final int seats;
    private final int tiles;
    private final NDManager manager;
    private final NDList sides = new NDList();
    private final Shape pair;
    private final NDArray typeQ;
    private final NDArray typeK;
    private final NDArray keyEvent;
    private final NDArray queryEvent;
    private final NDArray bothEvents;
    private final NDArray tileQ;
    private final NDArray tileK;
    private final NDArray ownerQ;
    private final NDArray ownerK;
    private final NDArray eventQ;
    private final NDArray eventK;
    private final NDArray riichiK;
    private final NDArray discardQ;
    private final NDArray linkQ;
    private final NDArray linkK;
    private final NDArray now;
    private final NDArray[] safetyQ;
    private final NDArray[] safetyK;
    private final NDArray tileRelations;

    /**
     * @param format 入力形式
     * @param attributes 関係の属性 [行, S, 属性]（INT32）
     * @param tokenTypes 位置ごとのトークンの型 [S]（INT32）
     * @param tileRelations 牌の関係のクラス [T·T]（INT32）
     */
    Pairs(InputFormat format, NDArray attributes, NDArray tokenTypes, NDArray tileRelations) {
      this.format = format;
      seats = format.seats();
      tiles = format.tiles();
      manager = attributes.getManager();
      long rows = attributes.getShape().get(0);
      long tokens = attributes.getShape().get(1);
      pair = new Shape(rows, tokens, tokens);
      typeQ = side(tokenTypes.reshape(1, tokens, 1));
      typeK = side(tokenTypes.reshape(1, 1, tokens));
      keyEvent = side(typeK.eq(EVENT));
      queryEvent = side(typeQ.eq(EVENT));
      bothEvents = side(queryEvent.logicalAnd(keyEvent));
      NDArray tile = column(attributes, RelationRole.TILE, 0);
      tileQ = side(query(tile));
      tileK = side(key(tile));
      NDArray owner = column(attributes, RelationRole.OWNER, 0);
      ownerQ = side(query(owner));
      ownerK = side(key(owner));
      NDArray event = column(attributes, RelationRole.EVENT, 0);
      eventQ = side(query(event));
      eventK = side(key(event));
      riichiK = side(key(column(attributes, RelationRole.OWNER_RIICHI_EVENT, 0)));
      discardQ = side(query(column(attributes, RelationRole.OWNER_LAST_DISCARD_EVENT, 0)));
      NDArray link = column(attributes, RelationRole.LINK, 0);
      linkQ = side(query(link));
      linkK = side(key(link));
      now =
          side(
              attributes
                  .get(new NDIndex(":, 0, {}", format.relationColumn(RelationRole.EVENT, 0)))
                  .reshape(rows, 1, 1));
      int opponents = format.relationRepeats(RelationRole.SAFETY);
      safetyQ = new NDArray[opponents];
      safetyK = new NDArray[opponents];
      for (int opponent = 0; opponent < opponents; opponent++) {
        NDArray safety = column(attributes, RelationRole.SAFETY, opponent);
        safetyQ[opponent] = side(query(safety));
        safetyK[opponent] = side(key(safety));
      }
      this.tileRelations = side(tileRelations);
    }

    /**
     * 下位のワード（型の組 + 16·牌の関係の前半 + 256·後半 + 4096·席）と上位のワード（立直 + 16·時間 + 256·繋がりと安全度）を作り、
     * INT16 の 2 ワードに詰める。下位のワードは INT16 に入るように 2 の補数に直す。
     */
    NDArray codes() {
      NDArray low = segment(this::tiles);
      add(low, segment(this::seat), ENTRIES * ENTRIES * ENTRIES);
      try (NDArray wraps = low.gte(WORD / 2);
          NDArray shift = integer(wraps)) {
        low.subi(shift.muli(WORD));
      }
      NDArray high = segment(this::riichi);
      add(high, segment(this::time), ENTRIES);
      add(high, segment(this::safety), ENTRIES * ENTRIES);
      NDArray words = NDArrays.stack(new NDList(low, high), 3);
      low.close();
      high.close();
      NDArray codes = words.toType(DataType.INT16, false);
      words.close();
      return codes;
    }

    /** セグメント 0・1・2: 型の組 + 16·牌の関係の前半 + 256·牌の関係の後半。 */
    private NDArray tiles() {
      NDArray bothTiles = tileQ.gt(0).logicalAnd(tileK.gt(0));
      NDArray relation;
      try (NDArray index = tileQ.sub(1).maximum(0).mul(tiles).add(tileK.sub(1).maximum(0));
          NDArray flat = index.reshape(-1)) {
        relation = NDArrays.gatherRows(tileRelations, flat).reshape(pair);
      }
      NDArray firstHalf = relation.lt(FIRST_HALF);
      NDArray low = keep(bothTiles.logicalAnd(firstHalf), relation.add(1));
      low.muli(ENTRIES).addi(typeQ.mul(4).add(typeK));
      try (NDArray second = relation.sub(FIRST_HALF - 1)) {
        keep(bothTiles.logicalAnd(firstHalf.logicalNot()), second);
        low.addi(second.muli(ENTRIES * ENTRIES));
      }
      return low;
    }

    /** セグメント 3: 席。 */
    private NDArray seat() {
      NDArray hasQ = ownerQ.gt(0);
      NDArray hasK = ownerK.gt(0);
      NDArray seat =
          keep(hasQ.logicalAnd(hasK), ownerK.sub(ownerQ).addi(seats).modi(seats).addi(1));
      try (NDArray keyOnly = select(hasK.logicalAnd(hasQ.logicalNot()), ownerK.add(seats))) {
        seat.addi(keyOnly);
      }
      try (NDArray queryOnly = select(hasQ.logicalAnd(hasK.logicalNot()), ownerQ.add(2 * seats))) {
        seat.addi(queryOnly);
      }
      return seat;
    }

    /** セグメント 4: 立直。 */
    private NDArray riichi() {
      NDArray code = integer(discardQ.gt(0).logicalAnd(eventK.gt(discardQ)));
      code.muli(4).addi(phase(eventK, riichiK).addi(1)).muli(integer(keyEvent));
      try (NDArray seatKey = phase(eventQ, riichiK).addi(9)) {
        code.addi(keep(queryEvent.logicalAnd(typeK.eq(SEAT)), seatKey));
      }
      return code;
    }

    /** セグメント 5: 時間。 */
    private NDArray time() {
      NDArray delta = eventK.sub(eventQ);
      NDArray step;
      try (NDArray magnitude = delta.abs()) {
        step = count(magnitude, 1, 2, 4, 8);
      }
      NDArray time;
      try (step) {
        time = keep(bothEvents, delta.signi().muli(step).addi(5));
      }
      NDArray ageStep = count(now.sub(eventK), 5, 9, 17).addi(10);
      try (NDArray age = select(queryEvent.logicalNot().logicalAnd(keyEvent), ageStep)) {
        return time.addi(age);
      }
    }

    /** セグメント 6: 繋がり・安全度。 */
    private NDArray safety() {
      NDArray forward = linkQ.gt(0).logicalAnd(linkQ.eq(eventK));
      NDArray backward =
          linkK.gt(0).logicalAnd(linkK.eq(eventQ)).logicalAnd(forward.logicalNot());
      NDArray code = integer(bothEvents.logicalAnd(forward));
      try (NDArray linked = integer(bothEvents.logicalAnd(backward))) {
        code.addi(linked.muli(2));
      }
      try (NDArray key = opponentSafety(ownerQ, safetyK).addi(3)) {
        NDArray seatToTile =
            typeQ.eq(SEAT).logicalAnd(typeK.eq(TILE)).logicalAnd(ownerQ.gt(SELF));
        code.addi(keep(seatToTile, key));
      }
      try (NDArray query = opponentSafety(ownerK, safetyQ).addi(7)) {
        NDArray tileToSeat =
            typeQ.eq(TILE).logicalAnd(typeK.eq(SEAT)).logicalAnd(ownerK.gt(SELF));
        code.addi(keep(tileToSeat, query));
      }
      return code;
    }

    /** セグメントの値を、セグメントごとのマネージャーで作る（セグメントの中で作った配列は、値を返す前に捨てる）。 */
    private NDArray segment(Supplier<NDArray> body) {
      try (NDManager scope = manager.newSubManager()) {
        sides.tempAttach(scope);
        NDArray value = body.get();
        value.attach(manager);
        return value;
      }
    }

    private NDArray side(NDArray array) {
      sides.add(array);
      return array;
    }

    /** 属性の列 [行, S]。 */
    private NDArray column(NDArray attributes, RelationRole field, int repeat) {
      return attributes.get(new NDIndex(":, :, {}", format.relationColumn(field, repeat)));
    }
  }

  /** クエリの側 [行, S, 1]。 */
  private static NDArray query(NDArray column) {
    return column.expandDims(2);
  }

  /** キーの側 [行, 1, S]。 */
  private static NDArray key(NDArray column) {
    return column.expandDims(1);
  }

  /** 真偽を 0/1 の INT32 にする。 */
  private static NDArray integer(NDArray condition) {
    return condition.toType(DataType.INT32, false);
  }

  /** ワードに値の {@code weight} 倍を足す。ワードと値はその場で書き換え、値は閉じる。 */
  private static void add(NDArray word, NDArray value, int weight) {
    try (value) {
      word.addi(value.muli(weight));
    }
  }

  /** 条件が真の所だけ値を残し、ほかを 0 にする。値をその場で書き換えるので、値は条件と同じ形で、作ったばかりの配列でなければならない。 */
  private static NDArray keep(NDArray condition, NDArray value) {
    try (NDArray mask = integer(condition)) {
      return value.muli(mask);
    }
  }

  /** 条件が真の所だけ値を残し、ほかを 0 にする。結果は新しい配列に書く（値は条件より小さい形でもよい）。 */
  private static NDArray select(NDArray condition, NDArray value) {
    try (NDArray mask = integer(condition)) {
      return value.mul(mask);
    }
  }

  /** 値がいくつの下限以上かを数える。Σ [値 ≥ 下限]（INT32）。 */
  private static NDArray count(NDArray value, int... bounds) {
    NDArray count = atLeast(value, bounds[0]);
    for (int bound = 1; bound < bounds.length; bound++) {
      try (NDArray more = atLeast(value, bounds[bound])) {
        count.addi(more);
      }
    }
    return count;
  }

  /** [値 ≥ 下限]（INT32）。 */
  private static NDArray atLeast(NDArray value, int bound) {
    try (NDArray condition = value.gte(bound)) {
      return integer(condition);
    }
  }

  /** イベントの番号が、立直のイベントの番号より前か後かを返す（立直なし 0、前 1、宣言 2、後 3）。 */
  private static NDArray phase(NDArray number, NDArray riichi) {
    NDArray phase;
    try (NDArray reached = number.gte(riichi)) {
      phase = integer(reached);
    }
    try (NDArray passed = number.gt(riichi);
        NDArray after = integer(passed)) {
      phase.addi(after);
    }
    return phase.addi(1).muli(integer(riichi.gt(0)));
  }

  /**
   * 相手の席から見た牌の安全度 [行, S, S] を返す（席が相手でなければ 0）。
   *
   * @param seatOwner 席の側の持ち主（クエリの側かキーの側）
   * @param safety 相手ごとの、牌の側（席と反対の側）の安全度
   * @return 安全度
   */
  private static NDArray opponentSafety(NDArray seatOwner, NDArray[] safety) {
    NDArray result = select(seatOwner.eq(SELF + 1), safety[0]);
    for (int opponent = 1; opponent < safety.length; opponent++) {
      try (NDArray chosen = select(seatOwner.eq(SELF + 1 + opponent), safety[opponent])) {
        result.addi(chosen);
      }
    }
    return result;
  }
}

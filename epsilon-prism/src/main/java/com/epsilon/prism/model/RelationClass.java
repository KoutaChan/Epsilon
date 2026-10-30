package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.ai.model.MahjongTileRelation;
import com.epsilon.core.Tile;
import com.epsilon.prism.hip.RelationCodes;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.RelationAttribute;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.Tensor;

/**
 * 関係の偏りの符号。トークンの対 (問い合わせ i, 鍵 j) ごとに 7 区画の 4 bit の符号を作り、下位の区画から 4 bit ずつ INT16 の
 * 2 語に詰める（packedRelationScaledDotProductAttention の形、区画 s の符号 c は表の行 16s + c を引く）。整数の演算だけで作るので、CPU と
 * GPU で同じ値になる。HIP の推論の K0 と、GPU の学習の自前の演算（{@link RelationCodes}、T0）も同じ式で作り、T0 はこの eager の符号と
 * 詰め物の位置までビットで同じ。
 *
 * <p>属性の値は {@link RelationAttribute} の書き方のまま（0 はなし、番号は番号 + 1）。型は位置で決まる（{@link TokenType}）。
 *
 * <ol start="0">
 *   <li>型の対: 4·型(i) + 型(j)。
 *   <li>牌の関係の前半: 両方に牌があり、{@link MahjongTileRelation} の類 c が 15 未満なら c + 1。
 *   <li>牌の関係の後半: 両方に牌があり、c が 15 以上なら c − 14。
 *   <li>席: 両方に持ち主があれば 1 + (持ち主(j) − 持ち主(i)) mod 4、j だけなら 4 + 持ち主(j)、i だけなら 8 + 持ち主(i)（持ち主は ID の値
 *       1〜4）。
 *   <li>立直: 鍵が出来事なら 1 + 前後(番号(j), 立直(j)) + 4·[番号(j) &gt; 最後の打牌(i)]、問い合わせが出来事で鍵が席なら 9 + 前後(番号(i),
 *       立直(j))。前後は立直がなければ 0、前・宣言・後で 1・2・3。最後の打牌の項は i に最後の打牌があるときだけ。
 *   <li>時間: 両方が出来事なら 5 + 符号(δ)·段(|δ|)（δ = 番号(j) − 番号(i)、段は |δ| が 0・1・2〜3・4〜7・8 以上で 0〜4）、問い合わせが出来事
 *       でなく鍵が出来事なら 10 + [d ≥ 5] + [d ≥ 9] + [d ≥ 17]（d = 番号(局) − 番号(j) は今からの距離で、直前の出来事が 1）。
 *   <li>繋がり・安全: 両方が出来事なら、i の繋がりが j なら 1、j の繋がりが i なら 2。席 → 牌で席が相手なら 3 + その相手から見た牌の安全、
 *       牌 → 席で席が相手なら 7 + その相手から見た牌の安全（安全は ID の値 0〜3）。
 * </ol>
 *
 * <p>当てはまらない区画の符号は 0。詰め物のトークン（行の局・席・牌と出来事より後ろの位置）は属性が全て 0 の出来事として同じ式で作り、
 * 注意の印が外す。
 */
final class RelationClass {

  /** 区画の数。 */
  static final int SEGMENTS = 7;

  /** 区画あたりの表の行。 */
  static final int ENTRIES = 16;

  /** 関係の偏りの表の行。 */
  static final int ROWS = SEGMENTS * ENTRIES;

  /** 前半の区画が持つ牌の関係の類の数。 */
  private static final int FIRST_HALF = ENTRIES - 1;

  /** INT16 の語の符号つきの上限（これ以上は 2 の補数に直す）。 */
  private static final int WORD = 1 << 16;

  private static final int ROUND = TokenType.ROUND.ordinal();
  private static final int SEAT = TokenType.SEAT.ordinal();
  private static final int TILE = TokenType.TILE.ordinal();
  private static final int EVENT = TokenType.EVENT.ordinal();

  /** 自分の席の持ち主の ID（相対席 0 + 1）。これより大きい ID が相手。 */
  private static final int SELF = 1;

  /** 自前の符号の演算に渡す入力の並び（推論の WeightLayout が WeightPack の頭に書く値と同じ）。 */
  private static final RelationCodes.Layout LAYOUT =
      new RelationCodes.Layout(
          TokenType.SEAT.perRow(),
          TokenType.TILE.perRow(),
          NetworkInputs.ROW_WIDTH,
          RowColumn.FIRST_EVENT.ordinal(),
          RowColumn.EVENTS.ordinal(),
          InputSchema.RELATION.width(),
          InputSchema.RELATION.column(RelationAttribute.TILE),
          InputSchema.RELATION.column(RelationAttribute.OWNER),
          InputSchema.RELATION.column(RelationAttribute.EVENT),
          InputSchema.RELATION.column(RelationAttribute.OWNER_RIICHI_EVENT),
          InputSchema.RELATION.column(RelationAttribute.OWNER_LAST_DISCARD_EVENT),
          InputSchema.RELATION.column(RelationAttribute.LINK),
          InputSchema.RELATION.column(RelationAttribute.SAFETY, 0),
          RelationAttribute.SAFETY.spec().repeats());

  static {
    if (MahjongTileRelation.values().length > FIRST_HALF + ENTRIES - 1) {
      throw new IllegalStateException("tile relations do not fit two segments");
    }
    if (ROUND != 0) {
      throw new IllegalStateException("the round token must come first");
    }
  }

  private RelationClass() {}

  /**
   * 符号を eager で作る（CPU・djl-rocm の注意・推論）。入力の関係の属性を [行, S] の位置へ散らしてから
   * {@link #codes(NDArray, NDArray, NDArray)} に渡す（詰め物の位置の属性は 0）。
   *
   * @param inputs 入力
   * @param index 順伝播の索引
   * @return 詰めた符号 [行, S, S, 2]（INT16。入力の管理元に付く）
   */
  static NDArray codes(NetworkInputs inputs, BatchIndex index) {
    int rows = inputs.rows();
    NDArray attributes =
        NDArrays.scatterRows(
                inputs.get(Tensor.RELATION).toType(DataType.INT32, false),
                index.tokenSlots,
                (long) rows * index.tokens)
            .reshape(rows, index.tokens, InputSchema.RELATION.width());
    return codes(attributes, index.tokenTypes, index.tileRelations);
  }

  /**
   * 符号を自前の演算で作る（GPU の学習。{@link #codes(NetworkInputs, BatchIndex)} とビットで同じ）。
   *
   * @param kernel 符号の演算（学習のレーンの計算ストリームに積む）
   * @param inputs 入力
   * @param index 順伝播の索引
   * @return 詰めた符号 [行, S, S, 2]（INT16。入力の管理元に付く）
   */
  static NDArray codes(RelationCodes kernel, NetworkInputs inputs, BatchIndex index) {
    Shape pairs = new Shape(inputs.rows(), index.tokens, index.tokens, RelationCodes.WORDS);
    NDArray codes = inputs.manager().create(pairs, DataType.INT16);
    kernel.codes(
        LAYOUT,
        inputs.rows(),
        inputs.events(),
        index.tokens,
        inputs.get(Tensor.RELATION),
        inputs.get(Tensor.ROW_COLUMN),
        index.tileRelations,
        codes);
    return codes;
  }

  /**
   * 符号を作る。
   *
   * @param attributes 関係の属性 [行, S, 属性]（INT32）
   * @param tokenTypes 位置ごとの型 [S]（INT32）
   * @param tileRelations 牌の関係の類 [34·34]（INT32）
   * @return 詰めた符号 [行, S, S, 2]（INT16。属性の管理元に付く）
   */
  static NDArray codes(NDArray attributes, NDArray tokenTypes, NDArray tileRelations) {
    // 間の [行, S, S] の配列（約 90 個）は一時の管理元に付けて、符号を作ったらすぐ捨てる（DJL は管理元を閉じるまで配列を持つ）
    NDManager owner = attributes.getManager();
    try (NDManager temporary = owner.newSubManager()) {
      attributes.tempAttach(temporary);
      tokenTypes.tempAttach(temporary);
      tileRelations.tempAttach(temporary);
      NDArray codes = build(attributes, tokenTypes, tileRelations);
      codes.attach(owner);
      return codes;
    }
  }

  /** 符号を作る（{@link #codes(NDArray, NDArray, NDArray)}。配列は入力の管理元に付く）。 */
  private static NDArray build(NDArray attributes, NDArray tokenTypes, NDArray tileRelations) {
    long rows = attributes.getShape().get(0);
    long tokens = attributes.getShape().get(1);
    Shape pair = new Shape(rows, tokens, tokens);
    NDArray typeQ = tokenTypes.reshape(1, tokens, 1);
    NDArray typeK = tokenTypes.reshape(1, 1, tokens);

    NDArray tile = column(attributes, RelationAttribute.TILE, 0);
    NDArray owner = column(attributes, RelationAttribute.OWNER, 0);
    NDArray event = column(attributes, RelationAttribute.EVENT, 0);
    NDArray riichi = column(attributes, RelationAttribute.OWNER_RIICHI_EVENT, 0);
    NDArray lastDiscard = column(attributes, RelationAttribute.OWNER_LAST_DISCARD_EVENT, 0);
    NDArray link = column(attributes, RelationAttribute.LINK, 0);
    NDArray now =
        attributes
            .get(new NDIndex(":, 0, {}", InputSchema.RELATION.column(RelationAttribute.EVENT)))
            .reshape(rows, 1, 1);

    // 0: 型の対
    NDArray typePair = typeQ.mul(4).add(typeK).broadcast(pair);

    // 1・2: 牌の関係
    NDArray tileQ = query(tile);
    NDArray tileK = key(tile);
    NDArray bothTiles = tileQ.gt(0).logicalAnd(tileK.gt(0));
    NDArray relationIndex =
        tileQ.sub(1).maximum(0).mul(Tile.NUM_TILE_TYPES).add(tileK.sub(1).maximum(0));
    NDArray relation =
        NDArrays.gatherRows(tileRelations, relationIndex.reshape(-1)).reshape(pair);
    NDArray firstHalf = relation.lt(FIRST_HALF);
    NDArray tileFirst = select(bothTiles.logicalAnd(firstHalf), relation.add(1));
    NDArray tileSecond =
        select(bothTiles.logicalAnd(firstHalf.logicalNot()), relation.sub(FIRST_HALF - 1));

    // 3: 席
    NDArray ownerQ = query(owner);
    NDArray ownerK = key(owner);
    NDArray hasQ = ownerQ.gt(0);
    NDArray hasK = ownerK.gt(0);
    NDArray seat =
        select(hasQ.logicalAnd(hasK), ownerK.sub(ownerQ).add(4).mod(4).add(1))
            .add(select(hasK.logicalAnd(hasQ.logicalNot()), ownerK.add(4)))
            .add(select(hasQ.logicalAnd(hasK.logicalNot()), ownerQ.add(8)));

    // 4: 立直
    NDArray eventQ = query(event);
    NDArray eventK = key(event);
    NDArray riichiK = key(riichi);
    NDArray discardQ = query(lastDiscard);
    NDArray keyEvent = typeK.eq(EVENT);
    NDArray queryEvent = typeQ.eq(EVENT);
    NDArray afterDiscard = discardQ.gt(0).logicalAnd(eventK.gt(discardQ));
    NDArray riichiCode =
        select(keyEvent, phase(eventK, riichiK).add(1).add(integer(afterDiscard).mul(4)))
            .add(select(queryEvent.logicalAnd(typeK.eq(SEAT)), phase(eventQ, riichiK).add(9)));

    // 5: 時間
    NDArray bothEvents = queryEvent.logicalAnd(keyEvent);
    NDArray delta = eventK.sub(eventQ);
    NDArray magnitude = delta.abs();
    NDArray step =
        integer(magnitude.gte(1))
            .add(integer(magnitude.gte(2)))
            .add(integer(magnitude.gte(4)))
            .add(integer(magnitude.gte(8)));
    NDArray age = now.sub(eventK);
    NDArray ageStep =
        integer(age.gte(5)).add(integer(age.gte(9))).add(integer(age.gte(17))).add(10);
    NDArray time =
        select(bothEvents, delta.sign().mul(step).add(5))
            .add(select(queryEvent.logicalNot().logicalAnd(keyEvent), ageStep));

    // 6: 繋がり・安全
    NDArray linkQ = query(link);
    NDArray linkK = key(link);
    NDArray forward = linkQ.gt(0).logicalAnd(linkQ.eq(eventK));
    NDArray backward = linkK.gt(0).logicalAnd(linkK.eq(eventQ)).logicalAnd(forward.logicalNot());
    NDArray linkCode =
        integer(bothEvents.logicalAnd(forward))
            .add(integer(bothEvents.logicalAnd(backward)).mul(2));
    NDArray safetyOfKey = opponentSafety(attributes, ownerQ, true);
    NDArray safetyOfQuery = opponentSafety(attributes, ownerK, false);
    NDArray seatToTile = typeQ.eq(SEAT).logicalAnd(typeK.eq(TILE)).logicalAnd(ownerQ.gt(SELF));
    NDArray tileToSeat = typeQ.eq(TILE).logicalAnd(typeK.eq(SEAT)).logicalAnd(ownerK.gt(SELF));
    NDArray safetyCode =
        linkCode
            .add(select(seatToTile, safetyOfKey.add(3)))
            .add(select(tileToSeat, safetyOfQuery.add(7)));

    NDArray low =
        typePair
            .add(tileFirst.mul(ENTRIES))
            .add(tileSecond.mul(ENTRIES * ENTRIES))
            .add(seat.mul(ENTRIES * ENTRIES * ENTRIES));
    low = low.sub(integer(low.gte(WORD / 2)).mul(WORD));
    NDArray high =
        riichiCode.add(time.mul(ENTRIES)).add(safetyCode.mul(ENTRIES * ENTRIES)).broadcast(pair);
    return NDArrays.stack(new NDList(low.broadcast(pair), high), 3).toType(DataType.INT16, false);
  }

  /** 属性の列 [行, S]。 */
  private static NDArray column(NDArray attributes, RelationAttribute field, int repeat) {
    return attributes.get(new NDIndex(":, :, {}", InputSchema.RELATION.column(field, repeat)));
  }

  /** 問い合わせの側 [行, S, 1]。 */
  private static NDArray query(NDArray column) {
    return column.expandDims(2);
  }

  /** 鍵の側 [行, 1, S]。 */
  private static NDArray key(NDArray column) {
    return column.expandDims(1);
  }

  /** 真偽を 0/1 の INT32 にする。 */
  private static NDArray integer(NDArray condition) {
    return condition.toType(DataType.INT32, false);
  }

  /** 条件が真の所だけ値を残す（ほかは 0）。 */
  private static NDArray select(NDArray condition, NDArray value) {
    return value.mul(integer(condition));
  }

  /** 出来事の番号 n の、立直の番号 r との前後（立直なし 0、前 1、宣言 2、後 3）。 */
  private static NDArray phase(NDArray number, NDArray riichi) {
    return integer(riichi.gt(0))
        .mul(integer(number.gte(riichi)).add(integer(number.gt(riichi))).add(1));
  }

  /**
   * 相手の席から見た牌の安全。
   *
   * @param attributes 関係の属性
   * @param seatOwner 席の側の持ち主（問い合わせの側か鍵の側）
   * @param tileIsKey 牌が鍵の側なら true
   * @return 安全の値 [行, S, S]（席が相手でなければ 0）
   */
  private static NDArray opponentSafety(NDArray attributes, NDArray seatOwner, boolean tileIsKey) {
    NDArray result = null;
    for (int opponent = 0; opponent < RelationAttribute.SAFETY.spec().repeats(); opponent++) {
      NDArray safety = column(attributes, RelationAttribute.SAFETY, opponent);
      NDArray side = tileIsKey ? key(safety) : query(safety);
      NDArray chosen = select(seatOwner.eq(SELF + 1 + opponent), side);
      result = result == null ? chosen : result.add(chosen);
    }
    return result;
  }
}

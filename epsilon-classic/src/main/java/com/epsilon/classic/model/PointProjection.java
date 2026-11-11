package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.ClassicFields.ActionInt;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicFields.WinFact;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.Tensor;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;

/**
 * 和了したときの点数の計算（Major だけ）。公開されている和了の情報（翻・符のコード・役満の倍率）から、和了したときの点の移動・順位・
 * 半荘の進み方を、重みを持たない式で求める。求めるのは、すぐできる和了（ロン・ツモの合法手）と、遷移した後の待ちの 8 通り（下家・対面・
 * 上家からのロンとツモ、それぞれを赤の 5 で和了る場合）。点は 100 点単位で扱う。
 *
 * <p>1 つの和了の結果は 26 個の値で、順に、和了の印、ロン、ツモ、放銃者（相対席の one-hot 4、ロンのときだけ）、点の移動（相対席ではなく
 * 絶対席の 4、/ 100000）、和了の後の自分と各席の点差（相対席の 4、/ 100000）、和了の後の順位の one-hot 4、終局、親の連荘、次局へ進む、
 * 西入、終局の効用、待ちの印、赤の 5 の印。このうち、ロンとツモの二択へ直接足すのは終局・終局の効用・順位の one-hot の 6 個。役の無い和了と、
 * 包が成り立つ和了は、値を全て 0 にする。
 *
 * <p>終局になるのは、次のどれかのとき。誰かが飛んだとき、西場の最後の局で子が和了したとき、南場の最後の局以降で終わる条件を満たしたとき
 * （親の和了なら親がトップで返し点以上、子の和了なら誰かが返し点以上）。飛び・返し点・トップは、和了した席が場の供託（立直の打牌の後の
 * 待ちでは自分の出す 1 本も）を受け取った後の点で決める。終局の効用は、天鳳の効用ならラスの −1 だけを使い、ほかの効用では順位の効用を
 * 使う。
 */
final class PointProjection extends AbstractBlock {

  /** 和了 1 つ分の結果の値の数。 */
  static final int FEATURES = 26;

  /** ロンとツモの二択へ直接足す値の数。 */
  static final int GATE_FEATURES = 6;

  /** 待ちの和了の仕方の数。 */
  static final int WAIT_CASES = 8;

  private static final int FU_CODES = 12;
  private static final int HAN = 14;
  private static final int YAKUMAN = 16;
  private static final int TABLE = YAKUMAN * HAN * FU_CODES;
  private static final int SEATS = ClassicSchema.SEATS;
  private static final int UNIT = 100;
  private static final float SCORE_SCALE = 1000f;

  /** 待ちの和了の仕方ごとに、ロンなら 1、ツモなら 0。 */
  private static final int[] WAIT_RON = {1, 1, 1, 0, 1, 1, 1, 0};

  /** 待ちの和了の仕方ごとの放銃者の相対席。ツモなら 0。 */
  private static final int[] WAIT_SOURCE = {1, 2, 3, 0, 1, 2, 3, 0};

  /** 待ちの和了の仕方ごとに、赤の 5 で和了るなら 1。 */
  private static final int[] WAIT_AKA = {0, 0, 0, 0, 1, 1, 1, 1};

  private final Variant variant;
  private final float[] terminalUtility;
  private final FieldLayout<WinFact> factLayout;
  private final int playerSeat;
  private final int dealerSeat;
  private final int kyoku;
  private final int honba;
  private final int kyotaku;
  private final int source;
  private final int type;

  /**
   * [役満の倍率, 翻, 符のコード] で引く点の表（100 点単位、INT32）。この表から始まる 4 つの表は、順に、子のロンの点、親のロンの点、
   * 子のツモで子 1 人が払う点、子のツモで親が払う点（親のツモで子 1 人が払う点と同じ）。
   */
  private NDArray childRon;

  private NDArray dealerRon;
  private NDArray tsumoChild;
  private NDArray tsumoDealer;

  /**
   * 待ちの和了の仕方ごとの値 [1, 8]（INT32）。この配列から始まる 5 つの配列は、順に、ロンの印・ツモの印・放銃者の相対席・待ちの和了の印・
   * 赤の 5 で和了る印。
   */
  private NDArray waitRon;

  private NDArray waitTsumo;
  private NDArray waitSources;
  private NDArray waitSnapshots;
  private NDArray waitAka;

  /** [順位] 終局の効用（FP32）。 */
  private NDArray terminalUtilities;

  /** [1, 1, 4] 席の番号 0〜3（INT32）。 */
  private NDArray seatIds;

  PointProjection(Variant variant, RankUtility utility) {
    this.variant = variant;
    ClassicSchema schema = ClassicSchema.of(Architecture.MAJOR);
    FieldLayout<RoundInt> round = schema.layout(RoundInt.class);
    factLayout = schema.layout(WinFact.class);
    playerSeat = round.offset(RoundInt.PLAYER_SEAT);
    dealerSeat = round.offset(RoundInt.DEALER_RELATIVE_SEAT);
    kyoku = round.offset(RoundInt.KYOKU_INDEX);
    honba = round.offset(RoundInt.HONBA);
    kyotaku = round.offset(RoundInt.KYOTAKU);
    source = round.offset(RoundInt.SOURCE_PLAYER_RELATIVE_SEAT);
    type = schema.layout(ActionInt.class).offset(ActionInt.TYPE);
    terminalUtility = new float[SEATS];
    for (int rank = 0; rank < SEATS; rank++) {
      terminalUtility[rank] =
          utility == RankUtility.TENHOU ? (rank == SEATS - 1 ? -1f : 0f) : (float) utility.value(variant, rank);
    }
  }

  /**
   * すぐできる和了の合法手（ロンとツモ）の結果を求める。ほかの合法手の結果は 0。
   *
   * @return 結果 [行, 合法手, ...]
   */
  Projection actions(Pass pass, ClassicBatch batch) {
    int rows = batch.rows();
    NDArray state = batch.tensor(Tensor.STATE_CATEGORIES);
    NDArray ledger = batch.tensor(Tensor.POINT_LEDGER_100).toType(DataType.INT32, false).reshape(rows, 1, SEATS);
    NDArray winFacts = batch.tensor(Tensor.ACTION_WIN_FACTS).toType(DataType.INT32, false);
    NDArray types = batch.tensor(Tensor.ACTION_CATEGORIES).get("..., {}", type).toType(DataType.INT32, false);
    NDArray ron = types.eq(ActionType.RON_AGARI.ordinal() + 1).toType(DataType.INT32, false);
    NDArray tsumo = types.eq(ActionType.TSUMO_AGARI.ordinal() + 1).toType(DataType.INT32, false);
    NDArray zeros = ron.mul(0);
    NDArray sources = field(state, source).sub(1).reshape(rows, 1).mul(ron);
    return project(pass, ledger, winFacts, state, ron, tsumo, sources, zeros, zeros, zeros);
  }

  /**
   * 待ちの 8 通りの和了の結果を求める。
   *
   * @param ledger 点 [待ち, 4]（絶対席、100 点単位）
   * @param state 待ちごとの、状態の整数の列 [待ち, 列]
   * @param waitFacts 待ちごとの、ロンとツモそれぞれの和了の情報 [待ち, 2, 列]
   * @param aka 赤の 5 で和了れるなら 1 [待ち]（INT32）
   * @param riichi 遷移が立直の打牌なら 1 [待ち]（INT32）
   * @return 結果 [待ち, 8, ...]
   */
  Projection waits(Pass pass, NDArray ledger, NDArray state, NDArray waitFacts, NDArray aka, NDArray riichi) {
    long count = state.getShape().get(0);
    NDArray ronFacts = waitFacts.get(":, 0, :");
    NDArray tsumoFacts = waitFacts.get(":, 1, :");
    NDArray ronRed = red(ronFacts, aka);
    NDArray tsumoRed = red(tsumoFacts, aka);
    NDArray cases =
        NDArrays.stack(new NDList(ronFacts, ronFacts, ronFacts, tsumoFacts, ronRed, ronRed, ronRed, tsumoRed), 1);
    return project(
        pass,
        ledger.reshape(count, 1, SEATS),
        cases,
        state,
        pass.copy(waitRon),
        pass.copy(waitTsumo),
        pass.copy(waitSources),
        riichi.reshape(count, 1),
        pass.copy(waitSnapshots),
        aka.reshape(count, 1).mul(waitAka));
  }

  /** 赤の 5 で和了る場合の和了の情報を作る。赤の 5 で和了れるのはそれがまだ見えていないときだけで、翻は 1 つ増える。 */
  private NDArray red(NDArray winFacts, NDArray aka) {
    NDList fields = new NDList();
    for (WinFact fact : factLayout.fields()) {
      NDArray value = winFacts.get(":, {}", factLayout.offset(fact));
      fields.add(
          switch (fact) {
            case VALID -> value.mul(aka);
            case HAN_WITHOUT_URA -> value.add(1);
            default -> value;
          });
    }
    return NDArrays.stack(fields, 1);
  }

  /**
   * 和了の結果を式で求める。配列の先頭の 2 軸は [和了の行 N, 和了の候補 M]。
   *
   * @param ledger 点 [N, 1, 4]（INT32）
   * @param winFacts 和了の情報 [N, M, 列]（INT32）
   * @param state 状態の整数の列 [N, 列]
   * @param ron ロンなら 1 [N か 1, M]
   * @param tsumo ツモなら 1 [N か 1, M]
   * @param sources 放銃者の相対席 [N か 1, M]
   * @param riichi 和了の前に立直の供託を払うなら 1 [N, 1 か M]
   * @param snapshot 待ちの和了なら 1 [N か 1, M]
   * @param aka 赤の 5 で和了るなら 1 [N か 1, M]
   */
  private Projection project(
      Pass pass,
      NDArray ledger,
      NDArray winFacts,
      NDArray state,
      NDArray ron,
      NDArray tsumo,
      NDArray sources,
      NDArray riichi,
      NDArray snapshot,
      NDArray aka) {
    NDManager working = pass.working();
    long rows = state.getShape().get(0);
    NDArray seat = field(state, playerSeat).sub(1).reshape(rows, 1);
    NDArray dealer = field(state, dealerSeat).sub(1).reshape(rows, 1);
    NDArray round = field(state, kyoku).sub(1).reshape(rows, 1);
    NDArray honbaSticks = field(state, honba).reshape(rows, 1);
    NDArray deposits = field(state, kyotaku).reshape(rows, 1);

    NDArray valid =
        fact(winFacts, WinFact.VALID)
            .eq(1)
            .logicalAnd(fact(winFacts, WinFact.REQUIRES_PAO_CORRECTION).eq(0))
            .logicalAnd(ron.add(tsumo).gt(0))
            .toType(DataType.INT32, false);
    NDArray index =
        fact(winFacts, WinFact.YAKUMAN_MULTIPLIER)
            .mul(HAN * FU_CODES)
            .add(fact(winFacts, WinFact.HAN_WITHOUT_URA).minimum(HAN - 1).mul(FU_CODES))
            .add(fact(winFacts, WinFact.FU_CODE))
            .toType(DataType.INT64, false);
    NDArray dealerWins = dealer.eq(0).toType(DataType.INT32, false);
    NDArray childWins = dealerWins.neg().add(1);
    NDArray ronPayment =
        childRon
            .take(working, index)
            .mul(childWins)
            .add(dealerRon.take(working, index).mul(dealerWins))
            .add(honbaSticks.mul(variant.ronHonba() / UNIT));
    NDArray childShare = tsumoChild.take(working, index).add(honbaSticks.mul(Variant.HONBA / UNIT));
    NDArray dealerShare = tsumoDealer.take(working, index).add(honbaSticks.mul(Variant.HONBA / UNIT));
    NDArray received =
        dealerShare
            .mul(SEATS - 1)
            .mul(dealerWins)
            .add(dealerShare.add(childShare.mul(SEATS - 2)).mul(childWins));

    NDArray winner = oneHot(seat, DataType.INT32);
    NDArray payer = oneHot(seat.add(sources).mod(SEATS), DataType.INT32);
    NDArray dealerMask = oneHot(seat.add(dealer).mod(SEATS), DataType.INT32);
    NDArray ronDelta = winner.sub(payer).mul(ronPayment.expandDims(2));
    NDArray dealerShareBySeat = dealerShare.expandDims(2);
    NDArray childWinnerPays =
        dealerMask.mul(dealerShareBySeat).add(dealerMask.neg().add(1).mul(childShare.expandDims(2)));
    NDArray pays = dealerShareBySeat.mul(dealerWins.expandDims(2)).add(childWinnerPays.mul(childWins.expandDims(2)));
    NDArray tsumoDelta = winner.mul(received.expandDims(2).add(pays)).sub(pays);
    NDArray active = valid.expandDims(2);
    NDArray settlement =
        ronDelta.mul(ron.expandDims(2)).add(tsumoDelta.mul(tsumo.expandDims(2))).mul(active);
    int deposit = Variant.DEPOSIT / UNIT;
    NDArray finals =
        ledger
            .sub(winner.mul(riichi.expandDims(2)).mul(deposit))
            .add(settlement)
            .add(winner.mul(deposits.add(riichi).mul(deposit).expandDims(2)));
    NDArray self = finals.mul(winner).sum(new int[] {2});
    NDArray finalRank = rank(finals, self, seat);
    NDArray rankOneHot = oneHot(finalRank, DataType.FLOAT32);

    int threshold = variant.returnPoints() / UNIT;
    NDArray dealerContinues = dealer.eq(0);
    NDArray southOrLater = round.gte(variant.lastRound());
    NDArray finishes =
        dealerContinues
            .logicalAnd(self.gte(threshold))
            .logicalAnd(finalRank.eq(0))
            .logicalOr(dealerContinues.logicalNot().logicalAnd(finals.max(new int[] {2}).gte(threshold)));
    NDArray activeFloat = valid.toType(DataType.FLOAT32, false);
    NDArray end =
        finals
            .min(new int[] {2})
            .lt(0)
            .logicalOr(dealerContinues.logicalNot().logicalAnd(round.gte(variant.finalRound())))
            .logicalOr(southOrLater.logicalAnd(finishes))
            .toType(DataType.FLOAT32, false)
            .mul(activeFloat);
    NDArray notEnd = activeFloat.sub(end);
    NDArray dealerRepeat = notEnd.mul(dealerContinues.toType(DataType.FLOAT32, false));
    NDArray westExtension =
        notEnd
            .mul(dealerContinues.logicalNot().toType(DataType.FLOAT32, false))
            .mul(southOrLater.toType(DataType.FLOAT32, false));
    NDArray nextRound = notEnd.sub(dealerRepeat).sub(westExtension);
    NDArray utility = terminalUtilities.take(working, finalRank.toType(DataType.INT64, false)).mul(end);

    NDArray zero = activeFloat.mul(0);
    NDArray sourceOneHot =
        oneHot(sources, DataType.FLOAT32)
            .mul(ron.toType(DataType.FLOAT32, false).expandDims(2))
            .add(zero.expandDims(2));
    NDList gaps = new NDList();
    for (int relative = 0; relative < SEATS; relative++) {
      gaps.add(
          self.sub(relativeScore(finals, seat, relative)).toType(DataType.FLOAT32, false).div(SCORE_SCALE).expandDims(2));
    }
    NDArray activeVector = active.toType(DataType.FLOAT32, false);
    NDArray features =
        NDArrays.concat(
                new NDList(
                    activeVector,
                    flag(ron, zero),
                    flag(tsumo, zero),
                    sourceOneHot,
                    settlement.toType(DataType.FLOAT32, false).div(SCORE_SCALE),
                    NDArrays.concat(gaps, 2),
                    rankOneHot,
                    end.expandDims(2),
                    dealerRepeat.expandDims(2),
                    nextRound.expandDims(2),
                    westExtension.expandDims(2),
                    utility.expandDims(2),
                    flag(snapshot, zero),
                    flag(aka, zero)),
                2)
            .mul(activeVector);
    NDArray gates =
        NDArrays.concat(new NDList(end.expandDims(2), utility.expandDims(2), rankOneHot), 2).mul(activeVector);
    return new Projection(features, gates, activeFloat);
  }

  /** 0 か 1 の値を [N, M, 1] の FP32 にする。 */
  private static NDArray flag(NDArray value, NDArray zero) {
    return value.toType(DataType.FLOAT32, false).add(zero).expandDims(2);
  }

  private NDArray fact(NDArray winFacts, WinFact fact) {
    return winFacts.get("..., {}", factLayout.offset(fact));
  }

  private static NDArray field(NDArray state, int offset) {
    return state.get(":, {}", offset).toType(DataType.INT32, false);
  }

  /** 席の番号（0〜3）を末尾の軸の one-hot にする。 */
  private static NDArray oneHot(NDArray seats, DataType type) {
    int axis = seats.getShape().dimension();
    NDList masks = new NDList();
    for (int seat = 0; seat < SEATS; seat++) {
      masks.add(seats.eq(seat).toType(type, false).expandDims(axis));
    }
    return NDArrays.concat(masks, axis);
  }

  /** 和了した席から見て、指定した相対席にいる人の点 [N, M] を返す。 */
  private static NDArray relativeScore(NDArray scores, NDArray seat, int relative) {
    NDArray result = scores.get("..., 0").mul(0);
    for (int player = 0; player < SEATS; player++) {
      int absolute = (player + relative) % SEATS;
      result = result.add(scores.get("..., {}", absolute).mul(seat.eq(player).toType(DataType.INT32, false)));
    }
    return result;
  }

  /** 和了した席の順位を返す（0 始まり。同点なら起家に近い席が上）。 */
  private NDArray rank(NDArray scores, NDArray self, NDArray seat) {
    NDArray own = self.expandDims(2);
    return scores
        .gt(own)
        .logicalOr(scores.eq(own).logicalAnd(seat.expandDims(2).gt(seatIds)))
        .toType(DataType.INT32, false)
        .sum(new int[] {2});
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape cases = new Shape(1, WAIT_CASES);
    int[] tsumo = new int[WAIT_CASES];
    int[] snapshots = new int[WAIT_CASES];
    for (int index = 0; index < WAIT_CASES; index++) {
      tsumo[index] = 1 - WAIT_RON[index];
      snapshots[index] = 1;
    }
    waitRon = manager.create(WAIT_RON, cases);
    waitTsumo = manager.create(tsumo, cases);
    waitSources = manager.create(WAIT_SOURCE, cases);
    waitSnapshots = manager.create(snapshots, cases);
    waitAka = manager.create(WAIT_AKA, cases);
    terminalUtilities = manager.create(terminalUtility);
    int[] seats = new int[SEATS];
    for (int seat = 0; seat < SEATS; seat++) {
      seats[seat] = seat;
    }
    seatIds = manager.create(seats, new Shape(1, 1, SEATS));
    int[] childRonTable = new int[TABLE];
    int[] dealerRonTable = new int[TABLE];
    int[] childShares = new int[TABLE];
    int[] dealerShares = new int[TABLE];
    for (int yakuman = 0; yakuman < YAKUMAN; yakuman++) {
      for (int han = 0; han < HAN; han++) {
        for (int code = 0; code < FU_CODES; code++) {
          int index = (yakuman * HAN + han) * FU_CODES + code;
          int fu = switch (code) {
            case 0 -> 0;
            case 1 -> 20;
            case 2 -> 25;
            default -> code * 10;
          };
          int base = yakuman > 0 || (han >= 1 && fu >= 20) ? ScoreMath.base(han, fu, yakuman) : 0;
          if (base > 0) {
            childRonTable[index] = ScoreMath.ron(base, false) / UNIT;
            dealerRonTable[index] = ScoreMath.ron(base, true) / UNIT;
            childShares[index] = ScoreMath.tsumo(base, false, false) / UNIT;
            dealerShares[index] = ScoreMath.tsumo(base, true, false) / UNIT;
          }
        }
      }
    }
    childRon = manager.create(childRonTable);
    dealerRon = manager.create(dealerRonTable);
    tsumoChild = manager.create(childShares);
    tsumoDealer = manager.create(dealerShares);
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use actions or waits");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, FEATURES)};
  }

  /**
   * 和了の結果。
   *
   * @param features 26 個の値 [N, M, 26]（FP32）
   * @param gates 二択へ直接足す 6 個の値 [N, M, 6]（FP32）
   * @param valid 役があり、包が成り立たない和了なら 1 [N, M]（FP32）
   */
  record Projection(NDArray features, NDArray gates, NDArray valid) {}
}

package com.epsilon.major.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.ScorePayments;
import com.epsilon.major.ai.decision.input.DecisionFeatureCodec;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;

/** 公開情報だけの和了事実から、点棒移動・順位・半荘遷移の固定特徴量を計算する。 */
public final class EpsilonPointProjection extends AbstractBlock {

  static final int FEATURE_WIDTH = 26;
  static final int GATE_FEATURE_WIDTH = 6;

  private static final int FU_COUNT = 12;
  private static final int HAN_COUNT = 14;
  private static final int YAKUMAN_COUNT = 16;
  private static final int TABLE_SIZE = FU_COUNT * HAN_COUNT * YAKUMAN_COUNT;
  private static final int SCORE_NORMALIZER_100 = 1000;
  private static final int[] WAIT_RON = {1, 1, 1, 0, 1, 1, 1, 0};
  private static final int[] WAIT_SOURCE = {1, 2, 3, 0, 1, 2, 3, 0};
  private static final int[] WAIT_AKA = {0, 0, 0, 0, 1, 1, 1, 1};

  private final EpsilonUtilityProfile utilityProfile;
  private NDArray ronChild;
  private NDArray ronDealer;
  private NDArray tsumoSingle;
  private NDArray tsumoDouble;
  private NDArray rankUtility;
  private NDArray waitRon;
  private NDArray waitSource;
  private NDArray waitAka;
  private NDArray absoluteSeats;

  public EpsilonPointProjection(EpsilonUtilityProfile utilityProfile) {
    this.utilityProfile = utilityProfile;
  }

  Projection projectActions(
      NDArray pointLedger100,
      NDArray stateCategories,
      NDArray actionCategories,
      NDArray actionWinFacts) {
    return projectActions(pointLedger100, stateCategories, actionCategories, actionWinFacts, null);
  }

  Projection projectActions(
      NDArray pointLedger100,
      NDArray stateCategories,
      NDArray actionCategories,
      NDArray actionWinFacts,
      NDArray suppliedIndices) {
    NDManager manager = actionWinFacts.getManager();
    try (NDManager scope = manager.newSubManager()) {
      scope.tempAttachAll(pointLedger100, stateCategories, actionCategories, actionWinFacts);
      long rows = actionWinFacts.getShape().get(0);
      long actions = actionWinFacts.getShape().get(1);
      NDArray indices = suppliedIndices;
      if (indices == null) {
        NDArray types =
            actionCategories.get("...,{}", DecisionInputSchema.ActionInt.TYPE.ordinal());
        NDArray valid =
            actionWinFacts
                .get("...,{}", DecisionInputSchema.WinFact.VALID.ordinal())
                .eq(1)
                .logicalAnd(
                    actionWinFacts
                        .get(
                            "...,{}", DecisionInputSchema.WinFact.REQUIRES_PAO_CORRECTION.ordinal())
                        .eq(0))
                .logicalAnd(
                    types
                        .eq(DecisionFeatureCodec.actionType(Action.Type.RON_AGARI))
                        .logicalOr(
                            types.eq(DecisionFeatureCodec.actionType(Action.Type.TSUMO_AGARI))));
        indices = EpsilonMaskedRows.indices(valid);
      }
      Projection result;
      if (indices.size() == 0) {
        result =
            new Projection(
                scope.zeros(new Shape(rows, actions, FEATURE_WIDTH)),
                scope.zeros(new Shape(rows, actions, GATE_FEATURE_WIDTH)),
                scope.zeros(new Shape(rows, actions)));
      } else {
        NDArray rowIndices =
            indices.floorDivide(actions).toType(DataType.INT64, false);
        Projection compact =
            projectActionsDense(
                EpsilonMaskedRows.gather(pointLedger100, rowIndices),
                EpsilonMaskedRows.gather(stateCategories, rowIndices),
                EpsilonMaskedRows.gather(actionCategories.reshape(rows * actions, -1), indices)
                    .expandDims(1),
                EpsilonMaskedRows.gather(actionWinFacts.reshape(rows * actions, -1), indices)
                    .expandDims(1));
        result =
            new Projection(
                EpsilonMaskedRows.scatter(
                        compact.features().reshape(-1, FEATURE_WIDTH), indices, rows * actions)
                    .reshape(rows, actions, FEATURE_WIDTH),
                EpsilonMaskedRows.scatter(
                        compact.gateFeatures().reshape(-1, GATE_FEATURE_WIDTH),
                        indices,
                        rows * actions)
                    .reshape(rows, actions, GATE_FEATURE_WIDTH),
                EpsilonMaskedRows.scatter(
                        compact.validMask().reshape(-1, 1), indices, rows * actions)
                    .reshape(rows, actions));
      }
      manager.attachAll(result.features(), result.gateFeatures(), result.validMask());
      return result;
    }
  }

  Projection projectActionsDense(
      NDArray pointLedger100,
      NDArray stateCategories,
      NDArray actionCategories,
      NDArray actionWinFacts) {
    NDManager manager = actionWinFacts.getManager();
    try (NDManager scope = projectionScope(manager)) {
      long rows = actionWinFacts.getShape().get(0);
      long actions = actionWinFacts.getShape().get(1);
      NDArray actionTypes =
          actionCategories
              .get("...,{}", DecisionInputSchema.ActionInt.TYPE.ordinal())
              .toType(DataType.INT32, false);
      NDArray ron =
          actionTypes
              .eq(DecisionFeatureCodec.actionType(Action.Type.RON_AGARI))
              .toType(DataType.INT32, false);
      NDArray tsumo =
          actionTypes
              .eq(DecisionFeatureCodec.actionType(Action.Type.TSUMO_AGARI))
              .toType(DataType.INT32, false);
      NDArray source =
          round(stateCategories, DecisionInputSchema.RoundInt.SOURCE_PLAYER_RELATIVE_SEAT)
              .sub(1)
              .reshape(rows, 1);
      NDArray zeros = ron.mul(0);
      Projection result =
          project(
              pointLedger100.reshape(rows, 1, GameState.NUM_PLAYERS),
              actionWinFacts,
              round(stateCategories, DecisionInputSchema.RoundInt.PLAYER_SEAT)
                  .sub(1)
                  .reshape(rows, 1),
              round(stateCategories, DecisionInputSchema.RoundInt.DEALER_RELATIVE_SEAT)
                  .sub(1)
                  .reshape(rows, 1),
              round(stateCategories, DecisionInputSchema.RoundInt.KYOKU_INDEX)
                  .sub(1)
                  .reshape(rows, 1),
              round(stateCategories, DecisionInputSchema.RoundInt.HONBA).reshape(rows, 1),
              round(stateCategories, DecisionInputSchema.RoundInt.KYOTAKU).reshape(rows, 1),
              ron,
              tsumo,
              source.mul(ron),
              zeros,
              zeros,
              zeros);
      manager.attachAll(result.features(), result.gateFeatures(), result.validMask());
      return result;
    }
  }

  Projection projectWaits(
      NDArray pointLedger100,
      NDArray stateCategories,
      NDArray actionTypes,
      NDArray waitWinFacts,
      NDArray akaAvailable) {
    NDManager manager = waitWinFacts.getManager();
    try (NDManager scope = projectionScope(manager)) {
      long rows = waitWinFacts.getShape().get(0);
      long actions = waitWinFacts.getShape().get(1);
      NDArray ronFacts =
          waitWinFacts.get("...,{},:", DecisionInputSchema.WaitWinType.RON.ordinal());
      NDArray tsumoFacts =
          waitWinFacts.get("...,{},:", DecisionInputSchema.WaitWinType.TSUMO.ordinal());
      NDArray ronRed = redFacts(ronFacts, akaAvailable);
      NDArray tsumoRed = redFacts(tsumoFacts, akaAvailable);
      NDArray facts =
          NDArrays.stack(
              new NDList(
                  ronFacts, ronFacts, ronFacts, tsumoFacts, ronRed, ronRed, ronRed, tsumoRed),
              ronFacts.getShape().dimension() - 1);
      NDArray riichi =
          actionTypes
              .eq(DecisionFeatureCodec.actionType(Action.Type.RIICHI_DAHAI))
              .toType(DataType.INT32, false)
              .reshape(rows, actions, 1, 1, 1);
      NDArray aka = waitAka.reshape(1, 1, 1, 1, WAIT_AKA.length).mul(akaAvailable.expandDims(4));
      Projection result =
          project(
              pointLedger100.reshape(rows, 1, 1, 1, 1, GameState.NUM_PLAYERS),
              facts,
              round(stateCategories, DecisionInputSchema.RoundInt.PLAYER_SEAT)
                  .sub(1)
                  .reshape(rows, 1, 1, 1, 1),
              round(stateCategories, DecisionInputSchema.RoundInt.DEALER_RELATIVE_SEAT)
                  .sub(1)
                  .reshape(rows, 1, 1, 1, 1),
              round(stateCategories, DecisionInputSchema.RoundInt.KYOKU_INDEX)
                  .sub(1)
                  .reshape(rows, 1, 1, 1, 1),
              round(stateCategories, DecisionInputSchema.RoundInt.HONBA).reshape(rows, 1, 1, 1, 1),
              round(stateCategories, DecisionInputSchema.RoundInt.KYOTAKU)
                  .reshape(rows, 1, 1, 1, 1),
              waitRon.reshape(1, 1, 1, 1, WAIT_RON.length),
              waitRon.neg().add(1).reshape(1, 1, 1, 1, WAIT_RON.length),
              waitSource.reshape(1, 1, 1, 1, WAIT_SOURCE.length),
              riichi,
              waitRon.mul(0).add(1).reshape(1, 1, 1, 1, WAIT_RON.length),
              aka);
      manager.attachAll(result.features(), result.gateFeatures(), result.validMask());
      return result;
    }
  }

  /** 固定表から派生する一時配列をモデル寿命まで保持しない。 */
  private NDManager projectionScope(NDManager manager) {
    NDManager scope = manager.newSubManager();
    scope.tempAttachAll(
        ronChild,
        ronDealer,
        tsumoSingle,
        tsumoDouble,
        rankUtility,
        waitRon,
        waitSource,
        waitAka,
        absoluteSeats);
    return scope;
  }

  private Projection project(
      NDArray pointLedger100,
      NDArray facts,
      NDArray playerSeat,
      NDArray dealerRelativeSeat,
      NDArray kyoku,
      NDArray honba,
      NDArray kyotaku,
      NDArray ron,
      NDArray tsumo,
      NDArray sourceRelativeSeat,
      NDArray riichi,
      NDArray snapshot,
      NDArray aka) {
    NDArray stored = facts.toType(DataType.INT32, false).stopGradient();
    NDArray valid =
        stored
            .get("...,{}", DecisionInputSchema.WinFact.VALID.ordinal())
            .eq(1)
            .logicalAnd(
                stored
                    .get("...,{}", DecisionInputSchema.WinFact.REQUIRES_PAO_CORRECTION.ordinal())
                    .eq(0))
            .logicalAnd(ron.add(tsumo).gt(0))
            .toType(DataType.INT32, false);
    NDArray han = stored.get("...,{}", DecisionInputSchema.WinFact.HAN_WITHOUT_URA.ordinal());
    NDArray fuCode = stored.get("...,{}", DecisionInputSchema.WinFact.FU_CODE.ordinal());
    NDArray yakuman =
        stored.get("...,{}", DecisionInputSchema.WinFact.YAKUMAN_MULTIPLIER.ordinal());
    NDArray tableIndex =
        yakuman.mul(HAN_COUNT * FU_COUNT).add(han.minimum(13).mul(FU_COUNT)).add(fuCode);
    NDArray dealerWinner = dealerRelativeSeat.eq(0).toType(DataType.INT32, false);
    NDArray ronPayment =
        ronChild
            .take(tableIndex)
            .mul(dealerWinner.neg().add(1))
            .add(ronDealer.take(tableIndex).mul(dealerWinner))
            .add(honba.mul(3));
    NDArray singlePayment = tsumoSingle.take(tableIndex).add(honba);
    NDArray doublePayment = tsumoDouble.take(tableIndex).add(honba);
    NDArray tsumoReceived =
        doublePayment
            .mul(3)
            .mul(dealerWinner)
            .add(doublePayment.add(singlePayment.mul(2)).mul(dealerWinner.neg().add(1)));

    int prefixDimension = facts.getShape().dimension() - 1;
    NDArray winnerMask = seatOneHot(playerSeat, prefixDimension, DataType.INT32);
    NDArray sourceMask =
        seatOneHot(
            playerSeat.add(sourceRelativeSeat).mod(GameState.NUM_PLAYERS),
            prefixDimension,
            DataType.INT32);
    NDArray dealerMask =
        seatOneHot(
            playerSeat.add(dealerRelativeSeat).mod(GameState.NUM_PLAYERS),
            prefixDimension,
            DataType.INT32);
    NDArray ronDelta = winnerMask.sub(sourceMask).mul(ronPayment.expandDims(prefixDimension));
    NDArray doublePaymentBySeat = doublePayment.expandDims(prefixDimension);
    NDArray childWinnerPayment =
        dealerMask
            .mul(doublePaymentBySeat)
            .add(dealerMask.neg().add(1).mul(singlePayment.expandDims(prefixDimension)));
    NDArray payerPayment =
        doublePaymentBySeat
            .mul(dealerWinner.expandDims(prefixDimension))
            .add(childWinnerPayment.mul(dealerWinner.neg().add(1).expandDims(prefixDimension)));
    NDArray tsumoDelta =
        winnerMask
            .mul(tsumoReceived.expandDims(prefixDimension).add(payerPayment))
            .sub(payerPayment);
    NDArray activeVector = valid.expandDims(prefixDimension);
    NDArray settlement =
        ronDelta
            .mul(ron.expandDims(prefixDimension))
            .add(tsumoDelta.mul(tsumo.expandDims(prefixDimension)))
            .mul(activeVector);
    NDArray adjustedLedger =
        pointLedger100.sub(
            winnerMask
                .mul(riichi.expandDims(prefixDimension))
                .mul(HanchanProgression.RIICHI_COST / 100));
    NDArray terminationScores = adjustedLedger.add(settlement);
    NDArray finalScores =
        terminationScores.add(
            winnerMask.mul(
                kyotaku
                    .add(riichi)
                    .mul(HanchanProgression.RIICHI_COST / 100)
                    .expandDims(prefixDimension)));
    NDArray selfScore = finalScores.mul(winnerMask).sum(new int[] {prefixDimension});
    NDArray terminationSelfScore =
        terminationScores.mul(winnerMask).sum(new int[] {prefixDimension});
    NDArray terminationRank =
        rank(terminationScores, terminationSelfScore, playerSeat, prefixDimension);
    NDArray rank = rank(finalScores, selfScore, playerSeat, prefixDimension);
    NDArray rankOneHot = seatOneHot(rank, prefixDimension, DataType.FLOAT32);
    NDArray topScore = terminationScores.max(new int[] {prefixDimension}, false);
    NDArray busted = terminationScores.min(new int[] {prefixDimension}, false).lt(0);
    NDArray dealerContinues = dealerRelativeSeat.eq(0);
    NDArray topSelf = terminationRank.eq(0);
    NDArray southOrLater = kyoku.gte(HanchanProgression.SOUTH_END_KYOKU);
    NDArray finishAtSouthOrLater =
        dealerContinues
            .logicalAnd(terminationSelfScore.gte(HanchanProgression.TOP_THRESHOLD / 100))
            .logicalAnd(topSelf)
            .logicalOr(
                dealerContinues
                    .logicalNot()
                    .logicalAnd(topScore.gte(HanchanProgression.TOP_THRESHOLD / 100)));
    NDArray hanchanEnd =
        busted
            .logicalOr(kyoku.gte(HanchanProgression.WEST_END_KYOKU))
            .logicalOr(southOrLater.logicalAnd(finishAtSouthOrLater))
            .toType(DataType.FLOAT32, false)
            .mul(valid);
    NDArray active = valid.toType(DataType.FLOAT32, false);
    NDArray notEnd = active.sub(hanchanEnd);
    NDArray dealerRepeat = notEnd.mul(dealerContinues.toType(DataType.FLOAT32, false));
    NDArray westExtension =
        notEnd
            .mul(dealerContinues.logicalNot().toType(DataType.FLOAT32, false))
            .mul(southOrLater.toType(DataType.FLOAT32, false));
    NDArray nextRound = notEnd.sub(dealerRepeat).sub(westExtension);
    NDArray terminalUtility = rankUtility.take(rank).mul(hanchanEnd);
    NDArray broadcastZero = active.mul(0);
    NDArray sourceOneHot =
        seatOneHot(sourceRelativeSeat, prefixDimension, DataType.FLOAT32)
            .mul(ron.toType(DataType.FLOAT32, false).expandDims(prefixDimension))
            .add(broadcastZero.expandDims(prefixDimension));
    NDList scoreGaps = new NDList();
    for (int relativeSeat = 0; relativeSeat < GameState.NUM_PLAYERS; relativeSeat++) {
      scoreGaps.add(
          selfScore
              .sub(relativeScore(finalScores, playerSeat, relativeSeat))
              .toType(DataType.FLOAT32, false)
              .div(SCORE_NORMALIZER_100)
              .expandDims(prefixDimension));
    }

    NDList features = new NDList();
    features.add(activeVector);
    features.add(
        ron.toType(DataType.FLOAT32, false).add(broadcastZero).expandDims(prefixDimension));
    features.add(
        tsumo.toType(DataType.FLOAT32, false).add(broadcastZero).expandDims(prefixDimension));
    features.add(sourceOneHot);
    features.add(settlement.toType(DataType.FLOAT32, false).div(SCORE_NORMALIZER_100));
    features.add(NDArrays.concat(scoreGaps, prefixDimension));
    features.add(rankOneHot);
    features.add(hanchanEnd.expandDims(prefixDimension));
    features.add(dealerRepeat.expandDims(prefixDimension));
    features.add(nextRound.expandDims(prefixDimension));
    features.add(westExtension.expandDims(prefixDimension));
    features.add(terminalUtility.expandDims(prefixDimension));
    features.add(
        snapshot.toType(DataType.FLOAT32, false).add(broadcastZero).expandDims(prefixDimension));
    features.add(
        aka.toType(DataType.FLOAT32, false).add(broadcastZero).expandDims(prefixDimension));
    NDArray projected = NDArrays.concat(features, prefixDimension).mul(activeVector);
    NDArray gateFeatures =
        NDArrays.concat(
                new NDList(
                    hanchanEnd.expandDims(prefixDimension),
                    terminalUtility.expandDims(prefixDimension),
                    rankOneHot),
                prefixDimension)
            .mul(activeVector);
    return new Projection(projected, gateFeatures, active);
  }

  private static NDArray redFacts(NDArray facts, NDArray akaAvailable) {
    int axis = facts.getShape().dimension() - 1;
    NDList fields = new NDList();
    fields.add(
        facts
            .get("...,{}", DecisionInputSchema.WinFact.VALID.ordinal())
            .mul(akaAvailable)
            .expandDims(axis));
    fields.add(
        facts
            .get("...,{}", DecisionInputSchema.WinFact.HAN_WITHOUT_URA.ordinal())
            .add(1)
            .expandDims(axis));
    for (int field = DecisionInputSchema.WinFact.FU_CODE.ordinal();
        field < DecisionInputSchema.WinFact.values().length;
        field++) {
      fields.add(facts.get("...,{}", field).expandDims(axis));
    }
    return NDArrays.concat(fields, axis);
  }

  private static NDArray round(NDArray stateCategories, DecisionInputSchema.RoundInt field) {
    return stateCategories.get(":,{}", field.ordinal()).toType(DataType.INT32, false);
  }

  private static NDArray seatOneHot(NDArray seats, int axis, DataType dataType) {
    NDList masks = new NDList();
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      masks.add(seats.eq(seat).toType(dataType, false).expandDims(axis));
    }
    return NDArrays.concat(masks, axis);
  }

  private static NDArray relativeScore(NDArray scores, NDArray playerSeat, int relativeSeat) {
    NDArray result = scores.get("...,0").mul(0);
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      int seat = (player + relativeSeat) % GameState.NUM_PLAYERS;
      result =
          result.add(
              scores.get("...,{}", seat).mul(playerSeat.eq(player).toType(DataType.INT32, false)));
    }
    return result;
  }

  private NDArray rank(NDArray scores, NDArray selfScore, NDArray playerSeat, int axis) {
    NDArray self = selfScore.expandDims(axis);
    return scores
        .gt(self)
        .logicalOr(scores.eq(self).logicalAnd(absoluteSeats.lt(playerSeat.expandDims(axis))))
        .toType(DataType.INT32, false)
        .sum(new int[] {axis});
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    int[] childRon = new int[TABLE_SIZE];
    int[] dealerRon = new int[TABLE_SIZE];
    int[] single = new int[TABLE_SIZE];
    int[] doubled = new int[TABLE_SIZE];
    for (int yakuman = 0; yakuman < YAKUMAN_COUNT; yakuman++) {
      for (int han = 0; han < HAN_COUNT; han++) {
        for (int fuCode = 0; fuCode < FU_COUNT; fuCode++) {
          int index = yakuman * HAN_COUNT * FU_COUNT + han * FU_COUNT + fuCode;
          int fu = ScoreMath.decodeFuCode(fuCode);
          int basePoints = ScoreMath.basePoints(han, fu, yakuman);
          childRon[index] = ScorePayments.ronPoints(basePoints, false) / 100;
          dealerRon[index] = ScorePayments.ronPoints(basePoints, true) / 100;
          single[index] = ScorePayments.tsumoFromChild(basePoints, false) / 100;
          doubled[index] = ScorePayments.tsumoFromDealer(basePoints) / 100;
        }
      }
    }
    ronChild = manager.create(childRon);
    ronDealer = manager.create(dealerRon);
    tsumoSingle = manager.create(single);
    tsumoDouble = manager.create(doubled);
    rankUtility =
        utilityProfile == EpsilonUtilityProfile.TENHOU
            ? manager.create(new float[] {0.0f, 0.0f, 0.0f, -1.0f})
            : manager.create(
                new float[] {
                  utilityProfile.utilityForRank(0),
                  utilityProfile.utilityForRank(1),
                  utilityProfile.utilityForRank(2),
                  utilityProfile.utilityForRank(3)
                });
    waitRon = manager.create(WAIT_RON);
    waitSource = manager.create(WAIT_SOURCE);
    waitAka = manager.create(WAIT_AKA);
    absoluteSeats = manager.arange(GameState.NUM_PLAYERS).toType(DataType.INT32, false);
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    throw new UnsupportedOperationException("Use projectActions or projectWaits");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, FEATURE_WIDTH)};
  }

  record Projection(NDArray features, NDArray gateFeatures, NDArray validMask) {}
}

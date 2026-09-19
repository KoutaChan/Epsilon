package com.epsilon.major.ai.decision.policy;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.core.Action;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.ScorePayments;
import com.epsilon.major.ai.decision.input.DecisionFeatureCodec;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import org.testng.Assert;
import org.testng.annotations.Test;

public final class EpsilonPointProjectionTest {

  private static final int DELTA_OFFSET = 7;
  private static final int RANK_OFFSET = 15;
  private static final int HANCHAN_END = 19;
  private static final int WEST_EXTENSION = 22;
  private static final int TERMINAL_UTILITY = 23;
  private static final int AKA = 25;

  @Test(groups = "native")
  public void scoreTableMatchesScoreMathForChildDealerRonTsumoHonbaAndHanClamp() {
    int rows = 20 * 12 * 2 * 2;
    int[][] ledgers = new int[rows][4];
    int[][] states = new int[rows][];
    int[][] facts = new int[rows][];
    Action.Type[] types = new Action.Type[rows];
    int[] expectedBase = new int[rows];
    boolean[] expectedDealer = new boolean[rows];
    int row = 0;
    for (int han = 1; han <= 20; han++) {
      for (int fuCode = 0; fuCode < 12; fuCode++) {
        for (int dealer = 0; dealer < 2; dealer++) {
          for (int method = 0; method < 2; method++) {
            ledgers[row] = new int[] {250, 250, 250, 250};
            states[row] = state(0, 1, 0, dealer == 1 ? 0 : 1, 2, 0);
            facts[row] = fact(han, fuCode, 0);
            types[row] = method == 0 ? Action.Type.RON_AGARI : Action.Type.TSUMO_AGARI;
            expectedBase[row] = ScoreMath.normalBasePoints(han, ScoreMath.decodeFuCode(fuCode));
            expectedDealer[row] = dealer == 1;
            row++;
          }
        }
      }
    }

    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonPointProjection.Projection projection =
          projectActions(manager, EpsilonUtilityProfile.TENHOU, ledgers, states, types, facts);
      for (row = 0; row < rows; row++) {
        int base = expectedBase[row];
        boolean dealer = expectedDealer[row];
        if (types[row] == Action.Type.RON_AGARI) {
          int payment = roundUp100(base * (dealer ? 6 : 4)) / 100 + 6;
          assertDelta(projection.features(), row, payment, -payment, 0, 0);
        } else if (dealer) {
          int payment = roundUp100(base * 2) / 100 + 2;
          assertDelta(projection.features(), row, payment * 3, -payment, -payment, -payment);
        } else {
          int childPayment = roundUp100(base) / 100 + 2;
          int dealerPayment = roundUp100(base * 2) / 100 + 2;
          assertDelta(
              projection.features(),
              row,
              dealerPayment + childPayment * 2,
              -dealerPayment,
              -childPayment,
              -childPayment);
        }
      }

      int han13 = rowOf(13, ScoreMath.encodeFuCode(30), false, false);
      int han20 = rowOf(20, ScoreMath.encodeFuCode(30), false, false);
      for (int seat = 0; seat < 4; seat++) {
        Assert.assertEquals(
            delta100(projection.features(), han20, seat),
            delta100(projection.features(), han13, seat),
            "han must clamp at counted-yakuman");
      }
    }
  }

  @Test(groups = "native")
  public void yakumanSettlementMatchesCommonPaymentsAcrossSeatRotations() {
    int rows = 4 * 4 * 3 * 2;
    int[][] ledgers = new int[rows][4];
    int[][] states = new int[rows][];
    int[][] facts = new int[rows][];
    Action.Type[] types = new Action.Type[rows];
    int row = 0;
    for (int player = 0; player < 4; player++) {
      for (int dealerRelative = 0; dealerRelative < 4; dealerRelative++) {
        for (int sourceRelative = 1; sourceRelative < 4; sourceRelative++) {
          for (int method = 0; method < 2; method++) {
            ledgers[row] = new int[] {301, 249, 200, 250};
            states[row] = state(player, sourceRelative, 0, dealerRelative, 3, 2);
            facts[row] = new int[] {1, 0, 0, 2, 0};
            types[row] = method == 0 ? Action.Type.RON_AGARI : Action.Type.TSUMO_AGARI;
            row++;
          }
        }
      }
    }

    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonPointProjection.Projection projection =
          projectActions(manager, EpsilonUtilityProfile.TENHOU, ledgers, states, types, facts);
      row = 0;
      int basePoints = 16_000;
      for (int player = 0; player < 4; player++) {
        for (int dealerRelative = 0; dealerRelative < 4; dealerRelative++) {
          int dealer = (player + dealerRelative) % 4;
          boolean dealerWinner = dealerRelative == 0;
          for (int sourceRelative = 1; sourceRelative < 4; sourceRelative++) {
            int source = (player + sourceRelative) % 4;
            for (int method = 0; method < 2; method++) {
              int[] expected = new int[4];
              if (method == 0) {
                int payment = ScorePayments.ronPoints(basePoints, dealerWinner) / 100 + 9;
                expected[player] = payment;
                expected[source] = -payment;
              } else {
                for (int seat = 0; seat < 4; seat++) {
                  if (seat == player) continue;
                  int payment =
                      (seat == dealer
                                  ? ScorePayments.tsumoFromDealer(basePoints)
                                  : ScorePayments.tsumoFromChild(basePoints, dealerWinner))
                              / 100
                          + 3;
                  expected[player] += payment;
                  expected[seat] = -payment;
                }
              }
              assertDelta(
                  projection.features(),
                  row,
                  expected[0],
                  expected[1],
                  expected[2],
                  expected[3]);
              row++;
            }
          }
        }
      }
    }
  }

  @Test(groups = "native")
  public void terminationUsesPreKyotakuScoresAtSouthFourBoundary() {
    int[][] ledgers = {{289, 280, 220, 210}, {290, 280, 220, 210}};
    int[][] states = {state(0, 1, 7, 1, 0, 1), state(0, 1, 7, 1, 0, 1)};
    int[][] facts = {
      fact(1, ScoreMath.encodeFuCode(30), 0), fact(1, ScoreMath.encodeFuCode(30), 0)
    };
    Action.Type[] types = {Action.Type.RON_AGARI, Action.Type.RON_AGARI};

    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonPointProjection.Projection projection =
          projectActions(manager, EpsilonUtilityProfile.TENHOU, ledgers, states, types, facts);
      Assert.assertEquals(feature(projection.features(), 0, HANCHAN_END), 0f);
      Assert.assertEquals(feature(projection.features(), 0, WEST_EXTENSION), 1f);
      Assert.assertEquals(delta100(projection.features(), 0, 0), 10);
      Assert.assertFalse(
          HanchanProgression.shouldFinishAfterSingleWin(7, 1, 0, 29900, 27000, 22000, 21000));

      Assert.assertEquals(feature(projection.features(), 1, HANCHAN_END), 1f);
      Assert.assertTrue(
          HanchanProgression.shouldFinishAfterSingleWin(7, 1, 0, 30000, 27000, 22000, 21000));
    }
  }

  @Test(groups = "native")
  public void riichiCostPrecedesTerminationAndNewStickReturnsAfterward() {
    int[][] ledgers = {{290, 280, 220, 210}, {290, 280, 220, 210}};
    int[][] states = {state(0, 1, 7, 1, 0, 0), state(0, 1, 7, 1, 0, 0)};
    int[] actionTypes = {
      DecisionFeatureCodec.actionType(Action.Type.DAHAI),
      DecisionFeatureCodec.actionType(Action.Type.RIICHI_DAHAI)
    };
    int[] ronFact = fact(1, ScoreMath.encodeFuCode(30), 0);
    int[] tsumoFact = fact(1, ScoreMath.encodeFuCode(30), 0);

    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonPointProjection.Projection projection =
          projectWaits(manager, ledgers, states, actionTypes, ronFact, tsumoFact, new int[] {0, 0});
      Assert.assertEquals(waitFeature(projection.features(), 0, 0, HANCHAN_END), 1f);
      Assert.assertEquals(waitFeature(projection.features(), 1, 0, HANCHAN_END), 0f);
      Assert.assertEquals(waitFeature(projection.features(), 1, 0, WEST_EXTENSION), 1f);
      Assert.assertEquals(waitDelta100(projection.features(), 0, 0, 0), 10);
      Assert.assertEquals(waitDelta100(projection.features(), 1, 0, 0), 10);
    }
  }

  @Test(groups = "native")
  public void tenhouUtilityIsNegativeOnlyForTerminalFourthPlace() {
    int[][] ledgers = {{50, 350, 300, 300}, {300, 250, 250, 200}};
    int[][] states = {state(0, 1, 7, 1, 0, 0), state(0, 1, 7, 1, 0, 0)};
    int[][] facts = {
      fact(1, ScoreMath.encodeFuCode(30), 0), fact(1, ScoreMath.encodeFuCode(30), 0)
    };
    Action.Type[] types = {Action.Type.RON_AGARI, Action.Type.RON_AGARI};

    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonPointProjection.Projection projection =
          projectActions(manager, EpsilonUtilityProfile.TENHOU, ledgers, states, types, facts);
      Assert.assertEquals(feature(projection.features(), 0, HANCHAN_END), 1f);
      Assert.assertEquals(feature(projection.features(), 0, TERMINAL_UTILITY), -1f);
      Assert.assertEquals(feature(projection.features(), 0, RANK_OFFSET + 3), 1f);
      Assert.assertEquals(projection.gateFeatures().getFloat(0, 0, 1), -1f);

      Assert.assertEquals(feature(projection.features(), 1, HANCHAN_END), 1f);
      Assert.assertEquals(feature(projection.features(), 1, TERMINAL_UTILITY), 0f);
      Assert.assertEquals(feature(projection.features(), 1, RANK_OFFSET), 1f);
      Assert.assertEquals(projection.gateFeatures().getFloat(1, 0, 1), 0f);
    }
  }

  @Test(groups = "native")
  public void paoDisablesPointAndGateContextWithoutProjectingAnApproximation() {
    int[][] ledgers = {{250, 250, 250, 250}};
    int[][] states = {state(0, 1, 7, 1, 0, 2)};
    int[][] facts = {{1, 13, 0, 1, 1}};
    Action.Type[] types = {Action.Type.RON_AGARI};

    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonPointProjection.Projection projection =
          projectActions(manager, EpsilonUtilityProfile.TENHOU, ledgers, states, types, facts);
      Assert.assertEquals(projection.features().abs().sum().getFloat(), 0f);
      Assert.assertEquals(projection.gateFeatures().abs().sum().getFloat(), 0f);
      Assert.assertEquals(projection.validMask().sum().getFloat(), 0f);
    }
  }

  @Test(groups = "native")
  public void akaWaitProducesNormalAndRedVariantsForAllFourWinCases() {
    int[][] ledgers = {{250, 250, 250, 250}};
    int[][] states = {state(0, 1, 0, 1, 0, 0)};
    int[] actionTypes = {DecisionFeatureCodec.actionType(Action.Type.DAHAI)};
    int[] ronFact = fact(2, ScoreMath.encodeFuCode(30), 0);
    int[] tsumoFact = fact(2, ScoreMath.encodeFuCode(30), 0);

    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonPointProjection.Projection projection =
          projectWaits(manager, ledgers, states, actionTypes, ronFact, tsumoFact, new int[] {1});
      Assert.assertEquals(projection.features().getShape().get(4), 8);
      Assert.assertEquals(projection.validMask().sum().getFloat(), 8f);
      for (int scenario = 0; scenario < 4; scenario++) {
        Assert.assertEquals(waitFeature(projection.features(), 0, scenario, AKA), 0f);
        Assert.assertEquals(waitFeature(projection.features(), 0, scenario + 4, AKA), 1f);
      }
      Assert.assertEquals(waitDelta100(projection.features(), 0, 0, 0), 20);
      Assert.assertEquals(waitDelta100(projection.features(), 0, 4, 0), 39);
      Assert.assertEquals(waitDelta100(projection.features(), 0, 3, 0), 20);
      Assert.assertEquals(waitDelta100(projection.features(), 0, 7, 0), 40);
    }
  }

  private static EpsilonPointProjection.Projection projectActions(
      NDManager manager,
      EpsilonUtilityProfile utilityProfile,
      int[][] ledgers,
      int[][] states,
      Action.Type[] types,
      int[][] facts) {
    int rows = ledgers.length;
    int actionWidth = DecisionInputSchema.ActionInt.values().length;
    int[] actionCategories = new int[rows * actionWidth];
    for (int row = 0; row < rows; row++) {
      actionCategories[row * actionWidth + DecisionInputSchema.ActionInt.TYPE.ordinal()] =
          DecisionFeatureCodec.actionType(types[row]);
    }
    EpsilonPointProjection projection = new EpsilonPointProjection(utilityProfile);
    projection.initialize(manager, DataType.FLOAT32);
    return projection.projectActions(
        manager.create(ledgers),
        manager.create(states),
        manager.create(actionCategories).reshape(rows, 1, actionWidth),
        manager.create(facts).reshape(rows, 1, DecisionInputSchema.WinFact.values().length));
  }

  private static EpsilonPointProjection.Projection projectWaits(
      NDManager manager,
      int[][] ledgers,
      int[][] states,
      int[] actionTypes,
      int[] ronFact,
      int[] tsumoFact,
      int[] akaAvailable) {
    int rows = ledgers.length;
    int factWidth = DecisionInputSchema.WinFact.values().length;
    int[] waitFacts = new int[rows * 2 * factWidth];
    for (int row = 0; row < rows; row++) {
      int offset = row * 2 * factWidth;
      System.arraycopy(ronFact, 0, waitFacts, offset, factWidth);
      System.arraycopy(tsumoFact, 0, waitFacts, offset + factWidth, factWidth);
    }
    EpsilonPointProjection projection = new EpsilonPointProjection(EpsilonUtilityProfile.TENHOU);
    projection.initialize(manager, DataType.FLOAT32);
    return projection.projectWaits(
        manager.create(ledgers),
        manager.create(states),
        manager.create(actionTypes).reshape(rows, 1),
        manager.create(waitFacts).reshape(rows, 1, 1, 1, 2, factWidth),
        manager.create(akaAvailable).reshape(rows, 1, 1, 1));
  }

  private static int[] state(
      int playerSeat,
      int sourceRelativeSeat,
      int kyoku,
      int dealerRelativeSeat,
      int honba,
      int kyotaku) {
    int[] state = new int[DecisionInputSchema.RoundInt.values().length];
    state[DecisionInputSchema.RoundInt.PLAYER_SEAT.ordinal()] = playerSeat + 1;
    state[DecisionInputSchema.RoundInt.SOURCE_PLAYER_RELATIVE_SEAT.ordinal()] =
        sourceRelativeSeat + 1;
    state[DecisionInputSchema.RoundInt.KYOKU_INDEX.ordinal()] = kyoku + 1;
    state[DecisionInputSchema.RoundInt.DEALER_RELATIVE_SEAT.ordinal()] = dealerRelativeSeat + 1;
    state[DecisionInputSchema.RoundInt.HONBA.ordinal()] = honba;
    state[DecisionInputSchema.RoundInt.KYOTAKU.ordinal()] = kyotaku;
    return state;
  }

  private static int[] fact(int han, int fuCode, int pao) {
    return new int[] {1, han, fuCode, 0, pao};
  }

  private static void assertDelta(
      NDArray features, int row, int seat0, int seat1, int seat2, int seat3) {
    int[] expected = {seat0, seat1, seat2, seat3};
    for (int seat = 0; seat < 4; seat++) {
      Assert.assertEquals(delta100(features, row, seat), expected[seat], "row=" + row);
    }
  }

  private static int delta100(NDArray features, int row, int seat) {
    return Math.round(feature(features, row, DELTA_OFFSET + seat) * 1000);
  }

  private static float feature(NDArray features, int row, int field) {
    return features.getFloat(row, 0, field);
  }

  private static int waitDelta100(NDArray features, int row, int scenario, int seat) {
    return Math.round(waitFeature(features, row, scenario, DELTA_OFFSET + seat) * 1000);
  }

  private static float waitFeature(NDArray features, int row, int scenario, int field) {
    return features.getFloat(row, 0, 0, 0, scenario, field);
  }

  private static int roundUp100(int points) {
    return ((points + 99) / 100) * 100;
  }

  private static int rowOf(int han, int fuCode, boolean dealer, boolean tsumo) {
    return (((han - 1) * 12 + fuCode) * 2 + (dealer ? 1 : 0)) * 2 + (tsumo ? 1 : 0);
  }
}

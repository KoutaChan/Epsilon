package com.epsilon.grp;

import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 次局の結果の表の大きさと席の立場、OutcomeNetwork の確率の和と決定性、局の結果（RoundOutcome）への置き方を確かめる。 */
public class OutcomeNetworkTest {

  private static final Variant FOUR = Variant.TENHOU_4P;
  private static final int NONE = HandOutcomes.NONE;

  @Test
  public void theFourPlayerTableHasTheDeclaredSizes() {
    HandOutcomes hands = HandOutcomes.of(FOUR);
    Assert.assertEquals(hands.mainCount(), 45);
    Assert.assertEquals(hands.logits(), 277);
    Assert.assertEquals(hands.wins(), 16);
    Assert.assertEquals(hands.terms(), 45 + 16 * HandOutcomes.BASES + 2 * 45 * 4);
    Assert.assertEquals(hands.termStart(OutcomeTerm.GRADE), 45);
    Assert.assertEquals(hands.termStart(OutcomeTerm.RIICHI), 45 + 16 * HandOutcomes.BASES);
    OutcomeShape linked =
        new OutcomeShape(FOUR, GrpFixtures.plainFields(), 4, EnumSet.allOf(OutcomeLink.class));
    Assert.assertEquals(linked.linkStart(OutcomeLink.RANK), 277);
    Assert.assertEquals(linked.linkStart(OutcomeLink.WINNER), 277 + 4 * 2 * 4);
    Assert.assertEquals(linked.logits(), 277 + 4 * 2 * 4 + 4 * RiichiRole.values().length * 4);
    OutcomeShape winnerOnly =
        new OutcomeShape(FOUR, GrpFixtures.plainFields(), 4, EnumSet.of(OutcomeLink.WINNER));
    Assert.assertEquals(winnerOnly.linkStart(OutcomeLink.RANK), -1);
    Assert.assertEquals(winnerOnly.linkStart(OutcomeLink.WINNER), 277);
    int[] counts = new int[OutcomeSection.values().length];
    for (int main = 0; main < hands.mainCount(); main++) {
      OutcomeSection section = hands.section(main);
      counts[section.ordinal()]++;
      int expected =
          switch (section) {
            case TSUMO, RON, DOUBLE_RON, ABORTIVE -> 4;
            case DRAW -> Integer.bitCount(hands.tenpai(main));
          };
      Assert.assertEquals(hands.factorCount(main), expected, "main " + main);
      if (section == OutcomeSection.DOUBLE_RON) {
        Assert.assertTrue(hands.winner(main) < hands.secondWinner(main));
        Assert.assertNotEquals(hands.loser(main), hands.winner(main));
        Assert.assertNotEquals(hands.loser(main), hands.secondWinner(main));
        Assert.assertEquals(hands.role(main, hands.secondWinner(main)), RiichiRole.WINNER);
        Assert.assertEquals(hands.role(main, hands.loser(main)), RiichiRole.LOSER);
      }
    }
    Assert.assertEquals(counts, new int[] {4, 12, 12, 16, 1});
    int draw = GrpFixtures.main(hands, OutcomeSection.DRAW, NONE, NONE, NONE, 0b0101);
    Assert.assertEquals(hands.role(draw, 0), RiichiRole.TENPAI);
    Assert.assertNull(hands.role(draw, 1));
    Assert.assertEquals(hands.factorSeat(draw, 1), 2);
  }

  @DataProvider
  public static Object[][] shapes() {
    List<Object[]> shapes = new ArrayList<>();
    for (Variant variant : FinalRound.VARIANTS) {
      shapes.add(new Object[] {new OutcomeShape(variant, GrpFixtures.plainFields(), 12)});
      for (OutcomeLink link : OutcomeLink.values()) {
        shapes.add(
            new Object[] {
              new OutcomeShape(variant, GrpFixtures.plainFields(), 12, EnumSet.of(link))
            });
      }
      shapes.add(
          new Object[] {
            new OutcomeShape(
                variant, GrpFixtures.plainFields(), 12, EnumSet.allOf(OutcomeLink.class))
          });
    }
    return shapes.toArray(Object[][]::new);
  }

  @Test(dataProvider = "shapes")
  public void probabilitiesFormDistributions(OutcomeShape shape) {
    OutcomeNetwork network = GrpFixtures.outcomes(shape, 3, 0.6);
    OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
    HandOutcomes hands = shape.outcomes();
    Random random = new Random(5);
    for (int trial = 0; trial < 200; trial++) {
      network.forward(GrpFixtures.state(shape.variant(), random), workspace);
      double main = 0;
      for (int index = 0; index < hands.mainCount(); index++) {
        main += workspace.probability(hands.mainTerm(index));
        Assert.assertEquals(
            workspace.term(hands.mainTerm(index)),
            Math.log(workspace.probability(hands.mainTerm(index))),
            1e-9);
      }
      Assert.assertEquals(main, 1, 1e-12);
      for (int win = 0; win < hands.wins(); win++) {
        double grades = 0;
        for (int base = 0; base < HandOutcomes.BASES; base++) {
          int term = hands.gradeTerm(win, base);
          grades += workspace.probability(term);
          Assert.assertEquals(
              workspace.term(term), Math.log(workspace.probability(term)), 1e-9, "term " + term);
        }
        Assert.assertEquals(grades, 1, 1e-12);
      }
      for (int index = 0; index < hands.mainCount(); index++) {
        for (int factor = 0; factor < hands.factorCount(index); factor++) {
          int declared = hands.riichiTerm(index, hands.factorSeat(index, factor), true);
          int quiet = hands.riichiTerm(index, hands.factorSeat(index, factor), false);
          Assert.assertEquals(
              workspace.probability(declared) + workspace.probability(quiet), 1, 1e-12);
          Assert.assertEquals(
              workspace.term(declared), Math.log(workspace.probability(declared)), 1e-9);
          Assert.assertEquals(workspace.term(quiet), Math.log(workspace.probability(quiet)), 1e-9);
        }
      }
    }
  }

  /**
   * 足すロジットが全て 0 のネットワークは、足すロジットを持たない形の同じ重みのネットワークと、ビット単位で同じ項を出す。
   */
  @Test(dataProvider = "shapes")
  public void zeroLinksLeaveThePlainNetwork(OutcomeShape shape) {
    OutcomeShape plain = new OutcomeShape(shape.variant(), shape.fields(), shape.hidden());
    OutcomeNetwork base = GrpFixtures.outcomes(plain, 9, 0.7);
    float[] source = base.weights();
    int[] from = OutcomeTensor.starts(plain);
    int[] to = OutcomeTensor.starts(shape);
    float[] weights = new float[to[OutcomeTensor.values().length]];
    for (OutcomeTensor tensor : OutcomeTensor.values()) {
      int columns = tensor.columns(plain);
      for (int row = 0; row < tensor.rows(plain); row++) {
        System.arraycopy(
            source,
            from[tensor.ordinal()] + row * columns,
            weights,
            to[tensor.ordinal()] + row * tensor.columns(shape),
            columns);
      }
    }
    OutcomeNetwork widened = OutcomeNetwork.of(shape, weights);
    OutcomeWorkspace expected = OutcomeWorkspace.of(plain);
    OutcomeWorkspace actual = OutcomeWorkspace.of(shape);
    Random random = new Random(21);
    for (int trial = 0; trial < 100; trial++) {
      GrpState state = GrpFixtures.state(shape.variant(), random);
      base.forward(state, expected);
      widened.forward(state, actual);
      for (int term = 0; term < plain.outcomes().terms(); term++) {
        Assert.assertEquals(
            Double.doubleToRawLongBits(actual.term(term)),
            Double.doubleToRawLongBits(expected.term(term)),
            "term " + term);
      }
    }
  }

  /**
   * 順位のロジットは、打点ごとに、その打点で和了者が就く順位の分だけ打点のロジットに足される。和了者ごとのロジットは、和了者で
   * ない席の立直のロジットに、その席の立場と和了者の分だけ足される。順位は、打点ごとに精算して求めた値と同じ。
   */
  @Test(dataProvider = "variants")
  public void linksShiftTheLogitsTheyName(Variant variant) {
    OutcomeShape shape =
        new OutcomeShape(variant, GrpFixtures.plainFields(), 6, EnumSet.allOf(OutcomeLink.class));
    HandOutcomes hands = shape.outcomes();
    int seats = variant.seats();
    Random random = new Random(33);
    double[] logits = new double[shape.logits()];
    for (int index = 0; index < logits.length; index++) {
      logits[index] = random.nextGaussian();
    }
    float[] rounded = new float[logits.length];
    for (int index = 0; index < logits.length; index++) {
      rounded[index] = (float) logits[index];
    }
    OutcomeNetwork network = GrpFixtures.biased(shape, logits);
    OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
    WinRanks ranks = new WinRanks(variant);
    int[] filled = new int[HandOutcomes.BASES];
    TableState table = new TableState(variant);
    for (int trial = 0; trial < 60; trial++) {
      GrpState state = GrpFixtures.state(variant, random);
      state.copyInto(table);
      network.forward(state, workspace);
      for (int win = 0; win < hands.wins(); win++) {
        WinMethod method =
            hands.section(win) == OutcomeSection.TSUMO ? WinMethod.TSUMO : WinMethod.RON;
        int winner = hands.winner(win);
        ranks.ranks(table, win, filled);
        double[] expected = new double[HandOutcomes.BASES];
        double sum = 0;
        for (int base = 0; base < HandOutcomes.BASES; base++) {
          int rank = ranks.rank(table, win, base);
          Assert.assertEquals(filled[base], rank, "win " + win + " base " + base + " of " + state);
          int link =
              shape.linkStart(OutcomeLink.RANK)
                  + (winner * WinMethod.values().length + method.ordinal()) * seats
                  + rank;
          expected[base] =
              Math.exp((double) rounded[hands.gradeLogit(winner, method, base)] + rounded[link]);
          sum += expected[base];
        }
        for (int base = 0; base < HandOutcomes.BASES; base++) {
          Assert.assertEquals(
              workspace.probability(hands.gradeTerm(win, base)), expected[base] / sum, 1e-12);
        }
      }
      for (int main = 0; main < hands.mainCount(); main++) {
        for (int factor = 0; factor < hands.factorCount(main); factor++) {
          int seat = hands.factorSeat(main, factor);
          RiichiRole role = hands.role(main, seat);
          double z = rounded[hands.riichiLogit(seat, role)];
          if (role != RiichiRole.WINNER) {
            int link =
                shape.linkStart(OutcomeLink.WINNER)
                    + (seat * RiichiRole.values().length + role.ordinal()) * seats;
            z += hands.winner(main) == NONE ? 0 : rounded[link + hands.winner(main)];
            z += hands.secondWinner(main) == NONE ? 0 : rounded[link + hands.secondWinner(main)];
          }
          Assert.assertEquals(
              workspace.probability(hands.riichiTerm(main, seat, true)),
              1 / (1 + Math.exp(-z)),
              1e-12);
        }
      }
    }
  }

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "shapes")
  public void forwardIsDeterministic(OutcomeShape shape) {
    OutcomeNetwork network = GrpFixtures.outcomes(shape, 7, 0.8);
    OutcomeWorkspace first = OutcomeWorkspace.of(shape);
    OutcomeWorkspace second = OutcomeWorkspace.of(shape);
    Random random = new Random(11);
    TableState table = new TableState(shape.variant());
    for (int trial = 0; trial < 50; trial++) {
      GrpState state = GrpFixtures.state(shape.variant(), random);
      network.forward(state, first);
      double[] expected = new double[shape.outcomes().terms()];
      for (int term = 0; term < expected.length; term++) {
        expected[term] = first.term(term);
      }
      table.set(state.round(), state.honba(), state.sticks(), state.scores());
      network.forward(table, second);
      for (int term = 0; term < expected.length; term++) {
        Assert.assertEquals(
            Double.doubleToRawLongBits(second.term(term)),
            Double.doubleToRawLongBits(expected[term]),
            "term " + term);
      }
    }
  }

  /** 相対席は次局の親から数える。親が絶対席 3 なら、相対席 1・2 は絶対席 0・1。 */
  @Test
  public void placeUsesAbsoluteSeats() {
    HandOutcomes hands = HandOutcomes.of(FOUR);
    RoundOutcome outcome = new RoundOutcome(FOUR);
    int ron = GrpFixtures.main(hands, OutcomeSection.RON, 1, NONE, 2, 0);
    hands.place(ron, BaseClass.MANGAN.ordinal(), NONE, 0b0001, 3, outcome);
    Assert.assertEquals(outcome.kind(), EndKind.RON);
    Assert.assertEquals(outcome.winner(0), 0);
    Assert.assertEquals(outcome.loser(), 1);
    Assert.assertEquals(outcome.base(0), 2_000);
    Assert.assertEquals(outcome.riichiMask(), 0b0001);
    int pair = GrpFixtures.main(hands, OutcomeSection.DOUBLE_RON, 0, 2, 1, 0);
    hands.place(pair, BaseClass.B240.ordinal(), BaseClass.YAKUMAN.ordinal(), 0, 3, outcome);
    Assert.assertEquals(outcome.winnerCount(), 2);
    Assert.assertEquals(outcome.loser(), 0);
    Assert.assertEquals(outcome.winner(0), 1, "the winner nearer the loser comes first");
    Assert.assertEquals(outcome.base(0), 8_000);
    Assert.assertEquals(outcome.winner(1), 3);
    int draw = GrpFixtures.main(hands, OutcomeSection.DRAW, NONE, NONE, NONE, 0b0011);
    hands.place(draw, NONE, NONE, 0b1000, 3, outcome);
    Assert.assertEquals(outcome.kind(), EndKind.EXHAUSTIVE);
    Assert.assertEquals(outcome.tenpaiMask(), 0b1001);
    Assert.assertEquals(outcome.riichiMask(), 0b1000);
    hands.place(hands.mainCount() - 1, NONE, NONE, 0, 3, outcome);
    Assert.assertEquals(outcome.kind(), EndKind.ABORTIVE);
  }

  @Test
  public void shapesFollowTheEnumeratedVariants() {
    Assert.assertEquals(
        new OutcomeShape(Variant.TENHOU_3P, GrpFixtures.plainFields(), 8).outcomes(),
        HandOutcomes.of(Variant.TENHOU_3P));
    OutcomeShape shape = new OutcomeShape(FOUR, GrpFixtures.plainFields(), 8);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> OutcomeNetwork.of(shape, new float[3]));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> new OutcomeShape(FOUR, GrpFixtures.plainFields(), 0));
  }
}

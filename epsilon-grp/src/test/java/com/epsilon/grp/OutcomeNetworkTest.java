package com.epsilon.grp;

import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Random;
import org.testng.Assert;
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
    Assert.assertEquals(hands.terms(), 301);
    Assert.assertEquals(hands.termStart(OutcomeTerm.GRADE), 45);
    Assert.assertEquals(hands.termStart(OutcomeTerm.RIICHI), 45 + 208);
    int[] counts = new int[OutcomeSection.values().length];
    for (int main = 0; main < hands.mainCount(); main++) {
      OutcomeSection section = hands.section(main);
      counts[section.ordinal()]++;
      int expected = switch (section) {
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

  @Test
  public void probabilitiesFormDistributions() {
    OutcomeShape shape = new OutcomeShape(FOUR, GrpFixtures.plainFields(), 12);
    OutcomeNetwork network = GrpFixtures.outcomes(shape, 3, 0.6);
    OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
    HandOutcomes hands = shape.outcomes();
    Random random = new Random(5);
    for (int trial = 0; trial < 200; trial++) {
      network.forward(GrpFixtures.state(FOUR, random), workspace);
      double main = 0;
      for (int index = 0; index < hands.mainCount(); index++) {
        main += workspace.probability(hands.mainTerm(index));
      }
      Assert.assertEquals(main, 1, 1e-12);
      for (int seat = 0; seat < 4; seat++) {
        for (WinMethod method : WinMethod.values()) {
          double grades = 0;
          for (int base = 0; base < HandOutcomes.BASES; base++) {
            grades += workspace.probability(hands.gradeTerm(seat, method, base));
          }
          Assert.assertEquals(grades, 1, 1e-12);
        }
        for (RiichiRole role : RiichiRole.values()) {
          double declared = workspace.probability(hands.riichiTerm(seat, role, true));
          double quiet = workspace.probability(hands.riichiTerm(seat, role, false));
          Assert.assertEquals(declared + quiet, 1, 1e-12);
        }
      }
      for (int term = 0; term < hands.terms(); term++) {
        Assert.assertEquals(workspace.term(term), Math.log(workspace.probability(term)), 1e-9, "term " + term);
      }
    }
  }

  @Test
  public void forwardIsDeterministic() {
    OutcomeShape shape = new OutcomeShape(FOUR, GrpFixtures.plainFields(), 16);
    OutcomeNetwork network = GrpFixtures.outcomes(shape, 7, 0.8);
    OutcomeWorkspace first = OutcomeWorkspace.of(shape);
    OutcomeWorkspace second = OutcomeWorkspace.of(shape);
    Random random = new Random(11);
    TableState table = new TableState(FOUR);
    for (int trial = 0; trial < 50; trial++) {
      GrpState state = GrpFixtures.state(FOUR, random);
      network.forward(state, first);
      double[] expected = new double[shape.outcomes().terms()];
      for (int term = 0; term < expected.length; term++) {
        expected[term] = first.term(term);
      }
      table.set(state.round(), state.honba(), state.sticks(), state.scores());
      network.forward(table, second);
      for (int term = 0; term < expected.length; term++) {
        Assert.assertEquals(
            Double.doubleToRawLongBits(second.term(term)), Double.doubleToRawLongBits(expected[term]), "term " + term);
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
    Assert.assertThrows(IllegalArgumentException.class, () -> OutcomeNetwork.of(shape, new float[3]));
    Assert.assertThrows(IllegalArgumentException.class, () -> new OutcomeShape(FOUR, GrpFixtures.plainFields(), 0));
  }
}

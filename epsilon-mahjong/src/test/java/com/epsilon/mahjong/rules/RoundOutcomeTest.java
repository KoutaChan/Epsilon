package com.epsilon.mahjong.rules;

import static com.epsilon.mahjong.rules.RoundOutcome.NO_SEAT;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 局の結果のフィールド、ロンの和了者の並び、値の検査、同じオブジェクトの使い回しを確かめる。 */
public class RoundOutcomeTest {

  @Test
  public void startsAsADrawWithoutTenpai() {
    for (Variant variant : Variant.values()) {
      RoundOutcome outcome = new RoundOutcome(variant);
      Assert.assertSame(outcome.variant(), variant);
      Assert.assertEquals(outcome.kind(), EndKind.EXHAUSTIVE);
      Assert.assertNull(outcome.abortive());
      Assert.assertEquals(outcome.winnerCount(), 0);
      Assert.assertEquals(outcome.loser(), NO_SEAT);
      Assert.assertEquals(outcome.riichiMask(), 0);
      Assert.assertEquals(outcome.tenpaiMask(), 0);
      Assert.assertEquals(outcome.nagashiMask(), 0);
    }
  }

  @Test
  public void onlyTsumoAndRonAreWins() {
    Assert.assertTrue(EndKind.TSUMO.win());
    Assert.assertTrue(EndKind.RON.win());
    Assert.assertFalse(EndKind.EXHAUSTIVE.win());
    Assert.assertFalse(EndKind.ABORTIVE.win());
  }

  @Test
  public void tsumoHasOneWinnerAndNoLoser() {
    RoundOutcome outcome = new RoundOutcome(Variant.TENHOU_3P);
    outcome.setTsumo(8_000, 2, 1, 0b011);
    Assert.assertEquals(outcome.kind(), EndKind.TSUMO);
    Assert.assertEquals(outcome.winnerCount(), 1);
    Assert.assertEquals(outcome.winner(0), 2);
    Assert.assertEquals(outcome.base(0), 8_000);
    Assert.assertEquals(outcome.pao(0), 1);
    Assert.assertEquals(outcome.loser(), NO_SEAT);
    Assert.assertEquals(outcome.riichiMask(), 0b011);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> outcome.winner(1));
  }

  /** 和了者は、足した順に依らず、放銃者の次の席から手番の順に並ぶ。 */
  @Test
  public void ronWinnersRunFromTheDiscarderInTurnOrder() {
    RoundOutcome four = new RoundOutcome(Variant.TENHOU_4P);
    four.setRon(2_000, 1, 2, NO_SEAT, 0);
    four.addRonWinner(480, 0, NO_SEAT);
    Assert.assertEquals(four.kind(), EndKind.RON);
    Assert.assertEquals(four.loser(), 2);
    Assert.assertEquals(four.winnerCount(), 2);
    Assert.assertEquals(four.winner(0), 0);
    Assert.assertEquals(four.base(0), 480);
    Assert.assertEquals(four.winner(1), 1);
    Assert.assertEquals(four.base(1), 2_000);

    four.setRon(8_000, 3, 0, 2, 0);
    four.addRonWinner(960, 1, NO_SEAT);
    Assert.assertEquals(four.winner(0), 1);
    Assert.assertEquals(four.pao(0), NO_SEAT);
    Assert.assertEquals(four.winner(1), 3);
    Assert.assertEquals(four.pao(1), 2);

    RoundOutcome three = new RoundOutcome(Variant.TENHOU_3P);
    three.setRon(960, 0, 1, NO_SEAT, 0);
    three.addRonWinner(960, 2, NO_SEAT);
    Assert.assertEquals(three.winner(0), 2);
    Assert.assertEquals(three.winner(1), 0);
  }

  @Test
  public void ronRejectsTheDiscarderRepeatedWinnersAndTripleRon() {
    RoundOutcome four = new RoundOutcome(Variant.TENHOU_4P);
    Assert.assertThrows(IllegalArgumentException.class, () -> four.setRon(2_000, 1, 1, NO_SEAT, 0));
    four.setRon(2_000, 1, 0, NO_SEAT, 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> four.addRonWinner(2_000, 0, NO_SEAT));
    Assert.assertThrows(IllegalArgumentException.class, () -> four.addRonWinner(2_000, 1, NO_SEAT));
    four.addRonWinner(2_000, 2, NO_SEAT);
    Assert.assertThrows(IllegalArgumentException.class, () -> four.addRonWinner(2_000, 3, NO_SEAT));
    Assert.assertEquals(four.winnerCount(), 2);

    RoundOutcome three = new RoundOutcome(Variant.TENHOU_3P);
    three.setRon(2_000, 1, 0, NO_SEAT, 0);
    three.addRonWinner(2_000, 2, NO_SEAT);
    Assert.assertThrows(IllegalArgumentException.class, () -> three.addRonWinner(2_000, 0, NO_SEAT));
  }

  /** 包の席が放銃したロンは、包の無いロンとして書く。放銃者と包の席が同じロンは拒み、前の結果をそのまま残す。 */
  @Test
  public void theDiscarderCannotAlsoBeLiable() {
    RoundOutcome outcome = new RoundOutcome(Variant.TENHOU_4P);
    outcome.setRon(8_000, 1, 0, 3, 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> outcome.setRon(8_000, 2, 3, 3, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> outcome.addRonWinner(8_000, 2, 0));
    Assert.assertEquals(outcome.winnerCount(), 1);
    Assert.assertEquals(outcome.loser(), 0);
    Assert.assertEquals(outcome.pao(0), 3);
  }

  @Test
  public void onlyARonTakesMoreWinners() {
    RoundOutcome outcome = new RoundOutcome(Variant.TENHOU_4P);
    Assert.assertThrows(IllegalStateException.class, () -> outcome.addRonWinner(2_000, 1, NO_SEAT));
    outcome.setTsumo(2_000, 0, NO_SEAT, 0);
    Assert.assertThrows(IllegalStateException.class, () -> outcome.addRonWinner(2_000, 1, NO_SEAT));
  }

  @Test
  public void drawsCarryTheirMasks() {
    RoundOutcome outcome = new RoundOutcome(Variant.TENHOU_4P);
    outcome.setExhaustive(0b1010, 0b0100, 0b0010);
    Assert.assertEquals(outcome.kind(), EndKind.EXHAUSTIVE);
    Assert.assertEquals(outcome.tenpaiMask(), 0b1010);
    Assert.assertEquals(outcome.nagashiMask(), 0b0100);
    Assert.assertEquals(outcome.riichiMask(), 0b0010);
    outcome.setAbortive(AbortiveDraw.FOUR_RIICHI, 0b1111);
    Assert.assertEquals(outcome.kind(), EndKind.ABORTIVE);
    Assert.assertSame(outcome.abortive(), AbortiveDraw.FOUR_RIICHI);
    Assert.assertEquals(outcome.riichiMask(), 0b1111);
    Assert.assertEquals(outcome.tenpaiMask(), 0);
    Assert.assertEquals(outcome.nagashiMask(), 0);
  }

  @Test
  public void abortiveDrawsMustExistInTheVariant() {
    RoundOutcome three = new RoundOutcome(Variant.TENHOU_3P);
    for (AbortiveDraw draw : AbortiveDraw.values()) {
      if (Variant.TENHOU_3P.aborts(draw)) {
        three.setAbortive(draw, 0);
        Assert.assertSame(three.abortive(), draw);
      } else {
        Assert.assertThrows(IllegalArgumentException.class, () -> three.setAbortive(draw, 0));
      }
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setAbortive(null, 0));
  }

  @Test
  public void valuesOutsideTheVariantOrTheRulesAreRejected() {
    RoundOutcome three = new RoundOutcome(Variant.TENHOU_3P);
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setTsumo(1_000, 0, NO_SEAT, 0));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> three.setTsumo(2_000, 3, NO_SEAT, 0));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> three.setRon(2_000, 0, 3, NO_SEAT, 0));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> three.setTsumo(8_000, 0, 3, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setTsumo(8_000, 0, 0, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setTsumo(2_000, 0, 1, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setTsumo(2_000, 0, NO_SEAT, 0b1000));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setExhaustive(0b1000, 0, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setExhaustive(0, 0b1000, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setAbortive(AbortiveDraw.FOUR_KANS, -1));
  }

  @Test
  public void eachEndClearsTheFieldsOfThePreviousOne() {
    RoundOutcome outcome = new RoundOutcome(Variant.TENHOU_4P);
    outcome.setRon(8_000, 1, 0, 3, 0b0001);
    outcome.addRonWinner(2_000, 2, NO_SEAT);
    outcome.setTsumo(320, 3, NO_SEAT, 0);
    Assert.assertEquals(outcome.winnerCount(), 1);
    Assert.assertEquals(outcome.winner(0), 3);
    Assert.assertEquals(outcome.pao(0), NO_SEAT);
    Assert.assertEquals(outcome.loser(), NO_SEAT);
    Assert.assertEquals(outcome.riichiMask(), 0);
    outcome.setAbortive(AbortiveDraw.NINE_TERMINALS, 0b0100);
    Assert.assertEquals(outcome.winnerCount(), 0);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> outcome.winner(0));
    outcome.setExhaustive(0b0011, 0, 0);
    Assert.assertNull(outcome.abortive());
    outcome.setRon(2_000, 0, 1, NO_SEAT, 0);
    Assert.assertEquals(outcome.tenpaiMask(), 0);
    Assert.assertEquals(outcome.winnerCount(), 1);
  }

  @Test
  public void aRejectedSetterKeepsThePreviousOutcome() {
    RoundOutcome outcome = new RoundOutcome(Variant.TENHOU_4P);
    outcome.setRon(2_000, 1, 0, NO_SEAT, 0b0001);
    Assert.assertThrows(IllegalArgumentException.class, () -> outcome.setTsumo(2_000, 2, 2, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> outcome.addRonWinner(2_000, 1, NO_SEAT));
    Assert.assertEquals(outcome.kind(), EndKind.RON);
    Assert.assertEquals(outcome.winnerCount(), 1);
    Assert.assertEquals(outcome.winner(0), 1);
    Assert.assertEquals(outcome.loser(), 0);
    Assert.assertEquals(outcome.riichiMask(), 0b0001);
  }

  @Test
  public void aCopyHoldsTheSameOutcomeOfTheSameVariant() {
    RoundOutcome outcome = new RoundOutcome(Variant.TENHOU_4P);
    outcome.setRon(8_000, 2, 0, 1, 0b0100);
    outcome.addRonWinner(2_000, 1, NO_SEAT);
    RoundOutcome copy = new RoundOutcome(Variant.TENHOU_4P);
    copy.setAbortive(AbortiveDraw.FOUR_WINDS, 0b0011);
    copy.copyFrom(outcome);
    Assert.assertEquals(copy.toString(), outcome.toString());
    Assert.assertNull(copy.abortive());
    Assert.assertEquals(copy.winner(1), 2);
    Assert.assertEquals(copy.pao(1), 1);
    Assert.assertThrows(IllegalArgumentException.class, () -> copy.copyFrom(new RoundOutcome(Variant.TENHOU_3P)));
  }
}

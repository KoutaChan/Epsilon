package com.epsilon.mahjong.rules;

import static com.epsilon.mahjong.rules.RoundFixtures.abortive;
import static com.epsilon.mahjong.rules.RoundFixtures.advance;
import static com.epsilon.mahjong.rules.RoundFixtures.exhaustive;
import static com.epsilon.mahjong.rules.RoundFixtures.ron;
import static com.epsilon.mahjong.rules.RoundFixtures.scores;
import static com.epsilon.mahjong.rules.RoundFixtures.settle;
import static com.epsilon.mahjong.rules.RoundFixtures.table;
import static com.epsilon.mahjong.rules.RoundFixtures.tsumo;
import static com.epsilon.mahjong.rules.RoundOutcome.NO_SEAT;

import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 親の連荘、局と本場の進み方、半荘の終わり方、終局で残った供託と同点の扱いを、手で作った卓で確かめる。 */
public class ProgressionTest {

  private static final Variant FOUR = Variant.TENHOU_4P;
  private static final Variant THREE = Variant.TENHOU_3P;

  @Test
  public void theDealerStaysOnItsOwnWinOrTenpaiAndOnAbortiveDraws() {
    Assert.assertEquals(Progression.dealerOutcome(tsumo(FOUR, 240, 0, NO_SEAT, 0), 0), DealerOutcome.CONTINUES);
    Assert.assertEquals(Progression.dealerOutcome(tsumo(FOUR, 240, 1, NO_SEAT, 0), 0), DealerOutcome.PASSES);
    RoundOutcome doubleRon = ron(FOUR, 240, 1, 2, 0);
    doubleRon.addRonWinner(240, 3, NO_SEAT);
    Assert.assertEquals(Progression.dealerOutcome(doubleRon, 3), DealerOutcome.CONTINUES);
    Assert.assertEquals(Progression.dealerOutcome(doubleRon, 0), DealerOutcome.PASSES);
    Assert.assertEquals(Progression.dealerOutcome(exhaustive(THREE, 0b100, 0, 0), 2), DealerOutcome.CONTINUES);
    Assert.assertEquals(
        Progression.dealerOutcome(exhaustive(THREE, 0b011, 0b100, 0), 2), DealerOutcome.PASSES, "noten nagashi");
    Assert.assertEquals(
        Progression.dealerOutcome(abortive(THREE, AbortiveDraw.FOUR_KANS, 0), 1), DealerOutcome.ABORTIVE);
  }

  @Test
  public void aDealerWinRepeatsTheRoundWithOneMoreHonba() {
    TableState next = advance(new TableState(FOUR), tsumo(FOUR, 320, 0, NO_SEAT, 0));
    assertTable(next, 0, 1, 0, false);
    Assert.assertEquals(scores(next), new int[] {27_100, 24_300, 24_300, 24_300});
  }

  @Test
  public void aChildWinPassesTheDealerAndClearsTheHonba() {
    TableState start = table(FOUR, 0, 2, 1, 25_000, 25_000, 25_000, 24_000);
    TableState next = advance(start, ron(FOUR, 240, 2, 1, 0));
    assertTable(next, 1, 0, 0, false);
    Assert.assertEquals(next.dealer(), 1);
    Assert.assertEquals(scores(next), new int[] {25_000, 23_400, 27_600, 24_000});
  }

  @Test
  public void drawsAddAHonbaAndKeepTheDeposits() {
    TableState start = table(FOUR, 2, 1, 1, 25_000, 25_000, 25_000, 24_000);
    TableState kept = advance(start, exhaustive(FOUR, 0b0101, 0, 0b0001));
    assertTable(kept, 2, 2, 2, false);
    Assert.assertEquals(scores(kept), new int[] {25_500, 23_500, 26_500, 22_500});
    assertTable(advance(start, exhaustive(FOUR, 0b0001, 0, 0b0001)), 3, 2, 2, false);
    assertTable(advance(start, abortive(FOUR, AbortiveDraw.FOUR_WINDS, 0)), 2, 2, 1, false);
    assertTable(advance(start, abortive(FOUR, AbortiveDraw.FOUR_RIICHI, 0b1111)), 2, 2, 5, false);
  }

  /** ダブロンに親が入っていれば連荘になる。供託は、親でなくても放銃者から手番の順で最初の和了者が受け取る。 */
  @Test
  public void aDoubleRonWithTheDealerRepeatsTheRound() {
    TableState start = table(FOUR, 1, 0, 1, 25_000, 25_000, 25_000, 24_000);
    RoundOutcome outcome = ron(FOUR, 240, 1, 3, 0);
    outcome.addRonWinner(240, 0, NO_SEAT);
    TableState next = advance(start, outcome);
    assertTable(next, 1, 1, 0, false);
    Assert.assertEquals(scores(next), new int[] {27_000, 26_500, 25_000, 21_500});
  }

  @Test
  public void aScoreBelowZeroEndsTheGame() {
    TableState start = table(FOUR, 1, 0, 0, 25_000, 25_000, 49_000, 1_000);
    TableState next = advance(start, ron(FOUR, 480, 0, 3, 0));
    assertTable(next, 1, 0, 0, true);
    Assert.assertEquals(scores(next), new int[] {27_000, 25_000, 49_000, -1_000});
    Assert.assertEquals(next.rank(3), 3);
  }

  /** ちょうど 0 点は飛びではない。立直で 0 点になった席がいても半荘は続く。 */
  @Test
  public void exactlyZeroPlaysOn() {
    TableState start = table(FOUR, 1, 0, 0, 25_000, 25_000, 49_000, 1_000);
    TableState next = advance(start, ron(FOUR, 240, 0, 3, 0));
    assertTable(next, 2, 0, 0, false);
    Assert.assertEquals(next.score(3), 0);
    TableState riichi = advance(start, exhaustive(FOUR, 0b1111, 0, 0b1000));
    assertTable(riichi, 1, 1, 1, false);
    Assert.assertEquals(riichi.score(3), 0);
  }

  /** 飛びは 0 点未満で、ちょうど 0 点は飛ばない。半荘を終えるには返し点以上が要る（四人打ちは 30,000 点、三人打ちは 40,000 点）。 */
  @Test
  public void theBustAndReturnLines() {
    Assert.assertTrue(Progression.busts(-100));
    Assert.assertFalse(Progression.busts(0));
    Assert.assertTrue(Progression.reachesReturn(FOUR, 30_000));
    Assert.assertFalse(Progression.reachesReturn(FOUR, 29_900));
    Assert.assertTrue(Progression.reachesReturn(THREE, 40_000));
    Assert.assertFalse(Progression.reachesReturn(THREE, 39_900));
  }

  /**
   * 南 4 局（親は席 3）、供託 1 本。席 2 が立直し、席 0 が席 1 から 1,000 点をロンする。終局の判定に使う席 0 の点は、場の供託 2 本を
   * 受け取った 28,000 ＋ 1,000 ＋ 2,000 = 31,000 点で、返し点に届いて終わる。
   */
  @Test
  public void theJudgedScoreIncludesTheDepositsTheWinnerCollects() {
    TableState start = table(FOUR, 7, 0, 1, 28_000, 26_000, 24_000, 21_000);
    RoundOutcome outcome = ron(FOUR, 240, 0, 1, 0b0100);
    int[] delta = settle(start, outcome);
    int[] judged = new int[4];
    for (int seat = 0; seat < 4; seat++) {
      judged[seat] = Progression.judgedScore(start, delta, seat);
    }
    Assert.assertEquals(judged, new int[] {31_000, 25_000, 23_000, 21_000});
    TableState over = advance(start, outcome);
    assertTable(over, 7, 0, 0, true);
    Assert.assertEquals(scores(over), judged);
  }

  @Test
  public void aRiichiAtOneThousandBustsOnAnotherSeatsTsumo() {
    TableState start = table(FOUR, 1, 0, 0, 33_000, 33_000, 33_000, 1_000);
    TableState next = advance(start, tsumo(FOUR, 240, 0, NO_SEAT, 0b1000));
    assertTable(next, 1, 0, 0, true);
    Assert.assertEquals(scores(next), new int[] {35_100, 32_500, 32_700, -300});
  }

  @Test
  public void aBustOnADrawGivesTheRemainingDepositsToTheTop() {
    TableState start = table(FOUR, 2, 0, 1, 40_000, 30_000, 28_500, 500);
    TableState next = advance(start, exhaustive(FOUR, 0b0001, 0, 0));
    assertTable(next, 2, 0, 0, true);
    Assert.assertEquals(scores(next), new int[] {44_000, 29_000, 27_500, -500});
  }

  /** 四人打ちの南 4 局で親が流れたら、トップが 30000 点以上なら終わり、届かなければ西入する。 */
  @Test
  public void allLastEndsWhenTheDealerPassesAndTheTopReachesTheReturn() {
    TableState start = table(FOUR, 7, 0, 0, 31_000, 25_000, 24_000, 20_000);
    assertTable(advance(start, ron(FOUR, 240, 1, 2, 0)), 7, 0, 0, true);
    assertTable(advance(start, exhaustive(FOUR, 0b0001, 0, 0)), 7, 0, 0, true);
    TableState low = table(FOUR, 7, 0, 0, 29_000, 25_000, 24_000, 22_000);
    TableState west = advance(low, ron(FOUR, 240, 1, 2, 0));
    assertTable(west, 8, 0, 0, false);
    Assert.assertEquals(west.dealer(), 0);
    assertTable(advance(low, exhaustive(FOUR, 0, 0, 0)), 8, 1, 0, false);
  }

  /** 和了止め・聴牌止め: 親が続くときは、親がトップで返し点以上のときだけ終わる。 */
  @Test
  public void allLastDealerStopsOnlyAsTheTopAtTheReturn() {
    TableState lead = table(FOUR, 7, 0, 0, 26_000, 25_000, 24_000, 25_000);
    TableState stop = advance(lead, tsumo(FOUR, 960, 3, NO_SEAT, 0));
    assertTable(stop, 7, 0, 0, true);
    Assert.assertEquals(scores(stop), new int[] {24_000, 23_000, 22_000, 31_000});

    TableState behind = table(FOUR, 7, 0, 0, 33_000, 25_000, 22_000, 20_000);
    assertTable(advance(behind, tsumo(FOUR, 960, 3, NO_SEAT, 0)), 7, 1, 0, false);

    TableState low = table(FOUR, 7, 0, 0, 24_000, 24_000, 24_000, 28_000);
    assertTable(advance(low, ron(FOUR, 240, 3, 0, 0)), 7, 1, 0, false);

    TableState tenpai = table(FOUR, 7, 0, 0, 24_000, 24_000, 25_000, 27_000);
    TableState tenpaiStop = advance(tenpai, exhaustive(FOUR, 0b1000, 0, 0));
    assertTable(tenpaiStop, 7, 0, 0, true);
    Assert.assertEquals(tenpaiStop.score(3), 30_000, "exactly the return points stop");

    TableState tenpaiBehind = table(FOUR, 7, 0, 0, 34_000, 24_000, 22_000, 20_000);
    assertTable(advance(tenpaiBehind, exhaustive(FOUR, 0b1000, 0, 0)), 7, 1, 0, false);
  }

  /** 途中流局は親の連荘になり、最終局でも半荘は終わらない。 */
  @Test
  public void anAbortiveDrawNeverEndsTheGame() {
    TableState four = table(FOUR, 7, 0, 0, 20_000, 20_000, 20_000, 40_000);
    assertTable(advance(four, abortive(FOUR, AbortiveDraw.NINE_TERMINALS, 0)), 7, 1, 0, false);
    TableState three = table(THREE, 5, 0, 0, 30_000, 30_000, 45_000);
    assertTable(advance(three, abortive(THREE, AbortiveDraw.FOUR_KANS, 0)), 5, 1, 0, false);
    TableState last = table(FOUR, 11, 2, 1, 25_000, 25_000, 25_000, 24_000);
    TableState repeat = advance(last, abortive(FOUR, AbortiveDraw.NINE_TERMINALS, 0));
    assertTable(repeat, 11, 3, 1, false);
    Assert.assertEquals(scores(repeat), new int[] {25_000, 25_000, 25_000, 24_000});
  }

  /** トップが同点なら起家に近い席をトップとする。その席が親でなければ、親が続いても半荘は終わらない。 */
  @Test
  public void aTiedTopGoesToTheSeatNearerTheFirstDealer() {
    TableState four = table(FOUR, 7, 0, 0, 35_000, 16_500, 15_000, 33_500);
    TableState next = advance(four, ron(FOUR, 240, 3, 1, 0));
    Assert.assertEquals(scores(next), new int[] {35_000, 15_000, 15_000, 35_000});
    assertTable(next, 7, 1, 0, false);
    Assert.assertEquals(next.rank(0), 0);
    Assert.assertEquals(next.rank(3), 1);

    TableState three = table(THREE, 6, 0, 0, 39_500, 41_000, 24_500);
    TableState stop = advance(three, ron(THREE, 240, 0, 2, 0));
    Assert.assertEquals(scores(stop), new int[] {41_000, 41_000, 23_000});
    assertTable(stop, 6, 0, 0, true);
  }

  /** 和了止めも終局も、和了者が場の供託を受け取った後の点で決める。 */
  @Test
  public void stoppingCountsTheDepositsTheWinnerReceives() {
    TableState dealer = table(FOUR, 7, 0, 2, 24_000, 24_000, 23_600, 26_400);
    TableState stop = advance(dealer, ron(FOUR, 480, 3, 2, 0));
    assertTable(stop, 7, 0, 0, true);
    Assert.assertEquals(scores(stop), new int[] {24_000, 24_000, 20_700, 31_300});

    TableState child = table(FOUR, 7, 0, 2, 28_000, 25_000, 23_000, 22_000);
    TableState end = advance(child, ron(FOUR, 240, 0, 2, 0));
    assertTable(end, 7, 0, 0, true);
    Assert.assertEquals(scores(end), new int[] {31_000, 25_000, 22_000, 22_000});
  }

  /** 親が自分の立直棒を受け取って、ちょうど返し点のトップになる西場の局は和了止めで終わる。 */
  @Test
  public void aDealerReachingTheReturnWithItsOwnRiichiStickStops() {
    TableState west = table(FOUR, 8, 1, 0, 23_700, 28_700, 22_500, 25_100);
    TableState stop = advance(west, tsumo(FOUR, 960, 0, NO_SEAT, 0b0001));
    assertTable(stop, 8, 1, 0, true);
    Assert.assertEquals(scores(stop), new int[] {30_000, 26_600, 20_400, 23_000});
  }

  /** 西場は、トップが返し点に届いた局で終わる。ただし親が続くときに終わるのは、親がトップのときだけ。 */
  @Test
  public void theWestRoundEndsOnceTheTopReachesTheReturn() {
    TableState west = table(FOUR, 8, 0, 0, 29_000, 26_000, 25_000, 20_000);
    assertTable(advance(west, ron(FOUR, 1_280, 1, 3, 0)), 8, 0, 0, true);
    assertTable(advance(west, ron(FOUR, 240, 1, 3, 0)), 9, 0, 0, false);
    assertTable(advance(west, tsumo(FOUR, 240, 0, NO_SEAT, 0)), 8, 0, 0, true);

    TableState other = table(FOUR, 8, 0, 0, 26_000, 32_000, 22_000, 20_000);
    assertTable(advance(other, tsumo(FOUR, 240, 0, NO_SEAT, 0)), 8, 1, 0, false);
    TableState threeOther = table(THREE, 6, 0, 0, 30_000, 42_000, 33_000);
    assertTable(advance(threeOther, tsumo(THREE, 240, 0, NO_SEAT, 0)), 6, 1, 0, false);
  }

  /** 最終局は、親が流れれば点に関わらず終わる。 */
  @Test
  public void theFinalRoundEndsWhenTheDealerPasses() {
    TableState four = table(FOUR, 11, 2, 1, 25_000, 25_000, 25_000, 24_000);
    assertTable(advance(four, ron(FOUR, 240, 1, 2, 0)), 11, 2, 0, true);
    assertTable(advance(four, tsumo(FOUR, 240, 0, NO_SEAT, 0)), 11, 2, 0, true);
    TableState draw = advance(four, exhaustive(FOUR, 0b0001, 0, 0));
    assertTable(draw, 11, 2, 0, true);
    Assert.assertEquals(scores(draw), new int[] {29_000, 24_000, 24_000, 23_000});
    TableState three = table(THREE, 8, 0, 0, 36_000, 35_000, 34_000);
    assertTable(advance(three, ron(THREE, 240, 0, 1, 0)), 8, 0, 0, true);
  }

  /** 最終局でも親が続けば同じ局を繰り返し、終わるのは親が返し点以上のトップになったときだけ。 */
  @Test
  public void theFinalRoundRepeatsWhileTheDealerContinues() {
    TableState behind = table(FOUR, 11, 2, 1, 31_000, 25_000, 20_000, 23_000);
    assertTable(advance(behind, tsumo(FOUR, 240, 3, NO_SEAT, 0)), 11, 3, 0, false);
    assertTable(advance(behind, exhaustive(FOUR, 0b1000, 0, 0)), 11, 3, 1, false);

    TableState low = table(FOUR, 11, 2, 1, 25_000, 25_000, 25_000, 24_000);
    TableState repeat = advance(low, tsumo(FOUR, 240, 3, NO_SEAT, 0));
    assertTable(repeat, 11, 3, 0, false);
    Assert.assertEquals(scores(repeat), new int[] {24_300, 24_300, 24_300, 27_100});

    TableState lead = table(FOUR, 11, 0, 0, 25_000, 25_000, 22_000, 28_000);
    assertTable(advance(lead, tsumo(FOUR, 480, 3, NO_SEAT, 0)), 11, 0, 0, true);

    TableState three = table(THREE, 8, 0, 0, 36_000, 35_000, 34_000);
    assertTable(advance(three, exhaustive(THREE, 0b100, 0, 0)), 8, 1, 0, false);
  }

  /**
   * 点の配列を受け取る {@code finishes} は、どの局と点でも、また親が流れても続いても途中流局でも、同じ局と点を持つ卓を受け取る
   * {@code finishes} と同じ答えを返す。点と供託の合計は問わない。
   */
  @Test
  public void theScoreArrayFormDecidesLikeTheTableForm() {
    Random random = new Random(23);
    for (Variant variant : Variant.values()) {
      int[] scores = new int[variant.seats()];
      TableState table = new TableState(variant);
      for (int trial = 0; trial < 20_000; trial++) {
        int round = random.nextInt(variant.finalRound() + 1);
        for (int seat = 0; seat < scores.length; seat++) {
          scores[seat] = (random.nextInt(600) - 20) * 100;
          table.setScore(seat, scores[seat]);
        }
        table.setRound(round, 0, 0);
        for (DealerOutcome dealer : DealerOutcome.values()) {
          Assert.assertEquals(
              Progression.finishes(variant, round, scores, dealer),
              Progression.finishes(table, dealer),
              table + " " + dealer);
        }
      }
    }
  }

  /** 点の配列を受け取る {@code finishes} は、点の合計が合わなくても終局を決める。値の検査は局の番号と点の個数だけ。 */
  @Test
  public void theScoreArrayFormIgnoresTheTotalAndChecksTheShape() {
    int[] high = {31_000, 31_000, 31_000, 31_000};
    Assert.assertTrue(Progression.finishes(FOUR, 7, high, DealerOutcome.PASSES));
    Assert.assertFalse(
        Progression.finishes(FOUR, 7, high, DealerOutcome.CONTINUES), "a tie gives the top to seat 0, not the dealer");
    Assert.assertFalse(Progression.finishes(FOUR, 6, high, DealerOutcome.PASSES));
    Assert.assertTrue(Progression.finishes(FOUR, 0, new int[] {-100, 0, 0, 0}, DealerOutcome.ABORTIVE));
    Assert.assertTrue(Progression.finishes(THREE, 8, new int[] {1_000, 1_000, 1_000}, DealerOutcome.PASSES));
    Assert.assertFalse(Progression.finishes(THREE, 8, new int[] {1_000, 1_000, 1_000}, DealerOutcome.CONTINUES));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Progression.finishes(FOUR, -1, high, DealerOutcome.PASSES));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Progression.finishes(FOUR, 12, high, DealerOutcome.PASSES));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Progression.finishes(THREE, 0, high, DealerOutcome.PASSES));
  }

  /** 三人打ちは南 3 局から、トップが 40000 点以上なら終わる。 */
  @Test
  public void threePlayersStopAtFortyThousandFromSouthThree() {
    TableState south3 = table(THREE, 5, 0, 0, 39_000, 36_000, 30_000);
    assertTable(advance(south3, ron(THREE, 240, 0, 1, 0)), 5, 0, 0, true);
    assertTable(advance(south3, exhaustive(THREE, 0, 0, 0)), 6, 1, 0, false);
    TableState south2 = table(THREE, 4, 0, 0, 45_000, 30_000, 30_000);
    assertTable(advance(south2, ron(THREE, 240, 0, 1, 0)), 5, 0, 0, false);
  }

  /** 終局で残った供託はトップに渡し、同点なら起家に近い席が受け取る。和了で終われば和了者が受け取る。 */
  @Test
  public void remainingDepositsGoToTheTopAndTiesToTheSeatNearerTheFirstDealer() {
    TableState draw = table(FOUR, 7, 0, 1, 31_000, 31_000, 20_000, 17_000);
    TableState end = advance(draw, exhaustive(FOUR, 0b0100, 0, 0b0100));
    assertTable(end, 7, 0, 0, true);
    Assert.assertEquals(scores(end), new int[] {32_000, 30_000, 22_000, 16_000});
    Assert.assertEquals(end.rank(0), 0);
    Assert.assertEquals(end.rank(1), 1);

    TableState win = table(FOUR, 7, 0, 2, 35_000, 25_000, 20_000, 18_000);
    TableState won = advance(win, ron(FOUR, 240, 2, 3, 0));
    assertTable(won, 7, 0, 0, true);
    Assert.assertEquals(scores(won), new int[] {35_000, 25_000, 23_000, 17_000});
  }

  @Test
  public void intoMayBeTheStartTable() {
    TableState start = table(FOUR, 3, 1, 1, 30_000, 20_000, 26_000, 23_000);
    RoundOutcome outcome = ron(FOUR, 960, 1, 2, 0b0001);
    TableState expected = advance(start, outcome);
    Progression.next(start, outcome, settle(start, outcome), start);
    Assert.assertEquals(scores(start), scores(expected));
    assertTable(start, expected.round(), expected.honba(), expected.kyotaku(), expected.gameOver());
  }

  @Test
  public void inconsistentInputsAreRejected() {
    TableState start = new TableState(FOUR);
    RoundOutcome outcome = ron(FOUR, 240, 1, 0, 0);
    int[] delta = settle(start, outcome);
    TableState into = new TableState(FOUR);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Progression.next(start, outcome, new int[] {-1_000, 900, 0, 0}, into));
    Assert.assertThrows(IllegalArgumentException.class, () -> Progression.next(start, outcome, new int[3], into));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Progression.next(start, outcome, delta, new TableState(THREE)));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Progression.next(start, new RoundOutcome(THREE), delta, into));
    TableState over = new TableState(FOUR);
    over.finish();
    Assert.assertThrows(IllegalArgumentException.class, () -> Progression.next(over, outcome, delta, into));
  }

  private static void assertTable(TableState table, int round, int honba, int kyotaku, boolean gameOver) {
    Assert.assertEquals(table.round(), round, table.toString());
    Assert.assertEquals(table.honba(), honba, table.toString());
    Assert.assertEquals(table.kyotaku(), kyotaku, table.toString());
    Assert.assertEquals(table.gameOver(), gameOver, table.toString());
  }
}

package com.epsilon.mahjong.rules;

import static com.epsilon.mahjong.rules.RoundFixtures.abortive;
import static com.epsilon.mahjong.rules.RoundFixtures.exhaustive;
import static com.epsilon.mahjong.rules.RoundFixtures.ron;
import static com.epsilon.mahjong.rules.RoundFixtures.settle;
import static com.epsilon.mahjong.rules.RoundFixtures.table;
import static com.epsilon.mahjong.rules.RoundFixtures.tsumo;
import static com.epsilon.mahjong.rules.RoundOutcome.NO_SEAT;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 天鳳の点の移動を、手で求めた値で確かめる（四人打ちと三人打ち）。 */
public class SettlementTest {

  private static final Variant FOUR = Variant.TENHOU_4P;
  private static final Variant THREE = Variant.TENHOU_3P;

  @Test
  public void theDiscarderPaysTheRon() {
    TableState four = new TableState(FOUR);
    Assert.assertEquals(settle(four, ron(FOUR, 1_920, 2, 1, 0)), new int[] {0, -7_700, 7_700, 0});
    Assert.assertEquals(settle(four, ron(FOUR, 1_920, 0, 3, 0)), new int[] {11_600, 0, 0, -11_600});
    TableState three = new TableState(THREE);
    Assert.assertEquals(settle(three, ron(THREE, 2_000, 1, 2, 0)), new int[] {0, 8_000, -8_000});
    Assert.assertEquals(settle(three, ron(THREE, 2_000, 0, 2, 0)), new int[] {12_000, 0, -12_000});
  }

  @Test
  public void everyOtherSeatPaysAFourPlayerTsumo() {
    TableState start = new TableState(FOUR);
    Assert.assertEquals(settle(start, tsumo(FOUR, 240, 1, NO_SEAT, 0)), new int[] {-500, 1_100, -300, -300});
    Assert.assertEquals(settle(start, tsumo(FOUR, 320, 0, NO_SEAT, 0)), new int[] {2_100, -700, -700, -700});
  }

  /** 三人打ちのツモ。親の 20 符 4 翻は 2 席が 2600 点ずつ払い、子の跳満は 6000 点と 3000 点（いない席の分は誰も払わない）。 */
  @Test
  public void aThreePlayerTsumoLosesTheMissingSeatsShare() {
    TableState start = new TableState(THREE);
    Assert.assertEquals(settle(start, tsumo(THREE, 1_280, 0, NO_SEAT, 0)), new int[] {5_200, -2_600, -2_600});
    Assert.assertEquals(settle(start, tsumo(THREE, 3_000, 1, NO_SEAT, 0)), new int[] {-6_000, 9_000, -3_000});
  }

  /** 本場 1 本につき、ロンなら放銃者が 100 × (席の数 − 1) 点、ツモなら払う席がそれぞれ 100 点を足して払う。 */
  @Test
  public void honbaAddsAHundredPerPayingSeatAndPerOtherSeatOnARon() {
    TableState east2 = table(FOUR, 1, 2, 0, 25_000, 25_000, 25_000, 25_000);
    Assert.assertEquals(settle(east2, ron(FOUR, 240, 3, 0, 0)), new int[] {-1_600, 0, 0, 1_600});
    Assert.assertEquals(settle(east2, tsumo(FOUR, 240, 2, NO_SEAT, 0)), new int[] {-500, -700, 1_700, -500});
    TableState three = table(THREE, 0, 3, 0, 35_000, 35_000, 35_000);
    Assert.assertEquals(settle(three, ron(THREE, 2_000, 2, 0, 0)), new int[] {-8_600, 0, 8_600});
    Assert.assertEquals(settle(three, tsumo(THREE, 2_000, 1, NO_SEAT, 0)), new int[] {-4_300, 6_600, -2_300});
  }

  /** 立直の席は 1000 点を場に出す。和了なら場の供託を全て最初の和了者が受け取り、流局なら場に残る。 */
  @Test
  public void riichiDepositsGoToTheWinnerWithTheTableDeposits() {
    TableState start = table(FOUR, 0, 0, 2, 25_000, 24_000, 25_000, 24_000);
    Assert.assertEquals(settle(start, ron(FOUR, 2_000, 2, 3, 0b0101)), new int[] {-1_000, 0, 11_000, -8_000});
    Assert.assertEquals(
        settle(start, tsumo(FOUR, 2_000, 0, NO_SEAT, 0b0001)), new int[] {14_000, -4_000, -4_000, -4_000});
    Assert.assertEquals(settle(start, exhaustive(FOUR, 0b0101, 0, 0b0101)), new int[] {500, -1_500, 500, -1_500});

    TableState three = table(THREE, 0, 0, 1, 35_000, 34_000, 35_000);
    Assert.assertEquals(settle(three, ron(THREE, 2_000, 2, 1, 0b001)), new int[] {-1_000, -8_000, 10_000});
    Assert.assertEquals(settle(three, tsumo(THREE, 2_000, 0, NO_SEAT, 0b001)), new int[] {9_000, -4_000, -4_000});
    Assert.assertEquals(settle(three, exhaustive(THREE, 0b011, 0, 0b001)), new int[] {0, 1_000, -2_000});
  }

  /** ダブロンの本場と供託は、放銃者から手番の順で最初の和了者だけが受け取る。 */
  @Test
  public void doubleRonGivesHonbaAndDepositsToTheFirstWinnerFromTheDiscarder() {
    TableState four = table(FOUR, 0, 1, 1, 25_000, 25_000, 25_000, 24_000);
    RoundOutcome fourRon = ron(FOUR, 2_000, 1, 2, 0);
    fourRon.addRonWinner(480, 0, NO_SEAT);
    Assert.assertEquals(fourRon.winner(0), 0);
    Assert.assertEquals(settle(four, fourRon), new int[] {4_200, 8_000, -11_200, 0});

    // 席 2 は 3900 ＋ 本場 200 ＋ 供託 2 本を受け取り、自分の立直の 1000 点を払う。
    TableState three = table(THREE, 0, 1, 1, 35_000, 34_000, 35_000);
    RoundOutcome threeRon = ron(THREE, 960, 0, 1, 0b100);
    threeRon.addRonWinner(960, 2, NO_SEAT);
    Assert.assertEquals(threeRon.winner(0), 2);
    Assert.assertEquals(settle(three, threeRon), new int[] {5_800, -9_900, 5_100});
  }

  /** 包: ツモなら包の席が全額と本場を払う。ロンなら包の席と放銃者が折半し、本場は包の席が払う。 */
  @Test
  public void paoPaysTheWholeTsumoAndHalfTheRon() {
    TableState start = table(FOUR, 0, 1, 0, 25_000, 25_000, 25_000, 25_000);
    Assert.assertEquals(settle(start, tsumo(FOUR, 8_000, 1, 3, 0)), new int[] {0, 32_300, 0, -32_300});
    Assert.assertEquals(settle(start, tsumo(FOUR, 16_000, 0, 2, 0)), new int[] {96_300, 0, -96_300, 0});

    RoundOutcome outcome = new RoundOutcome(FOUR);
    outcome.setRon(8_000, 2, 1, 0, 0);
    Assert.assertEquals(settle(start, outcome), new int[] {-16_300, -16_000, 32_300, 0});

    outcome.setRon(8_000, 1, 0, 3, 0);
    outcome.addRonWinner(2_000, 2, NO_SEAT);
    Assert.assertEquals(settle(start, outcome), new int[] {-24_000, 32_300, 8_000, -16_300});
  }

  /** ダブロンの包は和了者ごとに決まる。本場は最初の和了者の分だけなので、2 人目の和了者に付いた包の席は折半の額だけを払う。 */
  @Test
  public void eachDoubleRonWinnerHasItsOwnPao() {
    TableState start = table(FOUR, 0, 1, 0, 25_000, 25_000, 25_000, 25_000);
    RoundOutcome second = new RoundOutcome(FOUR);
    second.setRon(2_000, 1, 0, NO_SEAT, 0);
    second.addRonWinner(8_000, 2, 3);
    Assert.assertEquals(settle(start, second), new int[] {-24_300, 8_300, 32_000, -16_000});

    RoundOutcome different = new RoundOutcome(FOUR);
    different.setRon(8_000, 1, 0, 3, 0);
    different.addRonWinner(8_000, 2, 1);
    Assert.assertEquals(settle(start, different), new int[] {-32_000, 16_300, 32_000, -16_300});

    RoundOutcome same = new RoundOutcome(FOUR);
    same.setRon(8_000, 1, 0, 3, 0);
    same.addRonWinner(8_000, 2, 3);
    Assert.assertEquals(settle(start, same), new int[] {-32_000, 32_300, 32_000, -32_300});
  }

  /**
   * 三人打ちの包: ツモなら、ツモ損で 2 席分を足した額と本場を包の席が払う。ロンなら包の席と放銃者が折半し、本場（200 点）は包の席が
   * 払う。
   */
  @Test
  public void aThreePlayerPaoPaysTheTwoTsumoSharesAndHalfTheRon() {
    TableState start = table(THREE, 0, 1, 0, 35_000, 35_000, 35_000);
    Assert.assertEquals(settle(start, tsumo(THREE, 8_000, 2, 1, 0)), new int[] {0, -24_200, 24_200});
    Assert.assertEquals(settle(start, tsumo(THREE, 8_000, 0, 2, 0)), new int[] {32_200, 0, -32_200});
    TableState east2 = table(THREE, 1, 1, 0, 35_000, 35_000, 35_000);
    Assert.assertEquals(settle(east2, tsumo(THREE, 8_000, 2, 1, 0)), new int[] {0, -24_200, 24_200});

    RoundOutcome outcome = new RoundOutcome(THREE);
    outcome.setRon(8_000, 2, 0, 1, 0);
    Assert.assertEquals(settle(start, outcome), new int[] {-16_000, -16_200, 32_200});
    outcome.setRon(8_000, 2, 1, 0, 0);
    Assert.assertEquals(settle(start, outcome), new int[] {-16_200, -16_000, 32_200});
  }

  /**
   * 包が付くのは役満の和了だけで、包の席は和了者でもロンの放銃者でもない（包の席が放銃したロンは包の無いロンとして書く）。外れれば
   * IllegalArgumentException を投げ、点の移動は書き換えない。自分の打牌へのロンも同じく拒む。
   */
  @Test
  public void paoNeedsAYakumanFromAnotherSeat() {
    int[] delta = new int[4];
    Assert.assertThrows(IllegalArgumentException.class, () -> Settlement.tsumo(FOUR, 2_000, 1, 0, 0, 3, delta));
    Assert.assertThrows(IllegalArgumentException.class, () -> Settlement.tsumo(FOUR, 8_000, 1, 0, 0, 1, delta));
    Assert.assertThrows(IllegalArgumentException.class, () -> Settlement.ron(FOUR, 8_000, 1, 0, 0, 0, 1, delta));
    Assert.assertThrows(IllegalArgumentException.class, () -> Settlement.ron(FOUR, 8_000, 1, 0, 0, 0, 0, delta));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Settlement.ron(FOUR, 8_000, 1, 1, 0, 0, NO_SEAT, delta));
    Assert.assertEquals(delta, new int[4]);
  }

  /** 不聴罰符: 総額 1000 × (席の数 − 1) 点を、不聴の席が等分して払い、聴牌の席が等分して受け取る。 */
  @Test
  public void notenPaymentsForEveryTenpaiCount() {
    TableState four = new TableState(FOUR);
    Assert.assertEquals(settle(four, exhaustive(FOUR, 0b0000, 0, 0)), new int[] {0, 0, 0, 0});
    Assert.assertEquals(settle(four, exhaustive(FOUR, 0b0100, 0, 0)), new int[] {-1_000, -1_000, 3_000, -1_000});
    Assert.assertEquals(settle(four, exhaustive(FOUR, 0b1001, 0, 0)), new int[] {1_500, -1_500, -1_500, 1_500});
    Assert.assertEquals(settle(four, exhaustive(FOUR, 0b0111, 0, 0)), new int[] {1_000, 1_000, 1_000, -3_000});
    Assert.assertEquals(settle(four, exhaustive(FOUR, 0b1111, 0, 0)), new int[] {0, 0, 0, 0});
    TableState three = new TableState(THREE);
    Assert.assertEquals(settle(three, exhaustive(THREE, 0b000, 0, 0)), new int[] {0, 0, 0});
    Assert.assertEquals(settle(three, exhaustive(THREE, 0b010, 0, 0)), new int[] {-1_000, 2_000, -1_000});
    Assert.assertEquals(settle(three, exhaustive(THREE, 0b101, 0, 0)), new int[] {1_000, -2_000, 1_000});
    Assert.assertEquals(settle(three, exhaustive(THREE, 0b111, 0, 0)), new int[] {0, 0, 0});
  }

  /** どの聴牌の組み合わせでも、点の移動の合計は 0。聴牌の席はどれも同じ額を受け取り、不聴の席はどれも同じ額を払う。 */
  @Test
  public void notenPaymentsBalanceForEveryMask() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      for (int mask = 0; mask <= variant.seatMask(); mask++) {
        int[] delta = new int[seats];
        Settlement.exhaustive(variant, 0, mask, 0, delta);
        int tenpai = Integer.bitCount(mask);
        int sum = 0;
        for (int seat = 0; seat < seats; seat++) {
          sum += delta[seat];
          int expected = 0;
          if (tenpai > 0 && tenpai < seats) {
            boolean listening = (mask & (1 << seat)) != 0;
            expected = listening ? variant.notenTotal() / tenpai : -variant.notenTotal() / (seats - tenpai);
          }
          Assert.assertEquals(delta[seat], expected, variant + " " + Integer.toBinaryString(mask));
        }
        Assert.assertEquals(sum, 0, variant + " " + Integer.toBinaryString(mask));
      }
    }
  }

  /** 流し満貫は満貫のツモと同じ払い（三人打ちはツモ損）で、本場と不聴罰符は付かない。 */
  @Test
  public void nagashiManganIsAManganTsumoWithoutHonba() {
    TableState four = table(FOUR, 0, 3, 0, 25_000, 25_000, 25_000, 25_000);
    Assert.assertEquals(settle(four, exhaustive(FOUR, 0b0001, 0b0100, 0)), new int[] {-4_000, -2_000, 8_000, -2_000});
    Assert.assertEquals(settle(four, exhaustive(FOUR, 0, 0b0001, 0)), new int[] {12_000, -4_000, -4_000, -4_000});
    Assert.assertEquals(settle(four, exhaustive(FOUR, 0, 0b0110, 0)), new int[] {-8_000, 6_000, 6_000, -4_000});
    TableState three = table(THREE, 0, 1, 0, 35_000, 35_000, 35_000);
    Assert.assertEquals(settle(three, exhaustive(THREE, 0b011, 0b010, 0)), new int[] {-4_000, 6_000, -2_000});
    Assert.assertEquals(settle(three, exhaustive(THREE, 0, 0b001, 0)), new int[] {8_000, -4_000, -4_000});
  }

  @Test
  public void abortiveDrawsMoveOnlyTheRiichiDeposits() {
    TableState four = table(FOUR, 2, 1, 1, 25_000, 25_000, 25_000, 24_000);
    Assert.assertEquals(
        settle(four, abortive(FOUR, AbortiveDraw.FOUR_RIICHI, 0b1111)), new int[] {-1_000, -1_000, -1_000, -1_000});
    Assert.assertEquals(settle(four, abortive(FOUR, AbortiveDraw.TRIPLE_RON, 0)), new int[] {0, 0, 0, 0});
    TableState three = new TableState(THREE);
    Assert.assertEquals(
        settle(three, abortive(THREE, AbortiveDraw.NINE_TERMINALS, 0b001)), new int[] {-1_000, 0, 0});
  }

  @Test
  public void applyOverwritesTheDeltaWhileTheStepsAddToIt() {
    TableState start = new TableState(FOUR);
    int[] delta = {5, 5, 5, 5};
    Settlement.apply(FOUR, start, ron(FOUR, 240, 1, 0, 0), delta);
    Assert.assertEquals(delta, new int[] {-1_000, 1_000, 0, 0});
    Settlement.ron(FOUR, 240, 2, 0, 0, 0, NO_SEAT, delta);
    Assert.assertEquals(delta, new int[] {-2_000, 1_000, 1_000, 0});
    Settlement.exhaustive(FOUR, 0, 0b0001, 0, delta);
    Assert.assertEquals(delta, new int[] {1_000, 0, 0, -1_000});
  }

  @Test
  public void mismatchedInputsAreRejected() {
    TableState four = new TableState(FOUR);
    RoundOutcome outcome = ron(FOUR, 240, 1, 0, 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> Settlement.apply(THREE, four, outcome, new int[3]));
    Assert.assertThrows(IllegalArgumentException.class, () -> Settlement.apply(FOUR, four, outcome, new int[3]));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Settlement.apply(FOUR, four, new RoundOutcome(THREE), new int[4]));
    TableState over = new TableState(FOUR);
    over.finish();
    Assert.assertThrows(IllegalArgumentException.class, () -> Settlement.apply(FOUR, over, outcome, new int[4]));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Settlement.exhaustive(THREE, 0, 0b1000, 0, new int[3]));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Settlement.tsumo(FOUR, 2_000, 1, 0, -1, NO_SEAT, new int[4]));
    Assert.assertThrows(
        IndexOutOfBoundsException.class, () -> Settlement.tsumo(THREE, 2_000, 3, 0, 0, NO_SEAT, new int[3]));
  }
}

package com.epsilon.grp.rules;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.Seats;
import com.epsilon.grp.train.HandResult;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 局の結果のコードを確かめる。確かめるのは、全ての終わり方の往復（四人打ちと三人打ち、全ての親）、コードから書き戻しても精算と局の
 * 進みが変わらないこと、置けないコードを拒むこと、コードの順とフィールドの位置。
 */
public class HandResultTest {

  /** フィールドの位置（上の桁から順に、終わり方 2・和了者 3・放銃者 3・打点 5・包 3・2 人目 3・打点 5・包 3・聴牌 4・流し 4・立直 4）。 */
  private static final int END_SHIFT = 37;

  private static final int LOSER_SHIFT = 31;
  private static final int BASE_SHIFT = 26;
  private static final int SECOND_SHIFT = 20;
  private static final int TENPAI_SHIFT = 8;

  private static final int[] SOME_BASES = {
    BaseClass.B240.base(), BaseClass.B960.base(), BaseClass.B1920.base(), BaseClass.MANGAN.base(),
    BaseClass.SANBAIMAN.base(), BaseClass.YAKUMAN.base(), BaseClass.YAKUMAN_6.base()
  };

  @Test
  public void everyEndingRoundTrips() {
    for (Variant variant : Variant.values()) {
      int checked = 0;
      for (int dealer = 0; dealer < variant.seats(); dealer++) {
        for (RoundOutcome outcome : outcomes(variant)) {
          assertRoundTrip(outcome, dealer);
          checked++;
        }
      }
      Assert.assertTrue(checked > 1_000, variant + ": only " + checked + " outcomes were checked");
    }
  }

  @Test
  public void placedCodesSettleAndProgressTheSame() {
    Random random = new Random(20261110);
    for (Variant variant : Variant.values()) {
      List<RoundOutcome> outcomes = outcomes(variant);
      RoundOutcome placed = new RoundOutcome(variant);
      TableState start = new TableState(variant);
      TableState original = new TableState(variant);
      TableState placedNext = new TableState(variant);
      int[] delta = new int[variant.seats()];
      int[] placedDelta = new int[variant.seats()];
      for (int trial = 0; trial < 20_000; trial++) {
        GrpFixtures.state(variant, random).copyInto(start);
        RoundOutcome outcome = outcomes.get(random.nextInt(outcomes.size()));
        HandResult.place(HandResult.raw(outcome, start.dealer()), start.dealer(), placed);
        Settlement.apply(variant, start, outcome, delta);
        Settlement.apply(variant, start, placed, placedDelta);
        Progression.next(start, outcome, delta, original);
        Progression.next(start, placed, placedDelta, placedNext);
        // 途中流局の種類や流し満貫の子の聴牌を落としても、点の動きと次の卓は同じ。
        String context = outcome + " from " + start;
        Assert.assertEquals(placedDelta, delta, context);
        Assert.assertEquals(placedNext.toString(), original.toString(), context);
      }
    }
  }

  @Test
  public void abortiveDrawsPlaceTheFirstKindOfTheVariant() {
    for (Variant variant : Variant.values()) {
      RoundOutcome outcome = new RoundOutcome(variant);
      outcome.setAbortive(AbortiveDraw.FOUR_KANS, 0b1);
      long code = HandResult.raw(outcome, 0);
      RoundOutcome placed = new RoundOutcome(variant);
      HandResult.place(code, 0, placed);
      Assert.assertEquals(placed.kind(), EndKind.ABORTIVE);
      Assert.assertEquals(placed.abortive(), AbortiveDraw.NINE_TERMINALS);
      Assert.assertEquals(placed.riichiMask(), 0b1);
    }
  }

  @Test
  public void codesOrderByEndingThenWinner() {
    Variant variant = Variant.TENHOU_4P;
    long tsumo = RulesFixtures.code(variant, 0, o -> o.setTsumo(BaseClass.YAKUMAN_6.base(), 3, RoundOutcome.NO_SEAT, 0));
    long ron = RulesFixtures.code(variant, 0, o -> o.setRon(BaseClass.B240.base(), 1, 2, RoundOutcome.NO_SEAT, 0));
    long laterRon = RulesFixtures.code(variant, 0, o -> o.setRon(BaseClass.B240.base(), 2, 3, RoundOutcome.NO_SEAT, 0));
    long exhaustive = RulesFixtures.code(variant, 0, o -> o.setExhaustive(0b1111, 0, 0b1111));
    long abortive = RulesFixtures.code(variant, 0, o -> o.setAbortive(AbortiveDraw.NINE_TERMINALS, 0));
    Assert.assertTrue(tsumo < ron && ron < laterRon && laterRon < exhaustive && exhaustive < abortive);
    Assert.assertEquals(HandResult.BITS, 39);
    for (long code : new long[] {tsumo, ron, exhaustive, abortive}) {
      Assert.assertEquals(code >>> HandResult.BITS, 0L);
      Assert.assertEquals(code >>> END_SHIFT, HandResult.end(code).ordinal());
    }
    Assert.assertEquals(HandResult.riichi(exhaustive), 0b1111);
  }

  @Test
  public void malformedCodesAreRejected() {
    Variant four = Variant.TENHOU_4P;
    RoundOutcome scratch = new RoundOutcome(four);
    long tsumo = RulesFixtures.code(four, 0, o -> o.setTsumo(BaseClass.MANGAN.base(), 1, RoundOutcome.NO_SEAT, 0));
    long ron = RulesFixtures.code(four, 0, o -> o.setRon(BaseClass.MANGAN.base(), 1, 2, RoundOutcome.NO_SEAT, 0));
    long doubleRon = RulesFixtures.code(four, 0, o -> {
      o.setRon(BaseClass.MANGAN.base(), 1, 0, RoundOutcome.NO_SEAT, 0);
      o.addRonWinner(BaseClass.B960.base(), 3, RoundOutcome.NO_SEAT);
    });
    long nagashi = RulesFixtures.code(four, 0, o -> o.setExhaustive(0b0001, 0b0010, 0));
    for (long code : new long[] {tsumo, ron, doubleRon, nagashi}) {
      Assert.assertTrue(HandResult.valid(code, scratch), Long.toHexString(code));
    }
    List<Long> broken = new ArrayList<>();
    broken.add(-1L);
    broken.add(tsumo | 1L << HandResult.BITS);
    // ツモなのに放銃者のフィールドがある
    broken.add(tsumo | 2L << LOSER_SHIFT);
    // 表に無い打点の番号
    broken.add(tsumo & ~(31L << BASE_SHIFT) | (long) BaseClass.values().length << BASE_SHIFT);
    // 役満でない和了に包がある
    broken.add(tsumo | 3L << 23);
    // ダブロンで和了者の順が逆（2 人目の方が放銃者に近い）
    long first = doubleRon >>> 34 & 7;
    long second = doubleRon >>> SECOND_SHIFT & 7;
    long baseA = doubleRon >>> BASE_SHIFT & 31;
    long baseB = doubleRon >>> 15 & 31;
    long swapped = doubleRon & ~(7L << 34 | 7L << SECOND_SHIFT | 31L << BASE_SHIFT | 31L << 15)
        | second << 34 | first << SECOND_SHIFT | baseB << BASE_SHIFT | baseA << 15;
    broken.add(swapped);
    // 流し満貫のある荒牌で、子が聴牌している
    broken.add(nagashi | 0b0100L << TENPAI_SHIFT);
    // ツモなのに 2 人目の和了者がいる
    broken.add(tsumo | 3L << SECOND_SHIFT);
    for (long code : broken) {
      Assert.assertFalse(HandResult.valid(code, scratch), Long.toHexString(code));
    }
    // 三人打ちには無い相対席 3
    RoundOutcome three = new RoundOutcome(Variant.TENHOU_3P);
    long fourthSeat = RulesFixtures.code(four, 0, o -> o.setTsumo(BaseClass.MANGAN.base(), 3, RoundOutcome.NO_SEAT, 0));
    Assert.assertTrue(HandResult.valid(fourthSeat, scratch));
    Assert.assertFalse(HandResult.valid(fourthSeat, three));
  }

  /** 往復を確かめる。置いた結果を詰め直すと同じコードになる。 */
  private static void assertRoundTrip(RoundOutcome outcome, int dealer) {
    Variant variant = outcome.variant();
    int seats = variant.seats();
    long code = HandResult.raw(outcome, dealer);
    String context = outcome + " with dealer " + dealer;
    Assert.assertEquals(code >>> HandResult.BITS, 0L, context);
    RoundOutcome scratch = new RoundOutcome(variant);
    Assert.assertTrue(HandResult.valid(code, scratch), context);
    Assert.assertEquals(HandResult.end(code), outcome.kind(), context);
    RoundOutcome placed = new RoundOutcome(variant);
    HandResult.place(code, dealer, placed);
    Assert.assertEquals(HandResult.raw(placed, dealer), code, context);
    Assert.assertEquals(placed.kind(), outcome.kind(), context);
    int riichi = outcome.riichiMask();
    switch (outcome.kind()) {
      case TSUMO, RON -> {
        Assert.assertEquals(placed.winnerCount(), outcome.winnerCount(), context);
        for (int i = 0; i < outcome.winnerCount(); i++) {
          Assert.assertEquals(placed.winner(i), outcome.winner(i), context);
          Assert.assertEquals(placed.base(i), outcome.base(i), context);
          Assert.assertEquals(placed.pao(i), outcome.pao(i), context);
        }
        Assert.assertEquals(placed.loser(), outcome.loser(), context);
      }
      case EXHAUSTIVE -> {
        Assert.assertEquals(placed.nagashiMask(), outcome.nagashiMask(), context);
        int tenpai = outcome.nagashiMask() != 0 ? outcome.tenpaiMask() & 1 << dealer : outcome.tenpaiMask();
        Assert.assertEquals(placed.tenpaiMask(), tenpai, context);
      }
      case ABORTIVE -> Assert.assertEquals(placed.abortive(), AbortiveDraw.NINE_TERMINALS, context);
    }
    Assert.assertEquals(placed.riichiMask(), riichi, context);
    int relative = 0;
    for (int seat = 0; seat < seats; seat++) {
      if ((riichi & 1 << seat) != 0) {
        relative |= 1 << Seats.relative(dealer, seat, seats);
      }
    }
    Assert.assertEquals(HandResult.riichi(code), relative, context);
  }

  /** そのルールで書ける結果を広く並べる（全ての和了者・放銃者・包の組、全ての聴牌・流し満貫の組、いくつかの立直）。 */
  private static List<RoundOutcome> outcomes(Variant variant) {
    int seats = variant.seats();
    int all = variant.seatMask();
    int[] riichiMasks = {0, 1, 1 << seats - 1, all, 0b0101 & all};
    List<RoundOutcome> outcomes = new ArrayList<>();
    for (int riichi : riichiMasks) {
      for (int seat = 0; seat < seats; seat++) {
        int winner = seat;
        for (int base : SOME_BASES) {
          outcomes.add(outcome(variant, o -> o.setTsumo(base, winner, RoundOutcome.NO_SEAT, riichi)));
          for (int loser = 0; loser < seats; loser++) {
            if (loser == winner) {
              continue;
            }
            int ronLoser = loser;
            outcomes.add(outcome(variant, o -> o.setRon(base, winner, ronLoser, RoundOutcome.NO_SEAT, riichi)));
            for (int second = 0; second < seats; second++) {
              if (second == winner || second == loser) {
                continue;
              }
              int other = second;
              outcomes.add(outcome(variant, o -> {
                o.setRon(base, winner, ronLoser, RoundOutcome.NO_SEAT, riichi);
                o.addRonWinner(SOME_BASES[(other + 2) % SOME_BASES.length], other, RoundOutcome.NO_SEAT);
              }));
            }
          }
        }
        for (int pao = 0; pao < seats; pao++) {
          if (pao == winner) {
            continue;
          }
          int liable = pao;
          int yakuman = BaseClass.YAKUMAN_2.base();
          outcomes.add(outcome(variant, o -> o.setTsumo(yakuman, winner, liable, riichi)));
          for (int loser = 0; loser < seats; loser++) {
            if (loser == winner || loser == pao) {
              continue;
            }
            int ronLoser = loser;
            outcomes.add(outcome(variant, o -> o.setRon(yakuman, winner, ronLoser, liable, riichi)));
            for (int second = 0; second < seats; second++) {
              if (second == winner || second == loser) {
                continue;
              }
              int other = second;
              int otherPao = liableFor(other, ronLoser, seats);
              outcomes.add(outcome(variant, o -> {
                o.setRon(yakuman, winner, ronLoser, liable, riichi);
                o.addRonWinner(BaseClass.YAKUMAN.base(), other, otherPao);
              }));
            }
          }
        }
      }
      for (int tenpai = 0; tenpai <= all; tenpai++) {
        int mask = tenpai | riichi;
        outcomes.add(outcome(variant, o -> o.setExhaustive(mask, 0, riichi)));
        for (int nagashi = 1; nagashi <= all; nagashi++) {
          int flowing = nagashi;
          outcomes.add(outcome(variant, o -> o.setExhaustive(mask, flowing, riichi)));
        }
      }
      outcomes.add(outcome(variant, o -> o.setAbortive(AbortiveDraw.NINE_TERMINALS, riichi)));
    }
    return outcomes;
  }

  /** 和了者でも放銃者でもない最初の席（2 人目の和了の包の席に使う。三人打ちでは 1 人目の和了者になる）。 */
  private static int liableFor(int winner, int loser, int seats) {
    int seat = 0;
    while (seat == winner || seat == loser) {
      seat++;
    }
    return seat;
  }

  private static RoundOutcome outcome(Variant variant, Consumer<RoundOutcome> fill) {
    RoundOutcome outcome = new RoundOutcome(variant);
    fill.accept(outcome);
    return outcome;
  }

  @Test
  public void riichiMasksRotateWithTheDealer() {
    Variant variant = Variant.TENHOU_4P;
    for (int dealer = 0; dealer < 4; dealer++) {
      for (int mask = 0; mask < 16; mask++) {
        int riichi = mask;
        long code = RulesFixtures.code(variant, dealer, o -> o.setExhaustive(riichi, 0, riichi));
        int expected = 0;
        for (int seat = 0; seat < 4; seat++) {
          if ((mask & 1 << seat) != 0) {
            expected |= 1 << Seats.relative(dealer, seat, 4);
          }
        }
        Assert.assertEquals(HandResult.riichi(code), expected, Arrays.toString(new int[] {dealer, mask}));
      }
    }
  }
}

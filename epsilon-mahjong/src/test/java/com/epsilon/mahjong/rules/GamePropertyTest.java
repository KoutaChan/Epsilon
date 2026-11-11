package com.epsilon.mahjong.rules;

import static com.epsilon.mahjong.rules.RoundOutcome.NO_SEAT;

import java.util.Arrays;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 乱数で作った局の結果で半荘を最後まで進め、全ての局末で次のことを確かめる。点と供託の合計は変わらない。局は同じか 1 つ先で、進むのは
 * 親が流れたときだけ。本場は子だけの和了で 0 に戻り、ほかは 1 増える。供託は和了で 0 になり、流局では立直の分だけ増える。終局はルールの
 * とおりに決まる。
 */
public class GamePropertyTest {

  private static final int GAMES = 3_000;

  /** 1 半荘の局数の上限。乱数の結果でここに届く確率は無視できる。 */
  private static final int MAX_ROUNDS = 1_000;

  /** 終盤のルールも確かめられるように、南場の最後の局まで続く半荘と西入する半荘が十分な数あることも確かめる。 */
  @Test
  public void randomGamesConservePointsAndAdvanceLegally() {
    for (Variant variant : Variant.values()) {
      Random random = new Random(17 + variant.ordinal());
      AbortiveDraw[] draws =
          Arrays.stream(AbortiveDraw.values()).filter(variant::aborts).toArray(AbortiveDraw[]::new);
      int late = 0;
      int west = 0;
      for (int game = 0; game < GAMES; game++) {
        int last = play(variant, draws, random);
        late += last >= variant.lastRound() ? 1 : 0;
        west += last > variant.lastRound() ? 1 : 0;
      }
      Assert.assertTrue(late >= GAMES / 10, variant + ": " + late + " games reached the last round");
      Assert.assertTrue(west >= GAMES / 100, variant + ": " + west + " games went west");
    }
  }

  /** 半荘を最後まで進め、終局した局の番号を返す。 */
  private static int play(Variant variant, AbortiveDraw[] draws, Random random) {
    TableState table = new TableState(variant);
    TableState next = new TableState(variant);
    RoundOutcome outcome = new RoundOutcome(variant);
    int[] delta = new int[variant.seats()];
    for (int rounds = 0; rounds < MAX_ROUNDS; rounds++) {
      randomOutcome(table, draws, random, outcome);
      Settlement.apply(variant, table, outcome, delta);
      Progression.next(table, outcome, delta, next);
      check(table, outcome, delta, next);
      if (next.gameOver()) {
        return next.round();
      }
      table.copyFrom(next);
    }
    throw new AssertionError("No end after " + MAX_ROUNDS + " rounds: " + table);
  }

  private static void check(TableState start, RoundOutcome outcome, int[] delta, TableState next) {
    String context = start + " " + outcome + " " + Arrays.toString(delta) + " -> " + next;
    Variant variant = start.variant();
    int seats = variant.seats();
    boolean win = outcome.kind().win();
    int deposits = start.kyotaku() + Integer.bitCount(outcome.riichiMask());
    int remaining = win ? 0 : deposits;
    int moved = 0;
    int total = Variant.DEPOSIT * next.kyotaku();
    int[] settled = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      moved += delta[seat];
      total += next.score(seat);
      settled[seat] = start.score(seat) + delta[seat];
    }
    Assert.assertEquals(moved, Variant.DEPOSIT * (start.kyotaku() - remaining), context);
    Assert.assertEquals(total, seats * variant.startPoints(), context);
    Assert.assertEquals(next.gameOver(), expectedOver(start, outcome, settled), context);
    if (next.gameOver()) {
      Assert.assertEquals(next.round(), start.round(), context);
      Assert.assertEquals(next.honba(), start.honba(), context);
      Assert.assertEquals(next.kyotaku(), 0, context);
      int top = Placement.top(settled);
      for (int seat = 0; seat < seats; seat++) {
        int award = seat == top ? Variant.DEPOSIT * remaining : 0;
        Assert.assertEquals(next.score(seat), settled[seat] + award, context);
      }
      return;
    }
    DealerOutcome dealer = Progression.dealerOutcome(outcome, start.dealer());
    boolean passes = dealer == DealerOutcome.PASSES;
    Assert.assertEquals(next.round(), passes ? start.round() + 1 : start.round(), context);
    Assert.assertEquals(next.honba(), passes && win ? 0 : start.honba() + 1, context);
    Assert.assertEquals(next.kyotaku(), remaining, context);
    Assert.assertTrue(next.round() <= variant.finalRound(), context);
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(next.score(seat), settled[seat], context);
      Assert.assertTrue(next.score(seat) >= 0, context);
    }
  }

  /** 精算の後の点（和了なら供託を受け取った後、流局なら供託を場に残したまま）から、終局するかをルールどおりに求める。 */
  private static boolean expectedOver(TableState start, RoundOutcome outcome, int[] settled) {
    Variant variant = start.variant();
    for (int points : settled) {
      if (points < 0) {
        return true;
      }
    }
    if (start.round() < variant.lastRound()) {
      return false;
    }
    int top = Placement.top(settled);
    boolean reached = settled[top] >= variant.returnPoints();
    return switch (Progression.dealerOutcome(outcome, start.dealer())) {
      case PASSES -> reached || start.round() == variant.finalRound();
      case CONTINUES -> reached && top == start.dealer();
      case ABORTIVE -> false;
    };
  }

  /** 局の結果を乱数で作る。立直できるのは 1000 点以上の席だけ。和了が 6 割（ロンは時々ダブロン）、荒牌が 3 割、途中流局が 1 割。 */
  private static void randomOutcome(TableState table, AbortiveDraw[] draws, Random random, RoundOutcome outcome) {
    int seats = table.variant().seats();
    int riichiMask = 0;
    for (int seat = 0; seat < seats; seat++) {
      if (table.score(seat) >= Variant.DEPOSIT && random.nextInt(4) == 0) {
        riichiMask |= 1 << seat;
      }
    }
    int roll = random.nextInt(10);
    if (roll < 3) {
      int base = randomBase(random);
      int winner = random.nextInt(seats);
      outcome.setTsumo(base, winner, randomPao(random, base, winner, NO_SEAT, seats), riichiMask);
    } else if (roll < 6) {
      int loser = random.nextInt(seats);
      int winner = otherSeat(random, loser, seats);
      int base = randomBase(random);
      outcome.setRon(base, winner, loser, randomPao(random, base, winner, loser, seats), riichiMask);
      int second = otherSeat(random, loser, seats);
      if (second != winner && random.nextInt(5) == 0) {
        int secondBase = randomBase(random);
        outcome.addRonWinner(secondBase, second, randomPao(random, secondBase, second, loser, seats));
      }
    } else if (roll < 9) {
      int nagashiMask = random.nextInt(10) == 0 ? 1 << random.nextInt(seats) : 0;
      outcome.setExhaustive(random.nextInt(1 << seats), nagashiMask, riichiMask);
    } else {
      outcome.setAbortive(draws[random.nextInt(draws.length)], riichiMask);
    }
  }

  /** 基本点を乱数で選ぶ。4 回に 3 回は満貫以下から、残りは表の全ての値から選ぶ。 */
  private static int randomBase(Random random) {
    BaseClass[] bases = BaseClass.values();
    int bound = random.nextInt(4) == 0 ? bases.length : BaseClass.MANGAN.ordinal() + 1;
    return bases[random.nextInt(bound)].base();
  }

  /**
   * 役満なら 3 回に 1 回、和了者でも放銃者でもない席を包の席にする。
   *
   * @param loser ロンの放銃者。ツモなら {@code NO_SEAT}
   */
  private static int randomPao(Random random, int base, int winner, int loser, int seats) {
    if (base % BaseClass.YAKUMAN.base() != 0 || random.nextInt(3) != 0) {
      return NO_SEAT;
    }
    int pao = otherSeat(random, winner, seats);
    while (pao == loser) {
      pao = otherSeat(random, winner, seats);
    }
    return pao;
  }

  private static int otherSeat(Random random, int seat, int seats) {
    return (seat + 1 + random.nextInt(seats - 1)) % seats;
  }
}

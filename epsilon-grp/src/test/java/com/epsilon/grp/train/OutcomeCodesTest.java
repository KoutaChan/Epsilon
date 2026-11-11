package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeSection;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.grp.RiichiRole;
import com.epsilon.grp.Seats;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Consumer;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 局の結果の生の形のコードと、それを局の結果のネットワーク q の項に直す OutcomeCodes を確かめる。生の形は和了者の立直も残し、
 * 精算まで元の結果と同じになる。q の表のどの結果も、コードにして戻せば元に戻る。包、流し満貫、
 * 聴牌でない席の立直は表せない（0 を返す）。エンジンで打った局の生のコードは、遷移の探索（TransitionSearch）が再現する結果の 1 つに
 * 当たる。数のファイルから作る q の行は、数え上げる局ごとに結果を 1 つ持つ。半荘の分け方と行の並びは、半荘の記録から作る行と同じ。
 */
public class OutcomeCodesTest {

  private static final int NONE = HandOutcomes.NONE;

  /**
   * ツモ・ロン・ダブロン・包・流し満貫・荒牌・途中流局を、和了者の立直も入れて乱数で作る。生の形のコードは、結果に置き直してから
   * もう一度コードにしても同じになる。立直は全て残り、どの卓から精算しても、点数・局・本場・供託・終局が元の結果と同じになる。
   */
  @Test
  public void rawCodesKeepEveryRiichi() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      Random random = new Random(3 + seats);
      RoundOutcome outcome = new RoundOutcome(variant);
      RoundOutcome placed = new RoundOutcome(variant);
      RoundOutcome scratch = new RoundOutcome(variant);
      TableState start = new TableState(variant);
      TableState expected = new TableState(variant);
      TableState actual = new TableState(variant);
      int[] delta = new int[seats];
      int[] placedDelta = new int[seats];
      Map<EndKind, Integer> kinds = new EnumMap<>(EndKind.class);
      int winnerRiichi = 0;
      for (int trial = 0; trial < 20_000; trial++) {
        GrpFixtures.state(variant, random).copyInto(start);
        int riichi = 0;
        for (int seat = 0; seat < seats; seat++) {
          if (start.score(seat) >= Variant.DEPOSIT && random.nextInt(3) == 0) {
            riichi |= 1 << seat;
          }
        }
        fill(outcome, random, riichi);
        int dealer = start.dealer();
        long raw = HandResult.raw(outcome, dealer);
        String context = outcome + " with dealer " + dealer;
        Assert.assertTrue(HandResult.valid(raw, scratch), context);
        HandResult.place(raw, dealer, placed);
        Assert.assertEquals(HandResult.raw(placed, dealer), raw, context);
        Assert.assertEquals(placed.riichiMask(), outcome.riichiMask(), context);
        Assert.assertEquals(placed.kind(), outcome.kind(), context);
        Settlement.apply(variant, start, outcome, delta);
        Settlement.apply(variant, start, placed, placedDelta);
        Assert.assertEquals(placedDelta, delta, context);
        Progression.next(start, outcome, delta, expected);
        Progression.next(start, placed, placedDelta, actual);
        Assert.assertEquals(actual.toString(), expected.toString(), context);
        if (outcome.kind().win() && (outcome.riichiMask() & 1 << outcome.winner(0)) != 0) {
          winnerRiichi++;
        }
        kinds.merge(outcome.kind(), 1, Integer::sum);
      }
      Assert.assertEquals(kinds.size(), EndKind.values().length, variant + " " + kinds);
      Assert.assertTrue(winnerRiichi > 1_000, variant + ": " + winnerRiichi);
    }
  }

  /**
   * q の表の全ての結果（終わり方 × 打点 × 立直。立直は、その有無を確率に入れる席のもの）を生の形のコードにして項に直すと、表の
   * 結果と同じ項を同じ順に書く。コードを結果 1 つに直すと、表の終わり方・打点・2 人目の打点・立直した席（親から見た相対席）に戻る。
   * 親はどの席でもよい。包のある和了、流し満貫、荒牌で聴牌でない席の立直は、0 を返して何も書かず、結果にも直せない。
   */
  @Test
  public void everyTableOutcomeRoundTripsThroughTheCodes() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      HandOutcomes hands = HandOutcomes.of(variant);
      OutcomeCodes codes = new OutcomeCodes(variant);
      Assert.assertEquals(codes.variant(), variant);
      RoundOutcome outcome = new RoundOutcome(variant);
      short[] expected = new short[hands.maxTerms()];
      short[] actual = new short[hands.maxTerms() + 2];
      long checked = 0;
      for (int main = 0; main < hands.mainCount(); main++) {
        OutcomeSection section = hands.section(main);
        int bases = section == OutcomeSection.TSUMO || section == OutcomeSection.RON
            || section == OutcomeSection.DOUBLE_RON ? HandOutcomes.BASES : 1;
        int others = section == OutcomeSection.DOUBLE_RON ? HandOutcomes.BASES : 1;
        int factors = hands.factorCount(main);
        for (int dealer = 0; dealer < seats; dealer++) {
          for (int base = 0; base < bases; base++) {
            for (int other = 0; other < others; other++) {
              for (int mask = 0; mask < 1 << factors; mask++) {
                int relative = 0;
                int absolute = 0;
                for (int index = 0; index < factors; index++) {
                  if ((mask & 1 << index) != 0) {
                    int seat = hands.factorSeat(main, index);
                    relative |= 1 << seat;
                    absolute |= 1 << Seats.absolute(dealer, seat, seats);
                  }
                }
                int placedBase = bases == 1 ? NONE : base;
                int placedOther = others == 1 ? NONE : other;
                hands.place(main, placedBase, placedOther, absolute, dealer, outcome);
                long code = HandResult.raw(outcome, dealer);
                int count = OutcomeRows.outcomeTerms(hands, main, placedBase, placedOther, relative, expected, 0);
                Arrays.fill(actual, (short) 999);
                Assert.assertEquals(codes.terms(code, actual, 2), count, outcome.toString());
                Assert.assertEquals(
                    Arrays.copyOfRange(actual, 2, 2 + count), Arrays.copyOf(expected, count), outcome.toString());
                Assert.assertTrue(codes.decode(code), outcome.toString());
                Assert.assertEquals(codes.main(), main, outcome.toString());
                Assert.assertEquals(codes.base(), placedBase, outcome.toString());
                Assert.assertEquals(codes.other(), placedOther, outcome.toString());
                Assert.assertEquals(codes.riichi(), relative, outcome.toString());
                checked++;
              }
            }
          }
        }
      }
      Assert.assertTrue(checked > 50_000, variant + " " + checked);

      for (RoundOutcome unexplained : List.of(
          outcome(variant, o -> o.setTsumo(BaseClass.YAKUMAN.base(), 1, 0, 0)),
          outcome(variant, o -> o.setRon(BaseClass.YAKUMAN.base(), 1, 2, 0, 0)),
          outcome(variant, o -> {
            o.setRon(BaseClass.MANGAN.base(), 1, 0, RoundOutcome.NO_SEAT, 0);
            o.addRonWinner(BaseClass.YAKUMAN.base(), 2, 1);
          }),
          outcome(variant, o -> o.setExhaustive(0, 0b10, 0)),
          outcome(variant, o -> o.setExhaustive(0b1, 0b1, 0)),
          outcome(variant, o -> o.setExhaustive(0b1, 0, 0b11)))) {
        long code = HandResult.raw(unexplained, 0);
        Assert.assertTrue(HandResult.valid(code, new RoundOutcome(variant)), unexplained.toString());
        Arrays.fill(actual, (short) 999);
        Assert.assertEquals(codes.terms(code, actual, 0), 0, unexplained.toString());
        Assert.assertFalse(codes.decode(code), unexplained.toString());
        for (short term : actual) {
          Assert.assertEquals(term, 999, unexplained.toString());
        }
      }
    }
  }

  /**
   * エンジンで打った局ごとに、生のコードは結果に置き直せる。項に直せるなら、遷移の探索が再現する結果の 1 つと同じ項になる。直せない
   * のは包か流し満貫の局だけで、探索もその局では表せる結果の外にしか遷移を見つけない。和了者が立直していた和了も出て、そのときは
   * 和了者の立直を含む結果に当たる。局の列を数のファイルにすると、局の進みを再現できるかの確かめも通る。
   */
  @Test
  public void engineHandsMatchTheSearchedOutcomes() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      HandOutcomes hands = HandOutcomes.of(variant);
      OutcomeCodes codes = new OutcomeCodes(variant);
      TransitionSearch search = new TransitionSearch(variant);
      RoundOutcome scratch = new RoundOutcome(variant);
      short[] terms = new short[hands.maxTerms()];
      short[] found = new short[hands.maxTerms()];
      EngineGames.Played played = EngineGames.play(variant, 60, 1, 11 + seats);
      CountedHands.Builder counts = CountedHands.builder(variant, 1, 0, GrpFixtures.STAMP);
      Map<EndKind, Integer> kinds = new EnumMap<>(EndKind.class);
      int winnerRiichi = 0;
      int unexplained = 0;
      for (EngineGames.Hand hand : played.hands()) {
        TableState start = hand.start();
        int dealer = start.dealer();
        long code = HandResult.raw(hand.outcome(), dealer);
        String context = hand.gameId() + " " + start + " " + hand.outcome();
        Assert.assertTrue(HandResult.valid(code, scratch), context);
        counts.add(hand.gameId(), start, code);
        search.from(GrpState.of(start));
        int count = hand.next().gameOver() ? search.last(scores(hand.next())) : search.next(GrpState.of(hand.next()));
        int written = codes.terms(code, terms, 0);
        kinds.merge(hand.outcome().kind(), 1, Integer::sum);
        if (written == 0) {
          Assert.assertEquals(count, 0, context);
          Assert.assertTrue(search.extras(), context);
          unexplained++;
          continue;
        }
        Assert.assertTrue(count > 0, context);
        boolean matched = false;
        for (int index = 0; index < count && !matched; index++) {
          int length = OutcomeRows.outcomeTerms(
              hands, search.main(index), search.base(index), search.other(index), search.riichi(index), found, 0);
          matched = length == written && Arrays.equals(terms, 0, written, found, 0, length);
        }
        Assert.assertTrue(matched, context);
        if (hand.outcome().kind().win()
            && (hand.outcome().riichiMask() & 1 << hand.outcome().winner(0)) != 0) {
          winnerRiichi++;
        }
      }
      for (EndKind kind : List.of(EndKind.TSUMO, EndKind.RON, EndKind.EXHAUSTIVE)) {
        Assert.assertTrue(kinds.getOrDefault(kind, 0) > 0, variant + " " + kinds);
      }
      Assert.assertTrue(winnerRiichi > 0, variant + " " + kinds);
      Assert.assertTrue(unexplained * 50 < played.hands().size(), variant + " " + unexplained);
      Assert.assertEquals(counts.build().firstMismatch(), -1, variant.name());
    }
  }

  /**
   * 同じ半荘の数のファイルと記録から、数え上げる局の q の行を作る。数のファイルから作る行は、全て 1 局に結果が 1 つで、数え上げる
   * 卓のもの。行のキーと並び、学習用と早期終了用の検証への分け方、表せない局の数、キーに収まらない局の数は、記録から作る行と同じ
   * になる（どちらも、表せない局は包か流し満貫だけ）。記録から作る行には、結果を 1 つに決められない行がある。NLL は、数のファイル
   * から作る行（記録した結果だけ）の方が小さくならない。
   */
  @Test
  public void countFileRowsAreOneResultPerCoveredHand() {
    for (Variant variant : Variant.values()) {
      EngineGames.Played played = EngineGames.play(variant, 200, 500, 23 + variant.seats());
      CountedHands.Builder builder = CountedHands.builder(variant, 4, 0, GrpFixtures.STAMP);
      for (EngineGames.Hand hand : played.hands()) {
        builder.add(hand.gameId(), hand.start(), HandResult.raw(hand.outcome(), hand.start().dealer()));
      }
      CountedHands counted = builder.build();
      RecencyWindow window = new RecencyWindow(4, 1, 64);
      OutcomeRows.Split fromCounts =
          OutcomeRows.builder(variant, window, GateCohort.NEXT, FinalRound::covers).add(4, counted).build();
      OutcomeRows.Split fromRecords =
          OutcomeRows.builder(variant, window, GateCohort.NEXT, FinalRound::covers).add(4, played.games()).build();
      int ambiguous = 0;
      OutcomeShape shape = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
      OutcomeNetwork network = GrpFixtures.outcomes(shape, 29, 0.5);
      OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
      for (int part = 0; part < 2; part++) {
        OutcomeRows counts = part == 0 ? fromCounts.train() : fromCounts.stop();
        OutcomeRows records = part == 0 ? fromRecords.train() : fromRecords.stop();
        String context = variant + " part " + part + " " + counts + " " + records;
        Assert.assertTrue(counts.size() > 0, context);
        Assert.assertEquals(counts.size(), records.size(), context);
        Assert.assertEquals(counts.unexplained(), records.unexplained(), context);
        Assert.assertEquals(counts.excluded(), records.excluded(), context);
        Assert.assertEquals(counts.maxOutcomes(), 1, context);
        for (int row = 0; row < counts.size(); row++) {
          Assert.assertEquals(counts.key(row), records.key(row), context + " row " + row);
          Assert.assertEquals(counts.cohort(row), records.cohort(row), context);
          Assert.assertEquals(counts.weight(row), records.weight(row), context);
          Assert.assertEquals(counts.outcomes(row), 1, context);
          Assert.assertTrue(FinalRound.covers(GrpState.ofKey(variant, counts.key(row))), context);
          if (records.outcomes(row) > 1) {
            ambiguous++;
          }
        }
        Assert.assertTrue(
            counts.nll(network, workspace) >= records.nll(network, workspace) - 1e-12, context);
      }
      Assert.assertTrue(ambiguous > 0, variant + ": every recorded row had one outcome");
    }
  }

  /**
   * 南 4 で立直した子がツモ和了した局だけの半荘を、数のファイルから q の行にする。行の結果は生の形（和了者の立直あり）の 1 つ。
   * 和了者の立直に寄せた q の NLL は、その結果の項から求めた値と同じで、和了者の立直を消した結果の値とは違う。
   */
  @Test
  public void countFileRowsKeepTheWinnersRiichi() {
    Variant variant = Variant.TENHOU_4P;
    TableState south4 = new TableState(variant);
    south4.set(7, 0, 0, new int[] {26_000, 25_000, 25_000, 24_000});
    RoundOutcome tsumo = new RoundOutcome(variant);
    tsumo.setTsumo(BaseClass.MANGAN.base(), 1, RoundOutcome.NO_SEAT, 0b0011);
    long raw = HandResult.raw(tsumo, south4.dealer());
    RoundOutcome withoutWinnersRiichi = new RoundOutcome(variant);
    withoutWinnersRiichi.setTsumo(BaseClass.MANGAN.base(), 1, RoundOutcome.NO_SEAT, 0b0001);
    CountedHands.Builder builder = CountedHands.builder(variant, 4, 0, GrpFixtures.STAMP);
    for (long gameId = 1; gameId <= 40; gameId++) {
      builder.add(gameId, south4, raw);
    }
    RecencyWindow window = new RecencyWindow(4, 1, 64);
    OutcomeRows.Split rows =
        OutcomeRows.builder(variant, window, GateCohort.NEXT, FinalRound::covers).add(4, builder.build()).build();
    Assert.assertEquals(rows.train().size() + rows.stop().size(), 40);

    HandOutcomes hands = HandOutcomes.of(variant);
    double[] logits = new double[hands.logits()];
    for (int seat = 0; seat < 4; seat++) {
      logits[hands.riichiTerm(seat, RiichiRole.WINNER, true)] = 3;
    }
    OutcomeShape shape = new OutcomeShape(variant, GrpFixtures.plainFields(), 4);
    OutcomeNetwork network = GrpFixtures.biased(shape, logits);
    OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
    network.forward(GrpState.of(south4), workspace);
    OutcomeCodes codes = new OutcomeCodes(variant);
    double expected = nll(codes, raw, workspace);
    double dropped = nll(codes, HandResult.raw(withoutWinnersRiichi, south4.dealer()), workspace);
    Assert.assertTrue(dropped > expected + 1, expected + " " + dropped);
    for (OutcomeRows part : List.of(rows.train(), rows.stop())) {
      if (part.size() > 0) {
        Assert.assertEquals(part.nll(network, workspace), expected, 1e-12, part.toString());
      }
    }
  }

  /** コードが表す結果 1 つの −log q を返す。 */
  private static double nll(OutcomeCodes codes, long code, OutcomeWorkspace workspace) {
    short[] terms = new short[16];
    int count = codes.terms(code, terms, 0);
    double sum = 0;
    for (int index = 0; index < count; index++) {
      sum -= workspace.term(terms[index]);
    }
    return sum;
  }

  /**
   * 結果を 1 つ乱数で作る。ツモ・ロン・ダブロン・包のある役満・途中流局はそれぞれ 1/7、荒牌は 2/7 の確率で選ぶ。包は役満の和了にだけ
   * 付け、流し満貫は荒牌の 1/10 に付ける。
   */
  private static void fill(RoundOutcome outcome, Random random, int riichi) {
    int seats = outcome.variant().seats();
    BaseClass[] bases = BaseClass.values();
    int winner = random.nextInt(seats);
    int loser = Seats.absolute(winner, 1 + random.nextInt(seats - 1), seats);
    int base = bases[random.nextInt(bases.length)].base();
    int kind = random.nextInt(7);
    switch (kind) {
      case 0 -> outcome.setTsumo(base, winner, RoundOutcome.NO_SEAT, riichi);
      case 1 -> outcome.setRon(base, winner, loser, RoundOutcome.NO_SEAT, riichi);
      case 2 -> {
        outcome.setRon(base, winner, loser, RoundOutcome.NO_SEAT, riichi);
        int other = Seats.absolute(loser, 1 + random.nextInt(seats - 1), seats);
        if (other != winner) {
          outcome.addRonWinner(bases[random.nextInt(bases.length)].base(), other, RoundOutcome.NO_SEAT);
        }
      }
      case 3 -> {
        int pao = Seats.absolute(winner, 1 + random.nextInt(seats - 1), seats);
        int yakuman = BaseClass.YAKUMAN.base() * (1 + random.nextInt(2));
        if (pao != loser && random.nextBoolean()) {
          outcome.setRon(yakuman, winner, loser, pao, riichi);
        } else {
          outcome.setTsumo(yakuman, winner, pao, riichi);
        }
      }
      case 4, 5 -> {
        int tenpai = riichi;
        for (int seat = 0; seat < seats; seat++) {
          if (random.nextInt(5) < 2) {
            tenpai |= 1 << seat;
          }
        }
        int nagashi = random.nextInt(10) == 0 ? 1 << random.nextInt(seats) : 0;
        outcome.setExhaustive(tenpai, nagashi, riichi);
      }
      default -> outcome.setAbortive(AbortiveDraw.NINE_TERMINALS, riichi);
    }
  }

  private static RoundOutcome outcome(Variant variant, Consumer<RoundOutcome> fill) {
    RoundOutcome outcome = new RoundOutcome(variant);
    fill.accept(outcome);
    return outcome;
  }

  private static int[] scores(TableState table) {
    int[] scores = new int[table.variant().seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = table.score(seat);
    }
    return scores;
  }
}

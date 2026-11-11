package com.epsilon.grp.train;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeSection;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 次の局の実際の結果を、遷移を再現できた結果の種類で分ける処理を確かめる。ルールどおりに打った半荘の局末は、全て局の結果の
 * ネットワークが表せる結果になる。包、流し満貫、記録を作ったエンジンの終局の判定、100 点ずらした遷移は、それぞれ別の種類に分かれる。
 * 記録を作ったエンジンの判定で再現する遷移は、その判定で再現できる結果を全て持つ。選んだ局末の列でも種類は変わらない。
 * 今の順位のままで終わる率の誤差は、表せる局末の分と表せない局末の分に分かれる。この分け方は、比べる 2 つの値のそれぞれでも、その差
 * でも成り立つ。差の分け方は、2 つの値の分け方の差になる。局の結果の分類は、記録を作ったエンジンの判定で再現する局末も測る。席ごとに
 * 見ても親から見ても、全ての結果を 1 度ずつ数える。数のファイルから作った遷移は記録した結果を 1 つ持ち、終局かどうかは遷移の探索と
 * 同じになる。
 */
public class NextHandsTest {

  /** 最初の局末に特別な遷移を持つ半荘の数。どの半荘も局末は 2 つで、2 つ目の遷移はルールどおり。 */
  private static final int SPECIAL = 4;

  @Test
  public void transitionsAreSortedByKind() {
    List<GameRecord> games = games();
    HandEnds ends = HandEnds.of(OutcomeFixtures.FOUR, RankUtility.TENHOU, games);
    NextHands next = NextHands.of(ends, games);
    NextHands.Kind[] expected = {
      NextHands.Kind.ENDING, NextHands.Kind.PAO, NextHands.Kind.NAGASHI, NextHands.Kind.MISMATCH
    };
    for (int game = 0; game < SPECIAL; game++) {
      int first = ends.firstHand(game);
      Assert.assertEquals(next.kind(first), expected[game], "game " + game);
      if (expected[game] != NextHands.Kind.ENDING) {
        Assert.assertEquals(next.found(first), 0);
      }
      Assert.assertEquals(next.kind(first + 1), NextHands.Kind.OUTCOME, "game " + game);
      Assert.assertTrue(next.gameOver(first + 1) && !next.gameOver(first));
    }
    for (int h = ends.firstHand(SPECIAL); h < ends.handEnds(); h++) {
      Assert.assertEquals(next.kind(h), NextHands.Kind.OUTCOME, "hand end " + h);
      Assert.assertTrue(next.found(h) > 0);
    }
    // 南 4 の親は席 3 なので、席 0 は相対席 1、席 2 は相対席 3。供託を受け取る前の点数は、和了者が立直していてもいなくても 30,000 点に
    // 届かない。そのため、和了者の立直のあり・なしの 2 つが残る。
    HandOutcomes hands = HandOutcomes.of(OutcomeFixtures.FOUR);
    int ending = ends.firstHand(0);
    Assert.assertEquals(next.found(ending), 2);
    Set<Integer> riichi = new HashSet<>();
    for (int k = 0; k < next.found(ending); k++) {
      int main = next.main(ending, k);
      Assert.assertEquals(hands.section(main), OutcomeSection.RON);
      Assert.assertEquals(hands.winner(main), 1);
      Assert.assertEquals(hands.loser(main), 3);
      Assert.assertEquals(next.base(ending, k), BaseClass.B240.ordinal());
      riichi.add(next.riichi(ending, k));
    }
    Assert.assertEquals(riichi, Set.of(0, 0b0010));
    List<NextHands.KindCount> counts = next.kinds(h -> true);
    for (NextHands.Kind kind : expected) {
      Assert.assertEquals(counts.get(kind.ordinal()).handEnds(), 1, kind.toString());
    }
    Assert.assertEquals(counts.get(NextHands.Kind.OUTCOME.ordinal()).handEnds(), ends.handEnds() - SPECIAL);

    HandEnds odd = ends.select(h -> h % 2 == 1);
    NextHands selected = NextHands.of(odd, games);
    for (int index = 0; index < odd.handEnds(); index++) {
      int h = 2 * index + 1;
      Assert.assertEquals(selected.kind(index), next.kind(h));
      Assert.assertEquals(selected.found(index), next.found(h));
      for (int k = 0; k < next.found(h); k++) {
        Assert.assertEquals(selected.main(index, k), next.main(h, k));
        Assert.assertEquals(selected.base(index, k), next.base(h, k));
        Assert.assertEquals(selected.riichi(index, k), next.riichi(h, k));
      }
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> NextHands.of(ends, games.subList(1, games.size())));
  }

  @Test
  public void keepErrorsAndOutcomeClassesAddUp() {
    List<GameRecord> games = games();
    HandEnds ends = HandEnds.of(OutcomeFixtures.FOUR, RankUtility.TENHOU, games);
    NextHands next = NextHands.of(ends, games);
    GrpShape leafShape = GrpFixtures.shape(OutcomeFixtures.FOUR, 8, false);
    GrpNetwork leaf = GrpFixtures.network(leafShape, 13, 0.5);
    GrpNetwork other = GrpFixtures.network(leafShape, 17, 0.5);
    HandEndValues values = HandEndValues.of(ends, leaf, GrpWorkspace.of(leafShape));
    HandEndValues reference = HandEndValues.of(ends, other, GrpWorkspace.of(leafShape));
    NextHands.KeepSplit split = next.keep(values, reference, h -> true, (group, rank, gap) -> true);
    Assert.assertEquals(split.seatHands(), ends.seatHands());
    Assert.assertEquals(split.missingShare(), (double) SPECIAL / ends.handEnds(), 1e-15);
    for (NextHands.Split part : List.of(split.candidate(), split.reference(), split.difference())) {
      Assert.assertEquals(part.explained().mean(), part.all().mean() - part.represented().mean(), 1e-15);
      Assert.assertEquals(
          part.explained().mean(), split.missingShare() * (part.missing().mean() - part.represented().mean()), 1e-12);
      Assert.assertTrue(part.explained().standardError() > 0);
    }
    NextHands.Split candidate = split.candidate();
    NextHands.Split difference = split.difference();
    Assert.assertEquals(difference.all().mean(), candidate.all().mean() - split.reference().all().mean(), 1e-12);
    Assert.assertEquals(
        difference.missing().mean(), candidate.missing().mean() - split.reference().missing().mean(), 1e-12);
    Assert.assertEquals(
        difference.explained().mean(), candidate.explained().mean() - split.reference().explained().mean(), 1e-12);
    NextHands.KeepSplit same = next.keep(values, values, h -> true, (group, rank, gap) -> true);
    Assert.assertEquals(same.difference().all().mean(), 0.0);
    Assert.assertEquals(same.difference().explained().mean(), 0.0);

    OutcomeShape shape = new OutcomeShape(OutcomeFixtures.FOUR, GrpFixtures.plainFields(), 8);
    OutcomeNetwork network = GrpFixtures.outcomes(shape, 11, 0.5);
    OutcomeClasses.Result classes =
        OutcomeClasses.measure(next, h -> network, h -> true, () -> OutcomeWorkspace.of(shape));
    Assert.assertEquals(classes.ending(), 1);
    Assert.assertEquals(classes.skipped(), SPECIAL - 1);
    Assert.assertEquals(classes.handEnds(), ends.handEnds() - (SPECIAL - 1));
    long pooledSeats = 0;
    long pooledEnding = 0;
    long pooledSkipped = 0;
    for (OutcomeClasses.SeatCell cell : classes.seats()) {
      double predicted = 0;
      double realized = 0;
      for (int kind = 0; kind <= OutcomeClasses.SeatClass.DRAW.ordinal(); kind++) {
        predicted += cell.rates().get(kind).predicted();
        realized += cell.rates().get(kind).realized();
      }
      Assert.assertEquals(predicted, 1, 1e-9, cell.toString());
      Assert.assertEquals(realized, 1, 1e-9, cell.toString());
      double wins = cell.rates().get(OutcomeClasses.SeatClass.TSUMO.ordinal()).realized()
          + cell.rates().get(OutcomeClasses.SeatClass.RON.ordinal()).realized();
      Assert.assertTrue(cell.rates().get(OutcomeClasses.SeatClass.BIG_WIN.ordinal()).realized() <= wins + 1e-12);
      if (cell.group() == -1) {
        pooledSeats += cell.seatHands();
        pooledEnding += cell.ending();
        pooledSkipped += cell.skipped();
      }
    }
    Assert.assertEquals(pooledSeats, 4 * classes.handEnds());
    Assert.assertEquals(pooledEnding, 4 * classes.ending());
    Assert.assertTrue(pooledSkipped <= 4 * classes.skipped());
    for (OutcomeClasses.DealerCell cell : classes.dealers()) {
      double predicted = 0;
      double realized = 0;
      for (OutcomeClasses.Rate rate : cell.rates()) {
        predicted += rate.predicted();
        realized += rate.realized();
      }
      Assert.assertEquals(predicted, 1, 1e-9, cell.toString());
      Assert.assertEquals(realized, 1, 1e-9, cell.toString());
    }
  }

  /**
   * エンジンで打った半荘の数のファイルから遷移を作る。次の局で半荘が終わるかどうかと、次の局の後に順位が変わるかどうかは、半荘の
   * 記録から遷移を探索して求めたものと同じになる。表せる局末は記録した結果だけを持ち、それは探索が再現する結果の 1 つ。表せない局末は結果を持たず、包のある
   * 和了なら {@code PAO}、流し満貫なら {@code NAGASHI} になる。選んだ局末の列でも同じ。数のファイルが局末と合わなければ（ルール・
   * 半荘・局の数・局の始まりの卓）、例外を投げる。
   */
  @Test
  public void countFileTransitionsMatchTheSearch() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      HandOutcomes hands = HandOutcomes.of(variant);
      OutcomeCodes codes = new OutcomeCodes(variant);
      EngineGames.Played played = EngineGames.play(variant, 120, 1, 31 + seats);
      Map<Long, List<EngineGames.Hand>> byGame = new HashMap<>();
      for (EngineGames.Hand hand : played.hands()) {
        byGame.computeIfAbsent(hand.gameId(), id -> new ArrayList<>()).add(hand);
      }
      CountedHands counted = EngineGames.counted(variant, played.hands());
      HandEnds ends = HandEnds.of(variant, RankUtility.TENHOU, played.games());
      NextHands recorded = NextHands.of(ends, counted);
      NextHands searched = NextHands.of(ends, played.games());
      short[] terms = new short[hands.maxTerms()];
      short[] found = new short[hands.maxTerms()];
      int representable = 0;
      int rankChanges = 0;
      for (int h = 0; h < ends.handEnds(); h++) {
        EngineGames.Hand hand = byGame.get(ends.gameId(ends.game(h))).get(h - ends.firstHand(ends.game(h)) + 1);
        String context = variant + " " + hand;
        Assert.assertEquals(recorded.gameOver(h), searched.gameOver(h), context);
        Assert.assertEquals(recorded.gameOver(h), hand.next().gameOver(), context);
        double changed = nextChanged(recorded, h);
        Assert.assertEquals(changed, nextChanged(searched, h), context);
        rankChanges += (int) changed;
        int written = codes.terms(HandResult.raw(hand.outcome(), hand.start().dealer()), terms, 0);
        if (written == 0) {
          Assert.assertEquals(recorded.found(h), 0, context);
          Assert.assertEquals(recorded.kind(h), unrepresented(hand.outcome()), context);
          continue;
        }
        representable++;
        Assert.assertEquals(recorded.kind(h), NextHands.Kind.OUTCOME, context);
        Assert.assertEquals(recorded.found(h), 1, context);
        int length = OutcomeRows.outcomeTerms(
            hands, recorded.main(h, 0), recorded.base(h, 0), recorded.other(h, 0), recorded.riichi(h, 0), found, 0);
        Assert.assertTrue(Arrays.equals(terms, 0, written, found, 0, length), context);
        boolean matched = false;
        for (int k = 0; k < searched.found(h) && !matched; k++) {
          length = OutcomeRows.outcomeTerms(
              hands, searched.main(h, k), searched.base(h, k), searched.other(h, k), searched.riichi(h, k), found, 0);
          matched = Arrays.equals(terms, 0, written, found, 0, length);
        }
        Assert.assertTrue(matched, context);
      }
      Assert.assertTrue(representable > 500, variant + " " + representable);
      Assert.assertTrue(rankChanges > 0 && rankChanges < ends.handEnds(), variant + " " + rankChanges);

      HandEnds odd = ends.select(h -> h % 2 == 1);
      NextHands selected = NextHands.of(odd, counted);
      for (int index = 0; index < odd.handEnds(); index++) {
        int h = 2 * index + 1;
        Assert.assertEquals(selected.kind(index), recorded.kind(h));
        Assert.assertEquals(selected.found(index), recorded.found(h));
        Assert.assertEquals(selected.gameOver(index), recorded.gameOver(h));
        for (int k = 0; k < recorded.found(h); k++) {
          Assert.assertEquals(selected.main(index, k), recorded.main(h, k));
          Assert.assertEquals(selected.base(index, k), recorded.base(h, k));
          Assert.assertEquals(selected.other(index, k), recorded.other(h, k));
          Assert.assertEquals(selected.riichi(index, k), recorded.riichi(h, k));
        }
      }

      Variant other = variant == Variant.TENHOU_4P ? Variant.TENHOU_3P : Variant.TENHOU_4P;
      Assert.assertThrows(
          IllegalArgumentException.class, () -> NextHands.of(ends, CountedHands.builder(other, 4, 0, GrpFixtures.STAMP).build()));
      long gameId = -1;
      for (GameRecord game : played.games()) {
        if (game.handEnds().size() >= 2) {
          gameId = game.gameId();
          break;
        }
      }
      List<EngineGames.Hand> chosen = byGame.get(gameId);
      List<EngineGames.Hand> missing = new ArrayList<>(played.hands());
      missing.removeAll(chosen);
      List<EngineGames.Hand> fewer = new ArrayList<>(played.hands());
      fewer.remove(chosen.get(chosen.size() - 1));
      List<EngineGames.Hand> shifted = new ArrayList<>(played.hands());
      EngineGames.Hand second = chosen.get(1);
      shifted.set(shifted.indexOf(second), new EngineGames.Hand(
          gameId, shiftedStart(second.start()), second.outcome(), second.next()));
      for (List<EngineGames.Hand> broken : List.of(missing, fewer, shifted)) {
        CountedHands wrong = EngineGames.counted(variant, broken);
        Assert.assertThrows(IllegalArgumentException.class, () -> NextHands.of(ends, wrong));
      }
    }
  }

  /** その局末だけを数えたときの、次の局の後の順位が今の順位と違う率（0 か 1）。 */
  private static double nextChanged(NextHands next, int h) {
    return next.kinds(other -> other == h).get(next.kind(h).ordinal()).nextChanged().mean();
  }

  /** 表せない結果の種類を返す。包のある和了なら PAO、流し満貫なら NAGASHI、ほかは MISMATCH。 */
  private static NextHands.Kind unrepresented(RoundOutcome outcome) {
    if (outcome.kind() == EndKind.EXHAUSTIVE) {
      return outcome.nagashiMask() != 0 ? NextHands.Kind.NAGASHI : NextHands.Kind.MISMATCH;
    }
    for (int index = 0; index < outcome.winnerCount(); index++) {
      if (outcome.pao(index) != RoundOutcome.NO_SEAT) {
        return NextHands.Kind.PAO;
      }
    }
    return NextHands.Kind.MISMATCH;
  }

  /** 席 0 から席 1 へ 100 点動かした卓を返す。点数と供託の和は変わらない。 */
  private static TableState shiftedStart(TableState start) {
    int[] scores = new int[start.variant().seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = start.score(seat);
    }
    scores[0] -= 100;
    scores[1] += 100;
    TableState shifted = new TableState(start.variant());
    shifted.set(start.round(), start.honba(), start.kyotaku(), scores);
    return shifted;
  }

  /**
   * 最初の局末からの遷移が特別な 4 つの半荘と、ルールどおりに打った半荘を作る。
   *
   * <ol>
   *   <li>南 4（親は席 3、供託 1 本）で、席 0 が席 2 から 1,000 点をロンし、供託を受け取ると 30,500 点になる。Progression では
   *       半荘が終わるが、供託を受け取る前の 29,500 点で判定すると西 1 へ続く。
   *   <li>東 2 で、席 2 が席 3 から役満をロンする。包の席 0 が半分を払う。
   *   <li>東 2 で、子の席 3 が流し満貫になる（親は不聴で、本場が増える）。
   *   <li>東 2 から、100 点ずらした東 3 へ。
   * </ol>
   */
  private static List<GameRecord> games() {
    List<GameRecord> games = new ArrayList<>();
    GrpState south4 = new GrpState(OutcomeFixtures.FOUR, 7, 0, 1, new int[] {28_500, 26_500, 23_000, 21_000});
    RoundOutcome ron = new RoundOutcome(OutcomeFixtures.FOUR);
    ron.setRon(BaseClass.B240.base(), 0, 2, RoundOutcome.NO_SEAT, 0);
    int[] delta = new int[4];
    Settlement.apply(OutcomeFixtures.FOUR, table(south4), ron, delta);
    int[] scores = south4.scores();
    for (int seat = 0; seat < 4; seat++) {
      scores[seat] += delta[seat];
    }
    GrpState west1 = new GrpState(OutcomeFixtures.FOUR, 8, 0, 0, scores);
    RoundOutcome tsumo = new RoundOutcome(OutcomeFixtures.FOUR);
    tsumo.setTsumo(BaseClass.MANGAN.base(), 1, RoundOutcome.NO_SEAT, 0);
    games.add(new GameRecord(OutcomeFixtures.FOUR, 1, List.of(south4, west1), finalScores(west1, tsumo)));

    GrpState east2 = new GrpState(OutcomeFixtures.FOUR, 1, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    RoundOutcome pao = new RoundOutcome(OutcomeFixtures.FOUR);
    pao.setRon(BaseClass.YAKUMAN.base(), 2, 3, 0, 0);
    RoundOutcome nagashi = new RoundOutcome(OutcomeFixtures.FOUR);
    nagashi.setExhaustive(0, 0b1000, 0);
    long id = 2;
    for (RoundOutcome special : List.of(pao, nagashi)) {
      GrpState after = after(east2, special);
      games.add(new GameRecord(OutcomeFixtures.FOUR, id++, List.of(east2, after), finalScores(after, bust(after))));
    }
    GrpState shifted = new GrpState(OutcomeFixtures.FOUR, 2, 1, 0, new int[] {25_100, 24_900, 25_000, 25_000});
    games.add(new GameRecord(OutcomeFixtures.FOUR, id, List.of(east2, shifted), finalScores(shifted, bust(shifted))));
    games.addAll(OutcomeFixtures.played(40, 100, 5));
    return games;
  }

  /** トップの席が最下位の席から役満をロンする結果を作る。最下位の席は飛ぶ。 */
  private static RoundOutcome bust(GrpState state) {
    int lowest = 0;
    int highest = 0;
    for (int seat = 1; seat < 4; seat++) {
      lowest = state.score(seat) < state.score(lowest) ? seat : lowest;
      highest = state.score(seat) > state.score(highest) ? seat : highest;
    }
    RoundOutcome outcome = new RoundOutcome(OutcomeFixtures.FOUR);
    outcome.setRon(BaseClass.YAKUMAN.base(), highest, lowest, RoundOutcome.NO_SEAT, 0);
    return outcome;
  }

  /** 卓を結果 1 つで 1 局進めた卓を返す。半荘は続く前提。 */
  private static GrpState after(GrpState state, RoundOutcome outcome) {
    return GrpState.of(play(state, outcome));
  }

  /** 卓を結果 1 つで 1 局進めて半荘を終わらせ、終局の点数を返す。 */
  private static int[] finalScores(GrpState state, RoundOutcome outcome) {
    TableState end = play(state, outcome);
    Assert.assertTrue(end.gameOver(), end.toString());
    int[] scores = new int[4];
    for (int seat = 0; seat < 4; seat++) {
      scores[seat] = end.score(seat);
    }
    return scores;
  }

  private static TableState play(GrpState state, RoundOutcome outcome) {
    TableState start = table(state);
    TableState next = new TableState(OutcomeFixtures.FOUR);
    int[] delta = new int[4];
    Settlement.apply(OutcomeFixtures.FOUR, start, outcome, delta);
    Progression.next(start, outcome, delta, next);
    return next;
  }

  private static TableState table(GrpState state) {
    TableState table = new TableState(OutcomeFixtures.FOUR);
    state.copyInto(table);
    return table;
  }
}

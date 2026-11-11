package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeSection;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 1 局の前後で卓がどう変わったか（遷移）から、それを再現できる局の結果を探す処理を確かめる。打った半荘の遷移は全て見つかる。
 * 点数の動きで区別できない結果は、全て残る。包と流し満貫は、局の結果のネットワークでは表せない結果として見つかる。
 */
public class TransitionSearchTest {

  private static final int NONE = HandOutcomes.NONE;

  @Test
  public void everyPlayedTransitionIsFound() {
    TransitionSearch search = new TransitionSearch(OutcomeFixtures.FOUR);
    for (GameRecord game : OutcomeFixtures.played(150, 0, 3)) {
      List<GrpState> ends = game.handEnds();
      for (int hand = 0; hand <= ends.size(); hand++) {
        if (hand == 0) {
          search.fromStart();
        } else {
          search.from(ends.get(hand - 1));
        }
        int found = hand < ends.size() ? search.next(ends.get(hand)) : search.last(game.finalScores());
        Assert.assertTrue(found > 0, "game " + game.gameId() + " hand " + hand);
      }
    }
  }

  /**
   * 東 1 局で席 1 が席 2 から 2,000 点を得る遷移は、立直した席 2 から 1,000 点をロンしても、立直していない席 2 から 2,000 点を
   * ロンしても起こる。和了者が出した立直の供託は和了者に戻るので、それぞれに席 1 の立直のあり・なしの 2 通りがあり、合わせて 4 通りになる。
   */
  @Test
  public void everyOutcomeWithTheSamePointsIsKept() {
    TransitionSearch search = new TransitionSearch(OutcomeFixtures.FOUR);
    search.fromStart();
    int found = search.next(
        new GrpState(OutcomeFixtures.FOUR, 1, 0, 0, new int[] {25_000, 27_000, 23_000, 25_000}));
    Assert.assertEquals(found, 4);
    HandOutcomes hands = HandOutcomes.of(OutcomeFixtures.FOUR);
    Set<List<Integer>> outcomes = new HashSet<>();
    for (int index = 0; index < found; index++) {
      Assert.assertEquals(hands.section(search.main(index)), OutcomeSection.RON);
      Assert.assertEquals(hands.winner(search.main(index)), 1);
      Assert.assertEquals(hands.loser(search.main(index)), 2);
      Assert.assertEquals(search.other(index), NONE);
      outcomes.add(List.of(search.base(index), search.riichi(index)));
    }
    int cheap = BaseClass.B240.ordinal();
    int dear = BaseClass.B480.ordinal();
    Assert.assertEquals(
        outcomes, Set.of(List.of(cheap, 0b0100), List.of(cheap, 0b0110), List.of(dear, 0), List.of(dear, 0b0010)));
  }

  /**
   * 包のある役満のロン（放銃した席 3 と包の席 2 が 16,000 点ずつ払う）と、子の流し満貫（親が流れて本場が増える）は、局の結果の
   * ネットワークでは表せない。どちらも {@link TransitionSearch#extras} で見つかる。点数を 100 点ずらした遷移は、どの結果でも
   * 再現できない。
   */
  @Test
  public void paoAndNagashiAreFoundOutsideTheOutcomes() {
    TransitionSearch search = new TransitionSearch(OutcomeFixtures.FOUR);
    RoundOutcome pao = new RoundOutcome(OutcomeFixtures.FOUR);
    pao.setRon(BaseClass.YAKUMAN.base(), 1, 3, 2, 0);
    search.fromStart();
    Assert.assertEquals(search.next(advance(pao)), 0);
    Assert.assertTrue(search.extras());

    RoundOutcome nagashi = new RoundOutcome(OutcomeFixtures.FOUR);
    nagashi.setExhaustive(0, 0b1000, 0);
    GrpState afterNagashi = advance(nagashi);
    Assert.assertEquals(afterNagashi.honba(), 1);
    Assert.assertEquals(search.next(afterNagashi), 0);
    Assert.assertTrue(search.extras());

    Assert.assertEquals(
        search.next(new GrpState(OutcomeFixtures.FOUR, 1, 1, 0, new int[] {25_100, 24_900, 25_000, 25_000})), 0);
    Assert.assertFalse(search.extras());
  }

  /** 半荘の始めから、指定した結果で 1 局を終えた後の卓を返す。半荘は続く前提。 */
  private static GrpState advance(RoundOutcome outcome) {
    TableState start = new TableState(OutcomeFixtures.FOUR);
    TableState next = new TableState(OutcomeFixtures.FOUR);
    int[] delta = new int[4];
    Settlement.apply(OutcomeFixtures.FOUR, start, outcome, delta);
    Progression.next(start, outcome, delta, next);
    return GrpState.of(next);
  }
}

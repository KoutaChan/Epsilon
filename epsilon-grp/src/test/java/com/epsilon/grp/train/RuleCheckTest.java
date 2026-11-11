package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * ルールどおりに打った半荘は全て再現できる。包は、局の結果のネットワークが表せない結果として数える。壊した遷移は外れとして数え、
 * その例を残す。
 */
public class RuleCheckTest {

  @Test
  public void playedGamesAreReproducedByTheOutcomes() {
    List<GameRecord> games = OutcomeFixtures.played(200, 0, 7);
    long transitions = 0;
    for (GameRecord game : games) {
      transitions += game.handEnds().size() + 1;
    }
    RuleCheck.Count count = RuleCheck.count(OutcomeFixtures.FOUR, games, 5);
    Assert.assertEquals(count.transitions(), transitions);
    Assert.assertEquals(count.outcomes(), transitions);
    Assert.assertEquals(count.extras(), 0);
    Assert.assertEquals(count.mismatches(), 0);
    Assert.assertEquals(count.examples(), List.of());
  }

  /**
   * 半荘 1 は、東 1 局で包のある役満のロン（席 1 が席 3 から和了し、包は席 2）の後、東 2 局で席 2 が親のロンに放銃して飛ぶ。半荘 2 は、
   * 東 1 局の後の点数を 100 点ずらした記録で、東 2 局で席 3 が親の役満のロンに放銃して飛ぶ。
   */
  @Test
  public void paoAndBrokenTransitionsAreCountedApart() {
    RoundOutcome pao = new RoundOutcome(OutcomeFixtures.FOUR);
    pao.setRon(BaseClass.YAKUMAN.base(), 1, 3, 2, 0);
    TableState east2 = play(new TableState(OutcomeFixtures.FOUR), pao);
    RoundOutcome bust = new RoundOutcome(OutcomeFixtures.FOUR);
    bust.setRon(BaseClass.HANEMAN.base(), 1, 2, RoundOutcome.NO_SEAT, 0);
    GameRecord liable = new GameRecord(
        OutcomeFixtures.FOUR, 1, List.of(GrpState.of(east2)), scores(play(east2, bust)));

    GrpState broken = new GrpState(OutcomeFixtures.FOUR, 1, 0, 0, new int[] {25_100, 24_900, 25_000, 25_000});
    TableState brokenTable = new TableState(OutcomeFixtures.FOUR);
    brokenTable.set(1, 0, 0, broken.scores());
    RoundOutcome yakuman = new RoundOutcome(OutcomeFixtures.FOUR);
    yakuman.setRon(BaseClass.YAKUMAN.base(), 1, 3, RoundOutcome.NO_SEAT, 0);
    GameRecord wrong = new GameRecord(
        OutcomeFixtures.FOUR, 2, List.of(broken), scores(play(brokenTable, yakuman)));

    RuleCheck.Count count = RuleCheck.count(OutcomeFixtures.FOUR, List.of(liable, wrong), 5);
    Assert.assertEquals(count.transitions(), 4);
    Assert.assertEquals(count.outcomes(), 2);
    Assert.assertEquals(count.extras(), 1);
    Assert.assertEquals(count.mismatches(), 1);
    RuleCheck.Mismatch mismatch = count.examples().get(0);
    Assert.assertEquals(mismatch.gameId(), 2);
    Assert.assertEquals(mismatch.hand(), 0);
    Assert.assertEquals(mismatch.from().round(), 0);
    Assert.assertEquals(mismatch.next(), broken);
    Assert.assertNull(mismatch.finalScores());
  }

  /** 局を指定の結果で終えた後の卓を返す。 */
  private static TableState play(TableState start, RoundOutcome outcome) {
    TableState next = new TableState(OutcomeFixtures.FOUR);
    int[] delta = new int[4];
    Settlement.apply(OutcomeFixtures.FOUR, start, outcome, delta);
    Progression.next(start, outcome, delta, next);
    return next;
  }

  private static int[] scores(TableState table) {
    Assert.assertTrue(table.gameOver(), table.toString());
    int[] scores = new int[4];
    for (int seat = 0; seat < 4; seat++) {
      scores[seat] = table.score(seat);
    }
    return scores;
  }
}

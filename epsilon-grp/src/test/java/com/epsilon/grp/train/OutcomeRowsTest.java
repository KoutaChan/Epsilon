package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeWorkspace;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * OutcomeNetwork を学習する行を確かめる。各行は遷移を再現する結果を全て持つ。行の選び方と、学習用・早期終了用の検証への分け方は
 * GrpRows と同じ。周辺 NLL は手計算と合う。
 */
public class OutcomeRowsTest {

  @Test
  public void everyRowKeepsTheOutcomesThatReproduceIt() {
    RecencyWindow window = new RecencyWindow(10, 1, 64);
    OutcomeRows.Split rows = OutcomeFixtures.rows(window, 60);
    for (OutcomeRows part : List.of(rows.train(), rows.stop())) {
      Assert.assertTrue(part.size() > 0, part.toString());
      Assert.assertEquals(part.unexplained(), 0, part.toString());
      for (int row = 0; row < part.size(); row++) {
        Assert.assertTrue(part.outcomes(row) >= 1);
        Assert.assertTrue(window.contains(part.cohort(row)));
      }
      Assert.assertTrue(part.maxOutcomes() >= 1);
    }
    for (int row = 0; row < rows.stop().size(); row++) {
      Assert.assertTrue(rows.stop().cohort(row) > window.newest() - GrpRows.STOP_COHORTS);
    }
  }

  @Test
  public void onlyChosenHandEndsBecomeRows() {
    RecencyWindow window = new RecencyWindow(4, 1, 64);
    OutcomeRows.Split covered = OutcomeRows.builder(OutcomeFixtures.FOUR, window, GateCohort.NEXT, FinalRound::covers)
        .add(4, OutcomeFixtures.played(80, 0, 5))
        .add(3, OutcomeFixtures.played(80, 1_000, 6))
        .build();
    Assert.assertTrue(covered.train().size() > 0);
    for (int row = 0; row < covered.train().size(); row++) {
      GrpState state = GrpState.ofKey(OutcomeFixtures.FOUR, covered.train().key(row));
      Assert.assertTrue(FinalRound.covers(state), state.toString());
    }
  }

  /**
   * 東 2 局から全員不聴の荒牌で親が流れる遷移は、終わり方のうち聴牌の席が無い荒牌だけで再現できる。立直の項を持つ席は無い。
   * 一様なネットワークの NLL は log 45。
   */
  @Test
  public void aDrawWithoutTenpaiHasOneOutcome() {
    int[] even = {25_000, 25_000, 25_000, 25_000};
    GrpState east2 = new GrpState(OutcomeFixtures.FOUR, 1, 0, 0, even);
    GrpState east3 = new GrpState(OutcomeFixtures.FOUR, 2, 1, 0, even);
    GameRecord game = new GameRecord(OutcomeFixtures.FOUR, 5, List.of(east2, east3), even);
    RecencyWindow window = new RecencyWindow(3, 1, 64);
    OutcomeRows.Split split =
        OutcomeRows.builder(OutcomeFixtures.FOUR, window, GateCohort.NEXT, state -> state.round() == 1)
            .add(3, List.of(game))
            .build();
    OutcomeRows rows = split.train().size() == 1 ? split.train() : split.stop();
    Assert.assertEquals(rows.size(), 1);
    Assert.assertEquals(rows.outcomes(0), 1);
    OutcomeShape shape = new OutcomeShape(OutcomeFixtures.FOUR, GrpFixtures.plainFields(), 4);
    double nll = rows.nll(OutcomeFixtures.uniform(shape), OutcomeWorkspace.of(shape));
    Assert.assertEquals(nll, Math.log(45), 1e-12);
  }
}

package com.epsilon.grp.train;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.mahjong.rules.Variant;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの局の結果のネットワークを、エンジンで打った局を数えたファイルから学習する（{@code native} のグループ）。DJL で求めた
 * 対数確率の各項と周辺 NLL は、純 Java のネットワークの値と合う。学習すると、早期終了用の検証の行の NLL は、一様なネットワークの
 * NLL から 0.5 を超えて下がる。
 */
public class ThreePlayerOutcomeTrainerTest {

  private static final Variant THREE = Variant.TENHOU_3P;

  /** 許容誤差。float32 の積和と log softmax で、足す順が違うと出る誤差を見込む。 */
  private static final double TOLERANCE = 1e-4;

  @Test(groups = "native")
  public void threePlayerOutcomesAreLearnedFromCountFiles() {
    RecencyWindow window = RecencyWindow.uniform(5, 64);
    OutcomeRows.Builder builder = OutcomeRows.builder(THREE, window, GateCohort.NEXT, state -> true);
    for (int cohort = window.first(); cohort <= window.newest(); cohort++) {
      EngineGames.Played played = EngineGames.play(THREE, 150, cohort * 100_000L, cohort);
      CountedHands.Builder counts = CountedHands.builder(THREE, cohort, 0, GrpFixtures.STAMP);
      for (EngineGames.Hand hand : played.hands()) {
        counts.add(hand.gameId(), hand.start(), HandResult.raw(hand.outcome(), hand.start().dealer()));
      }
      builder.add(cohort, counts.build());
    }
    OutcomeRows.Split rows = builder.build();
    Assert.assertTrue(rows.train().size() > 1_000 && rows.stop().size() > 100, rows.toString());
    Assert.assertEquals(rows.train().maxOutcomes(), 1);
    OutcomeShape shape = new OutcomeShape(THREE, GrpFixtures.plainFields(), 16);

    OutcomeNetwork random = GrpFixtures.outcomes(shape, 3, 0.4);
    OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
    OutcomeRows train = rows.train();
    OutcomeBatch batch = new OutcomeBatch(shape, train.size(), train.maxOutcomes());
    for (int row = 0; row < train.size(); row++) {
      batch.add(train, row, 1f);
    }
    int terms = shape.outcomes().terms();
    try (OutcomeGraph graph = OutcomeGraph.of(shape, random.weights())) {
      float[] djl = graph.terms(batch);
      for (int row = 0; row < train.size(); row += 7) {
        random.forward(GrpState.ofKey(THREE, train.key(row)), workspace);
        for (int term = 0; term < terms; term++) {
          Assert.assertEquals(djl[row * terms + term], workspace.term(term), TOLERANCE, "row " + row + " term " + term);
        }
      }
      double expected = train.nll(random, workspace) * train.totalWeight();
      Assert.assertEquals(graph.weightedNll(batch), expected, TOLERANCE * Math.abs(expected));
    }

    double uniform = rows.stop().nll(OutcomeFixtures.uniform(shape), workspace);
    GrpTrainingRule rule = new GrpTrainingRule(GrpInit.SCRATCH, 1e-2, 0, 8, 512, 10, 2, 4, 11);
    OutcomeTrained trained = OutcomeTrainer.train(shape, null, rows.train(), rows.stop(), rule, () -> false);
    Assert.assertEquals(trained.ending(), TrainingEnding.COMPLETED);
    double learned = rows.stop().nll(trained.network(), workspace);
    Assert.assertEquals(learned, trained.stopNll(), 1e-3);
    Assert.assertTrue(learned < uniform - 0.5, "learned " + learned + " vs uniform " + uniform);
  }
}

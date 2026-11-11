package com.epsilon.grp.train;

import ai.djl.engine.Engine;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeWorkspace;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 局の結果のネットワークを DJL で学習するテスト（{@code native} のグループ）。対数確率の項と周辺 NLL は、純 Java のネットワークと
 * 許容誤差の範囲で合う。決まった分布で打った半荘で周辺尤度を学習すると、NLL は一様なネットワークより下がる。前のネットワークから
 * 続けて学ぶ（{@code WARM}）と、始めより悪くならない。停止のフラグが立っていれば、次のステップの前に止まる。torch のスレッドの数を
 * 指定した学習は、終われば元の数に戻す。
 */
public class OutcomeTrainerTest {

  /** 許容誤差。float32 の積和と log softmax で、足す順が違うことから出る誤差を見込む。 */
  private static final double TOLERANCE = 1e-4;

  @Test(groups = "native")
  public void djlTermsAndNllMatchThePureJavaNetwork() {
    OutcomeRows rows = OutcomeFixtures.rows(new RecencyWindow(4, 1, 64), 20).train();
    OutcomeShape shape = new OutcomeShape(OutcomeFixtures.FOUR, GrpFixtures.plainFields(), 16);
    OutcomeNetwork network = GrpFixtures.outcomes(shape, 3, 0.4);
    OutcomeBatch batch = new OutcomeBatch(shape, rows.size(), rows.maxOutcomes());
    for (int row = 0; row < rows.size(); row++) {
      batch.add(rows, row, 1f);
    }
    OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
    int terms = shape.outcomes().terms();
    try (OutcomeGraph graph = OutcomeGraph.of(shape, network.weights())) {
      float[] djl = graph.terms(batch);
      for (int row = 0; row < rows.size(); row++) {
        network.forward(GrpState.ofKey(OutcomeFixtures.FOUR, rows.key(row)), workspace);
        for (int term = 0; term < terms; term++) {
          Assert.assertEquals(djl[row * terms + term], workspace.term(term), TOLERANCE, "row " + row + " term " + term);
        }
      }
      double expected = rows.nll(network, workspace) * rows.totalWeight();
      Assert.assertEquals(graph.weightedNll(batch), expected, TOLERANCE * Math.abs(expected));
    }
  }

  @Test(groups = "native")
  public void marginalLikelihoodTrainingLowersTheNll() {
    OutcomeRows.Split rows = OutcomeFixtures.rows(new RecencyWindow(10, 1, 64), 400);
    OutcomeShape shape = new OutcomeShape(OutcomeFixtures.FOUR, GrpFixtures.plainFields(), 16);
    OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
    double uniform = rows.stop().nll(OutcomeFixtures.uniform(shape), workspace);
    GrpTrainingRule rule = new GrpTrainingRule(GrpInit.SCRATCH, 1e-2, 0, 8, 512, 10, 2, 4, 11);
    OutcomeTrained trained = OutcomeTrainer.train(shape, null, rows.train(), rows.stop(), rule, () -> false);
    Assert.assertEquals(trained.ending(), TrainingEnding.COMPLETED);
    double learned = rows.stop().nll(trained.network(), workspace);
    Assert.assertEquals(learned, trained.stopNll(), 1e-3);
    Assert.assertTrue(learned < uniform - 0.5, "learned " + learned + " vs uniform " + uniform);

    OutcomeTrained warmed = OutcomeTrainer.train(
        shape, trained.network(), rows.train(), rows.stop(), defaults(GrpInit.WARM, 13), () -> false);
    Assert.assertEquals(warmed.ending(), TrainingEnding.COMPLETED);
    Assert.assertTrue(warmed.stopNll() <= trained.stopNll() + 1e-4, warmed + " after " + trained);
  }

  @Test(groups = "native")
  public void closingStopsBeforeTheNextStep() {
    OutcomeRows.Split rows = OutcomeFixtures.rows(new RecencyWindow(4, 1, 64), 120);
    OutcomeShape shape = new OutcomeShape(OutcomeFixtures.FOUR, GrpFixtures.plainFields(), 8);
    OutcomeTrained stopped = OutcomeTrainer.train(
        shape, null, rows.train(), rows.stop(), defaults(GrpInit.SCRATCH, 1), () -> true);
    Assert.assertEquals(stopped.ending(), TrainingEnding.STOPPED);
    Assert.assertNull(stopped.network());
    Assert.assertEquals(stopped.halfEpochs(), 0);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> OutcomeTrainer.train(
            shape, null, rows.train(), rows.stop(), defaults(GrpInit.WARM, 1), () -> false));
  }

  /**
   * スレッドの数を指定した学習は、学習の間だけその数で回し、終われば元の数に戻す。学習が例外で終わっても戻す。1 未満の数は拒む。
   */
  @Test(groups = "native")
  public void threadCountHoldsOnlyDuringTraining() {
    OutcomeRows.Split rows = OutcomeFixtures.rows(new RecencyWindow(4, 1, 64), 60);
    OutcomeShape shape = new OutcomeShape(OutcomeFixtures.FOUR, GrpFixtures.plainFields(), 8);
    Engine.getEngine("PyTorch");
    int before = JniUtils.getNumThreads();
    int threads = before == 1 ? 2 : 1;
    int[] during = {-1};
    OutcomeTrained trained = OutcomeTrainer.train(
        shape, null, rows.train(), rows.stop(), defaults(GrpInit.SCRATCH, 1), threads, () -> {
          during[0] = JniUtils.getNumThreads();
          return false;
        });
    Assert.assertEquals(trained.ending(), TrainingEnding.COMPLETED);
    Assert.assertEquals(during[0], threads);
    Assert.assertEquals(JniUtils.getNumThreads(), before);

    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> OutcomeTrainer.train(
            shape, null, rows.train(), rows.stop(), defaults(GrpInit.WARM, 1), threads, () -> false));
    Assert.assertEquals(JniUtils.getNumThreads(), before);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> OutcomeTrainer.train(
            shape, null, rows.train(), rows.stop(), defaults(GrpInit.SCRATCH, 1), 0, () -> false));
    Assert.assertEquals(JniUtils.getNumThreads(), before);
  }

  /**
   * grp.training の既定の学び方で、学習 1 回の設定を作る。ゼロから学ぶときは学習率 1e-3 で 16 エポック、{@code WARM} のときは
   * 学習率 3e-4 で 1 エポック。
   */
  private static GrpTrainingRule defaults(GrpInit init, long seed) {
    return init == GrpInit.SCRATCH
        ? new GrpTrainingRule(init, 1e-3, 0, 16, 4096, 10, 2, 4, seed)
        : new GrpTrainingRule(init, 3e-4, 0, 1, 4096, 10, 2, 4, seed);
  }
}

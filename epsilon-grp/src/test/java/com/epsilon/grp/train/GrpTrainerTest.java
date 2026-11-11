package com.epsilon.grp.train;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpTensor;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * GRP を DJL で学習するテスト（{@code native} のグループ）。順伝播と損失は、純 Java のネットワークと許容誤差の範囲で合う。損失の
 * 勾配は中心差分と合う。重みはビットを変えずに往復する。小さな合成データで学習すると、早期終了用の検証の行の NLL は、順列の頻度だけで予測したときより下がる。
 * 前のネットワークから続けて学ぶ（{@code WARM}）と、始めより悪くならない。停止のフラグが立っていれば、次のステップの前に止まる。
 */
public class GrpTrainerTest {

  /** 許容誤差。float32 の積和で、足す順が違うことから出る誤差を見込む。 */
  private static final double TOLERANCE = 2e-5;

  /** 中心差分で重みを動かす幅。 */
  private static final double STEP = 1e-6;

  @Test(groups = "native")
  public void djlForwardMatchesThePureJavaNetwork() {
    for (Variant variant : Variant.values()) {
      for (boolean phaseHeads : new boolean[] {false, true}) {
        GrpShape shape = GrpFixtures.shape(variant, 24, phaseHeads);
        assertForwardMatches(shape, GrpWorkspace.of(shape));
      }
    }
  }

  @Test(groups = "native")
  public void weightsRoundTripThroughTheGraph() {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_4P, 12, true);
    float[] weights = GrpFixtures.network(shape, 3, 0.5).weights();
    try (GrpGraph graph = GrpGraph.of(shape, weights)) {
      float[] back = graph.weights();
      Assert.assertEquals(back.length, weights.length);
      for (int index = 0; index < weights.length; index++) {
        int expected = Float.floatToRawIntBits(weights[index]);
        Assert.assertEquals(Float.floatToRawIntBits(back[index]), expected, "weight " + index);
      }
    }
  }

  /**
   * 更新は勾配を持たない側の配列に書くが、メモリは同じなので、次のステップの順伝播（勾配を持つ側）にも届く。勾配はステップをまたいで
   * 足し込まれない。
   */
  @Test(groups = "native")
  public void updatesReachTheNextForwardAndGradientsDoNotCarryOver() {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_4P, 8, false);
    RowBatch batch = new RowBatch(shape, 64);
    for (GameRecord game : TrainFixtures.simulated(Variant.TENHOU_4P, 20, 0, 3)) {
      for (int handEnd = 0; handEnd < game.handEnds().size() && !batch.full(); handEnd++) {
        batch.add(game.handEnds().get(handEnd), game.realizedPermutation(handEnd), 1f);
      }
    }
    try (GrpGraph graph = GrpGraph.of(shape, GrpFixtures.network(shape, 7, 0.3).weights())) {
      double before = graph.weightedNll(batch) / batch.size;
      Assert.assertEquals(graph.step(batch, 1e-2, 0, 10).loss(), before, 1e-5);
      double updated = graph.weightedNll(batch) / batch.size;
      Assert.assertEquals(graph.step(batch, 1e-9, 0, 10).loss(), updated, 1e-5);
      Assert.assertTrue(updated < before, updated + " after " + before);
    }
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      NDArray x = manager.create(new float[] {1, 2, 3});
      x.setRequiresGradient(true);
      backward(manager, x, 2);
      try (NDArray gradient = x.getGradient()) {
        Assert.assertEquals(gradient.toFloatArray(), new float[] {2, 2, 2});
        gradient.muli(0);
      }
      backward(manager, x, 3);
      try (NDArray gradient = x.getGradient()) {
        Assert.assertEquals(gradient.toFloatArray(), new float[] {3, 3, 3});
      }
    }
  }

  @Test(groups = "native")
  public void lossMatchesAHandComputation() {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_3P, 16, true);
    GrpNetwork network = GrpFixtures.network(shape, 5, 0.6);
    GrpWorkspace workspace = GrpWorkspace.of(shape);
    List<GameRecord> games = TrainFixtures.simulated(Variant.TENHOU_3P, 40, 0, 7);
    RowBatch batch = new RowBatch(shape, 256);
    double expected = 0;
    float weight = 0.25f;
    outer:
    for (GameRecord game : games) {
      for (int handEnd = 0; handEnd < game.handEnds().size(); handEnd++) {
        if (batch.full()) {
          break outer;
        }
        GrpState state = game.handEnds().get(handEnd);
        int permutation = game.realizedPermutation(handEnd);
        batch.add(state, permutation, weight);
        expected -= weight * network.logProbability(state, permutation, workspace);
        weight = weight == 0.25f ? 1f : 0.25f;
      }
    }
    try (GrpGraph graph = GrpGraph.of(shape, network.weights())) {
      Assert.assertEquals(graph.weightedNll(batch), expected, 1e-4 * Math.abs(expected));
    }
  }

  /**
   * DJL で求めた損失の勾配が、全ての重みについて中心差分と合う。中心差分は double で計算した順伝播から求める。勾配は呼び出しを
   * またいで足し込まれず、重みも変わらない。
   */
  @Test(groups = "native")
  public void lossGradientMatchesFiniteDifferences() {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_4P, 6, false);
    float[] weights = GrpFixtures.network(shape, 31, 0.4).weights();
    RowBatch batch = new RowBatch(shape, 48);
    for (GameRecord game : TrainFixtures.simulated(Variant.TENHOU_4P, 12, 0, 41)) {
      for (int handEnd = 0; handEnd < game.handEnds().size() && !batch.full(); handEnd++) {
        float weight = handEnd % 2 == 0 ? 1f : 0.25f;
        batch.add(game.handEnds().get(handEnd), game.realizedPermutation(handEnd), weight);
      }
    }
    Assert.assertTrue(batch.full());
    float[] gradient;
    try (GrpGraph graph = GrpGraph.of(shape, weights)) {
      gradient = graph.gradient(batch);
      Assert.assertEquals(graph.gradient(batch), gradient, 1e-6f);
      Assert.assertEquals(graph.weights(), weights);
    }
    double[] point = new double[weights.length];
    for (int index = 0; index < weights.length; index++) {
      point[index] = weights[index];
    }
    for (int index = 0; index < point.length; index++) {
      double held = point[index];
      point[index] = held + STEP;
      double up = loss(shape, batch, point);
      point[index] = held - STEP;
      double down = loss(shape, batch, point);
      point[index] = held;
      double expected = (up - down) / (2 * STEP);
      Assert.assertEquals(gradient[index], expected, 5e-5 + 1e-3 * Math.abs(expected), "weight " + index);
    }
  }

  @Test(groups = "native")
  public void trainingLowersTheStopNllBelowTheBaseRates() {
    Variant variant = Variant.TENHOU_4P;
    RecencyWindow window = new RecencyWindow(10, 1, 64);
    GrpRows.Split rows = rows(variant, window, 1_500);
    GrpShape shape = GrpFixtures.shape(variant, 16, false);
    GrpTrainingRule rule = new GrpTrainingRule(GrpInit.SCRATCH, 3e-3, 0, 6, 512, 10, 2, 4, 11);
    GrpTrained trained = GrpTrainer.train(shape, null, rows.train(), rows.stop(), rule, () -> false);
    Assert.assertEquals(trained.ending(), TrainingEnding.COMPLETED);
    Assert.assertTrue(trained.halfEpochs() >= 2 && trained.halfEpochs() <= 12, trained.toString());
    double base = baseRateNll(rows.train(), rows.stop());
    double learned = stopNll(trained.network(), rows.stop());
    Assert.assertEquals(learned, trained.stopNll(), 1e-3);
    Assert.assertTrue(learned < base - 0.1, "learned " + learned + " vs base rates " + base);

    GrpTrained warmed = GrpTrainer.train(
        shape, trained.network(), rows.train(), rows.stop(), defaults(GrpInit.WARM, 13), () -> false);
    Assert.assertEquals(warmed.ending(), TrainingEnding.COMPLETED);
    Assert.assertTrue(warmed.stopNll() <= trained.stopNll() + 1e-4, warmed + " after " + trained);
  }

  @Test(groups = "native")
  public void phaseHeadsTrain() {
    Variant variant = Variant.TENHOU_4P;
    RecencyWindow window = new RecencyWindow(10, 2, 64);
    GrpRows.Split rows = rows(variant, window, 200);
    GrpShape shape = GrpFixtures.shape(variant, 8, true);
    GrpTrainingRule rule = new GrpTrainingRule(GrpInit.SCRATCH, 1e-3, 1e-5, 1, 256, 10, 2, 4, 17);
    GrpTrained trained = GrpTrainer.train(shape, null, rows.train(), rows.stop(), rule, () -> false);
    Assert.assertEquals(trained.ending(), TrainingEnding.COMPLETED);
    Assert.assertEquals(trained.network().shape(), shape);
    Assert.assertTrue(Double.isFinite(trained.stopNll()));
  }

  @Test(groups = "native")
  public void closingStopsBeforeTheNextStep() {
    Variant variant = Variant.TENHOU_3P;
    GrpRows.Split rows = rows(variant, new RecencyWindow(10, 1, 64), 100);
    GrpShape shape = GrpFixtures.shape(variant, 8, false);
    GrpTrained stopped = GrpTrainer.train(
        shape, null, rows.train(), rows.stop(), defaults(GrpInit.SCRATCH, 1), () -> true);
    Assert.assertEquals(stopped.ending(), TrainingEnding.STOPPED);
    Assert.assertNull(stopped.network());
    Assert.assertEquals(stopped.halfEpochs(), 0);
    GrpTrainingRule warm = defaults(GrpInit.WARM, 1);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> GrpTrainer.train(shape, null, rows.train(), rows.stop(), warm, () -> false));
  }

  /** スレッドの数を指定した学習は、終われば元のスレッドの数に戻す。 */
  @Test(groups = "native")
  public void threadCountIsRestoredAfterTraining() {
    Variant variant = Variant.TENHOU_4P;
    GrpRows.Split rows = rows(variant, new RecencyWindow(10, 1, 64), 100);
    GrpShape shape = GrpFixtures.shape(variant, 8, false);
    Engine.getEngine("PyTorch");
    int before = JniUtils.getNumThreads();
    int threads = before == 1 ? 2 : 1;
    GrpTrained trained = GrpTrainer.train(
        shape, null, rows.train(), rows.stop(), defaults(GrpInit.SCRATCH, 1), threads, () -> false);
    Assert.assertEquals(trained.ending(), TrainingEnding.COMPLETED);
    Assert.assertEquals(JniUtils.getNumThreads(), before);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> GrpTrainer.train(
            shape, null, rows.train(), rows.stop(), defaults(GrpInit.SCRATCH, 1), 0, () -> false));
  }

  private static void assertForwardMatches(GrpShape shape, GrpWorkspace workspace) {
    GrpNetwork network = GrpFixtures.network(shape, 23, 0.4);
    List<GrpState> states = new ArrayList<>();
    for (GameRecord game : TrainFixtures.simulated(shape.variant(), 60, 0, 29)) {
      states.addAll(game.handEnds());
    }
    RowBatch batch = new RowBatch(shape, states.size());
    for (GrpState state : states) {
      batch.add(state, 0, 1f);
    }
    double[] expected = new double[shape.outputs()];
    try (GrpGraph graph = GrpGraph.of(shape, network.weights())) {
      float[] logits = graph.logits(batch);
      for (int row = 0; row < states.size(); row++) {
        network.permutations(states.get(row), workspace, expected);
        double[] actual = softmax(logits, row * shape.outputs(), shape.outputs());
        for (int k = 0; k < expected.length; k++) {
          Assert.assertEquals(actual[k], expected[k], TOLERANCE, shape + " row " + row + " permutation " + k);
        }
      }
    }
  }

  /**
   * 範囲の各コホートで半荘を打ち、学習用の行と検証用の行に分ける。
   *
   * @param games コホートごとの半荘の数
   */
  private static GrpRows.Split rows(Variant variant, RecencyWindow window, int games) {
    GrpRows.Builder builder = GrpRows.builder(variant, window, GateCohort.NEXT);
    for (int cohort = window.first(); cohort <= window.newest(); cohort++) {
      builder.add(cohort, TrainFixtures.simulated(variant, games, cohort * 100_000L, cohort));
    }
    return builder.build();
  }

  /**
   * 学習用の行で順列ごとの重み付きの頻度を数え、その頻度で予測したときの、検証用の行での重み付きの NLL を返す。頻度が 0 の順列は
   * 0.5 として数える。
   */
  private static double baseRateNll(GrpRows train, GrpRows stop) {
    int outputs = GrpFixtures.shape(train.variant(), 1, false).outputs();
    double[] counts = new double[outputs];
    for (int row = 0; row < train.size(); row++) {
      counts[train.permutation(row)] += train.weight(row);
    }
    double total = 0;
    for (int k = 0; k < outputs; k++) {
      counts[k] = counts[k] > 0 ? counts[k] : 0.5;
      total += counts[k];
    }
    double sum = 0;
    for (int row = 0; row < stop.size(); row++) {
      sum -= stop.weight(row) * Math.log(counts[stop.permutation(row)] / total);
    }
    return sum / stop.totalWeight();
  }

  /** 検証用の行での重み付きの同時 NLL を、純 Java のネットワークで求める。 */
  private static double stopNll(GrpNetwork network, GrpRows stop) {
    GrpWorkspace workspace = GrpWorkspace.of(network.shape());
    double sum = 0;
    for (int row = 0; row < stop.size(); row++) {
      GrpState state = GrpState.ofKey(stop.variant(), stop.key(row));
      sum -= stop.weight(row) * network.logProbability(state, stop.permutation(row), workspace);
    }
    return sum / stop.totalWeight();
  }

  /**
   * バッチの損失 Σ w·(−Σ_π t(π)·log softmax(z)_π) / 行数 を double で求める。段階ごとの出力ヘッドを持たない形だけを扱い、重みは
   * GrpTensor の順に並ぶ。
   */
  private static double loss(GrpShape shape, RowBatch batch, double[] weights) {
    int width = shape.width();
    int hidden = shape.hidden();
    int outputs = shape.outputs();
    int[] at = GrpTensor.starts(shape);
    double[] first = new double[hidden];
    double[] second = new double[hidden];
    double[] z = new double[outputs];
    double total = 0;
    for (int row = 0; row < batch.size; row++) {
      for (int j = 0; j < hidden; j++) {
        double sum = weights[at[GrpTensor.B1.ordinal()] + j];
        for (int i = 0; i < width; i++) {
          sum += weights[at[GrpTensor.W1.ordinal()] + i * hidden + j] * batch.inputs[row * width + i];
        }
        first[j] = Math.max(sum, 0);
      }
      for (int j = 0; j < hidden; j++) {
        double sum = weights[at[GrpTensor.B2.ordinal()] + j];
        for (int i = 0; i < hidden; i++) {
          sum += weights[at[GrpTensor.W2.ordinal()] + i * hidden + j] * first[i];
        }
        second[j] = Math.max(sum, 0);
      }
      double max = Double.NEGATIVE_INFINITY;
      for (int k = 0; k < outputs; k++) {
        double sum = weights[at[GrpTensor.B3.ordinal()] + k];
        for (int i = 0; i < hidden; i++) {
          sum += weights[at[GrpTensor.W3.ordinal()] + i * outputs + k] * second[i];
        }
        z[k] = sum;
        max = Math.max(max, sum);
      }
      double exponents = 0;
      for (int k = 0; k < outputs; k++) {
        exponents += Math.exp(z[k] - max);
      }
      double log = max + Math.log(exponents);
      double crossEntropy = 0;
      for (int k = 0; k < outputs; k++) {
        crossEntropy += batch.targets[row * outputs + k] * (log - z[k]);
      }
      total += batch.weights[row] * crossEntropy;
    }
    return total / batch.size;
  }

  private static void backward(NDManager manager, NDArray x, float factor) {
    try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
      collector.backward(x.mul(factor).sum());
    }
  }

  private static double[] softmax(float[] logits, int from, int count) {
    double max = Double.NEGATIVE_INFINITY;
    for (int k = 0; k < count; k++) {
      max = Math.max(max, logits[from + k]);
    }
    double[] probabilities = new double[count];
    double sum = 0;
    for (int k = 0; k < count; k++) {
      probabilities[k] = Math.exp(logits[from + k] - max);
      sum += probabilities[k];
    }
    for (int k = 0; k < count; k++) {
      probabilities[k] /= sum;
    }
    return probabilities;
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

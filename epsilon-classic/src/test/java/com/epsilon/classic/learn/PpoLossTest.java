package com.epsilon.classic.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.classic.model.HlGauss;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Random;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 選んだ手だけで計算する PPO・エントロピー・価値の損失（CPU の FP32）を、手で計算した値と照合する。ほかに次のことを確かめる。clip の
 * 向き（優位が正で比が上に外れた行と、負で下に外れた行は勾配 0）。方策の損失に入れない行の勾配は 0。学習の前と同じ方策なら KL は 0 で、
 * 負にならない。エントロピーの項はエントロピーを上げる。アームが 2 つのバンディットを手で作ると、1 ステップで優位の高いアームの確率が
 * 上がる。価値の教師の HL-Gauss の分布も確かめる。
 */
public class PpoLossTest {

  private static final double TOLERANCE = 1e-5;

  private NDManager manager;

  @BeforeClass(groups = "native")
  public void open() {
    manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
  }

  @AfterClass(groups = "native", alwaysRun = true)
  public void close() {
    if (manager != null) {
      manager.close();
    }
  }

  /** 3 行 × 3 手（行 1 は合法手が 2 つ、行 2 は方策の損失に入れない）の損失と指標。 */
  @Test(groups = "native")
  public void theLossMatchesAHandComputation() {
    int rows = 3;
    int actions = 3;
    int bins = 5;
    float[] logits = {0.3f, -0.2f, 0.9f, 1.1f, 0.4f, 0f, -0.5f, 0.2f, 0.1f};
    float[] legal = {1, 1, 1, 1, 1, 0, 1, 1, 1};
    float[] chosen = {0, 1, 0, 1, 0, 0, 0, 0, 1};
    double[][] old = {{0.2, 0.5, 0.3}, {0.6, 0.4}, {0.3, 0.3, 0.4}};
    float[] rollout = new float[rows * actions];
    for (int row = 0; row < rows; row++) {
      for (int action = 0; action < old[row].length; action++) {
        rollout[row * actions + action] = (float) Math.log(old[row][action]);
      }
    }
    float[] advantage = {0.8f, -0.5f, 0.3f};
    float[] credit = {1f, 0.6f, 0.4f};
    float[] actor = {1, 1, 0};
    Random random = new Random(7);
    float[] valueLogits = new float[rows * bins];
    float[] valueTargets = new float[rows * bins];
    for (int row = 0; row < rows; row++) {
      double sum = 0;
      for (int bin = 0; bin < bins; bin++) {
        valueLogits[row * bins + bin] = (float) random.nextGaussian();
        valueTargets[row * bins + bin] = random.nextFloat();
        sum += valueTargets[row * bins + bin];
      }
      for (int bin = 0; bin < bins; bin++) {
        valueTargets[row * bins + bin] /= (float) sum;
      }
    }
    double clip = 0.2;
    double beta = 0.01;
    PpoLoss.Coefficients coefficients = new PpoLoss.Coefficients(clip, beta, 2, 3);

    double surrogate = 0;
    double entropy = 0;
    double kl = 0;
    double clipped = 0;
    double value = 0;
    for (int row = 0; row < rows; row++) {
      int count = old[row].length;
      double[] current = softmax(logits, row * actions, count);
      int a = 0;
      while (chosen[row * actions + a] == 0) {
        a++;
      }
      double ratio = current[a] / old[row][a];
      double bounded = Math.max(1 - clip, Math.min(1 + clip, ratio));
      double rowEntropy = 0;
      double rowKl = 0;
      for (int action = 0; action < count; action++) {
        rowEntropy -= current[action] * Math.log(current[action]);
        rowKl += old[row][action] * (Math.log(old[row][action]) - Math.log(current[action]));
      }
      if (actor[row] == 1) {
        surrogate += credit[row] * Math.min(ratio * advantage[row], bounded * advantage[row]);
        entropy += rowEntropy;
        kl += Math.max(0, rowKl);
        clipped += Math.abs(ratio - 1) > clip ? 1 : 0;
      }
      double[] predicted = softmax(valueLogits, row * bins, bins);
      for (int bin = 0; bin < bins; bin++) {
        value -= valueTargets[row * bins + bin] * Math.log(predicted[bin]);
      }
    }
    double objective = value / 3 - surrogate / 2 - beta * entropy / 2;

    NDArray parameters = manager.create(logits, new Shape(rows, actions));
    parameters.setRequiresGradient(true);
    NDArray legalMask = manager.create(legal, new Shape(rows, actions));
    try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
      PpoLoss.Terms terms =
          PpoLoss.compute(
              logProbabilities(parameters, legalMask), legalMask, manager.create(chosen, new Shape(rows, actions)),
              manager.create(rollout, new Shape(rows, actions)), manager.create(advantage),
              manager.create(credit), manager.create(actor), manager.create(valueLogits, new Shape(rows, bins)),
              manager.create(valueTargets, new Shape(rows, bins)), coefficients);
      assertEquals(terms.objective().getFloat(), objective, TOLERANCE);
      float[] sums = terms.sums().toFloatArray();
      assertEquals(sums[PpoLoss.Sum.SURROGATE.ordinal()], surrogate, TOLERANCE);
      assertEquals(sums[PpoLoss.Sum.ENTROPY.ordinal()], entropy, TOLERANCE);
      assertEquals(sums[PpoLoss.Sum.KL.ordinal()], kl, TOLERANCE);
      assertEquals(sums[PpoLoss.Sum.VALUE.ordinal()], value, TOLERANCE);
      assertEquals(sums[PpoLoss.Sum.CLIPPED.ordinal()], clipped, 0.0);
      collector.backward(terms.objective());
    }
    float[] gradient = parameters.getGradient().toFloatArray();
    for (int action = 0; action < actions; action++) {
      assertEquals(gradient[2 * actions + action], 0f, 0f, "the row outside the policy loss");
      assertTrue(Float.isFinite(gradient[action]) && Float.isFinite(gradient[actions + action]));
    }
    assertEquals(gradient[actions + 2], 0f, 0f, "the illegal action");
  }

  /**
   * 選んだ手の確率 0.5、比 1.5 か 0.6、優位 ±1 の 4 行: 優位が正で比が上に外れた行と、負で下に外れた行は勾配 0。ほかの 2 行は、優位が
   * 正なら選んだ手のロジットを上げ、負なら下げる向き。
   */
  @Test(groups = "native")
  public void clippingStopsTheGradientOnlyInTheDirectionOfTheAdvantage() {
    float[] advantage = {1, 1, -1, -1};
    double[] ratio = {1.5, 0.6, 0.6, 1.5};
    int rows = advantage.length;
    float[] rollout = new float[rows * 2];
    float[] chosen = new float[rows * 2];
    for (int row = 0; row < rows; row++) {
      double old = 0.5 / ratio[row];
      rollout[row * 2] = (float) Math.log(old);
      rollout[row * 2 + 1] = (float) Math.log(1 - old);
      chosen[row * 2] = 1;
    }
    float[] gradient = policyGradient(new float[rows * 2], chosen, rollout, advantage, 0.2, 0);
    for (int index : new int[] {0, 1, 4, 5}) {
      assertEquals(gradient[index], 0f, 0f, "logit " + index);
    }
    assertTrue(gradient[2] < 0 && gradient[3] > 0, "a positive advantage raises the chosen logit");
    assertTrue(gradient[6] > 0 && gradient[7] < 0, "a negative advantage lowers the chosen logit");
  }

  /** 学習の前と同じ方策の行の KL は 0 で、負にならない。 */
  @Test(groups = "native")
  public void theKlOfAnUnchangedPolicyIsZero() {
    Random random = new Random(11);
    int rows = 64;
    int actions = 7;
    float[] logits = new float[rows * actions];
    for (int index = 0; index < logits.length; index++) {
      logits[index] = (float) (3 * random.nextGaussian());
    }
    float[] legal = new float[rows * actions];
    Arrays.fill(legal, 1f);
    NDArray legalMask = manager.create(legal, new Shape(rows, actions));
    NDArray current = logProbabilities(manager.create(logits, new Shape(rows, actions)), legalMask);
    float[] ones = new float[rows];
    Arrays.fill(ones, 1f);
    float[] chosen = new float[rows * actions];
    for (int row = 0; row < rows; row++) {
      chosen[row * actions + random.nextInt(actions)] = 1;
    }
    PpoLoss.Terms terms =
        PpoLoss.compute(
            current, legalMask, manager.create(chosen, new Shape(rows, actions)), current.duplicate(),
            manager.create(ones), manager.create(ones), manager.create(ones), manager.zeros(new Shape(rows, 3)),
            manager.ones(new Shape(rows, 3)).div(3), new PpoLoss.Coefficients(0.2, 0, rows, rows));
    float[] sums = terms.sums().toFloatArray();
    assertEquals(sums[PpoLoss.Sum.KL.ordinal()], 0f, 0f);
    assertEquals(sums[PpoLoss.Sum.CLIPPED.ordinal()], 0f, 0f);
    assertEquals(sums[PpoLoss.Sum.SURROGATE.ordinal()], rows, 1e-4);
  }

  /** 優位 0 でエントロピーの項だけなら、1 ステップでどの行もエントロピーが上がる。 */
  @Test(groups = "native")
  public void theEntropyTermRaisesTheEntropy() {
    float[] logits = {2f, 0f, -1f, 0.5f, 3f, 0.1f};
    float[] chosen = {1, 0, 0, 0, 1, 0};
    float[] rollout = new float[6];
    float[] gradient = policyGradient(logits, chosen, rollout, new float[2], 0.2, 1.0, 3);
    for (int row = 0; row < 2; row++) {
      double before = entropy(softmax(logits, row * 3, 3));
      float[] stepped = new float[3];
      for (int action = 0; action < 3; action++) {
        stepped[action] = logits[row * 3 + action] - 0.1f * gradient[row * 3 + action];
      }
      double after = entropy(softmax(stepped, 0, 3));
      assertTrue(after > before, "row " + row + ": " + before + " -> " + after);
    }
  }

  /**
   * アームが 2 つのバンディット: 一様の方策で両方のアームを 32 回ずつ引くと、1 ステップで優位の高いアームの確率が上がる（優位を入れ替え
   * ると逆）。
   */
  @Test(groups = "native")
  public void oneStepOnATwoArmedBanditFavoursTheBetterArm() {
    for (float sign : new float[] {1, -1}) {
      int pulls = 64;
      float[] chosen = new float[pulls * 2];
      float[] advantage = new float[pulls];
      float[] rollout = new float[pulls * 2];
      for (int pull = 0; pull < pulls; pull++) {
        int arm = pull % 2;
        chosen[pull * 2 + arm] = 1;
        advantage[pull] = sign * (arm == 0 ? 1f : -0.5f);
        rollout[pull * 2] = (float) Math.log(0.5);
        rollout[pull * 2 + 1] = (float) Math.log(0.5);
      }
      NDArray arms = manager.zeros(new Shape(1, 2));
      arms.setRequiresGradient(true);
      NDArray legal = manager.ones(new Shape(pulls, 2));
      float[] ones = new float[pulls];
      Arrays.fill(ones, 1f);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray logits = arms.broadcast(new Shape(pulls, 2));
        PpoLoss.Terms terms =
            PpoLoss.compute(
                logProbabilities(logits, legal), legal, manager.create(chosen, new Shape(pulls, 2)),
                manager.create(rollout, new Shape(pulls, 2)), manager.create(advantage), manager.create(ones),
                manager.create(ones), manager.zeros(new Shape(pulls, 3)), manager.ones(new Shape(pulls, 3)).div(3),
                new PpoLoss.Coefficients(0.2, 0.001, pulls, pulls));
        collector.backward(terms.objective());
      }
      float[] gradient = arms.getGradient().toFloatArray();
      float[] stepped = {-0.5f * gradient[0], -0.5f * gradient[1]};
      double first = softmax(stepped, 0, 2)[0];
      assertTrue(sign > 0 ? first > 0.5 : first < 0.5, "sign " + sign + ": arm 0 probability " + first);
    }
  }

  /** HL-Gauss の教師の分布は和が 1 で、ビンの中心で平均すると効用に戻る。 */
  @Test(groups = "native")
  public void hlGaussTargetsSumToOneAndDecodeToTheUtility() {
    Variant variant = Variant.TENHOU_4P;
    HlGauss support = new HlGauss(variant, RankUtility.TENHOU);
    support.initialize(manager);
    float[] utilities = {-1f, -0.4f, 0f, 1f / 3, 0.5f, 2f / 3};
    float[] targets = support.targets(manager.create(utilities)).toFloatArray();
    float[] logs = new float[targets.length];
    for (int row = 0; row < utilities.length; row++) {
      double sum = 0;
      for (int bin = 0; bin < HlGauss.BINS; bin++) {
        float probability = targets[row * HlGauss.BINS + bin];
        assertTrue(probability >= 0, "probability " + probability);
        sum += probability;
        logs[row * HlGauss.BINS + bin] = (float) Math.log(Math.max(1e-30, probability));
      }
      assertEquals(sum, 1.0, 1e-5);
      assertEquals(support.decode(logs, row * HlGauss.BINS), utilities[row], 1e-3);
    }
  }

  /** 方策の項だけで計算したロジットの勾配（価値の項は、勾配の無い一様分布にする）。 */
  private float[] policyGradient(
      float[] logits, float[] chosen, float[] rollout, float[] advantage, double clip, double beta) {
    return policyGradient(logits, chosen, rollout, advantage, clip, beta, 2);
  }

  private float[] policyGradient(
      float[] logits, float[] chosen, float[] rollout, float[] advantage, double clip, double beta, int actions) {
    int rows = advantage.length;
    NDArray parameters = manager.create(logits, new Shape(rows, actions));
    parameters.setRequiresGradient(true);
    NDArray legal = manager.ones(new Shape(rows, actions));
    float[] ones = new float[rows];
    Arrays.fill(ones, 1f);
    try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
      PpoLoss.Terms terms =
          PpoLoss.compute(
              logProbabilities(parameters, legal), legal, manager.create(chosen, new Shape(rows, actions)),
              manager.create(rollout, new Shape(rows, actions)), manager.create(advantage), manager.create(ones),
              manager.create(ones), manager.zeros(new Shape(rows, 3)), manager.ones(new Shape(rows, 3)).div(3),
              new PpoLoss.Coefficients(clip, beta, rows, rows));
      collector.backward(terms.objective());
    }
    return parameters.getGradient().toFloatArray();
  }

  /** 合法手だけで正規化した対数確率（合法でない位置は大きな負の値）。 */
  private static NDArray logProbabilities(NDArray logits, NDArray legal) {
    return logits.add(legal.sub(1f).mul(1e4f)).logSoftmax(1);
  }

  private static double[] softmax(float[] logits, int offset, int count) {
    double maximum = Double.NEGATIVE_INFINITY;
    for (int index = 0; index < count; index++) {
      maximum = Math.max(maximum, logits[offset + index]);
    }
    double sum = 0;
    double[] probabilities = new double[count];
    for (int index = 0; index < count; index++) {
      probabilities[index] = Math.exp(logits[offset + index] - maximum);
      sum += probabilities[index];
    }
    for (int index = 0; index < count; index++) {
      probabilities[index] /= sum;
    }
    return probabilities;
  }

  private static double entropy(double[] probabilities) {
    double entropy = 0;
    for (double probability : probabilities) {
      entropy -= probability * Math.log(probability);
    }
    return entropy;
  }
}

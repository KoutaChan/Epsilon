package com.epsilon.ai.grp;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 隠れ状態＋1歩の評価が、系列全体を DJL で通した周辺確率と一致することを確かめる。 */
public class EpsilonGrpStepperTest {
  @Test(groups = "native")
  public void oneStepAfterTheHiddenStateMatchesTheFullSequence() {
    try (Model model = Model.newInstance("grp", Device.cpu(), "PyTorch")) {
      var block = new EpsilonGrpNetwork();
      model.setBlock(block);
      block.initialize(
          model.getNDManager(),
          DataType.FLOAT32,
          new Shape(-1, -1, EpsilonGrpFeature.FEATURE_SIZE),
          new Shape(-1));
      var inference = new EpsilonGrpInference(model, 64);
      EpsilonGrpStepper stepper = inference.stepper();
      var random = new SplittableRandom(11);
      for (int length = 1; length <= 16; length++) {
        float[] prefix = sequence(random, length);
        float[][] next = new float[5][];
        List<float[]> full = new ArrayList<>();
        for (int candidate = 0; candidate < next.length; candidate++) {
          next[candidate] = sequence(random, 1);
          float[] joined = Arrays.copyOf(prefix, prefix.length + next[candidate].length);
          System.arraycopy(next[candidate], 0, joined, prefix.length, next[candidate].length);
          full.add(joined);
        }
        float[][] stepped = new float[next.length][EpsilonGrpRanks.MATRIX_SIZE];
        stepper.marginalsAfter(stepper.hidden(prefix), next, stepped);
        List<float[]> expected = inference.predictMarginalProbabilities(full);
        for (int candidate = 0; candidate < next.length; candidate++)
          for (int i = 0; i < EpsilonGrpRanks.MATRIX_SIZE; i++)
            Assert.assertEquals(
                stepped[candidate][i],
                expected.get(candidate)[i],
                1e-5f,
                "length " + length + " candidate " + candidate);
      }
    }
  }

  /** 局番号・本場・供託・点数の尺度に近い乱数の系列。 */
  private static float[] sequence(SplittableRandom random, int steps) {
    float[] values = new float[steps * EpsilonGrpFeature.FEATURE_SIZE];
    for (int step = 0; step < steps; step++) {
      int[] scores = new int[4];
      for (int seat = 0; seat < 4; seat++) scores[seat] = 5_000 + random.nextInt(45_000);
      float[] feature =
          EpsilonGrpFeature.fromProgress(
              random.nextInt(12), random.nextInt(4), random.nextInt(3), scores);
      System.arraycopy(feature, 0, values, step * feature.length, feature.length);
    }
    return values;
  }
}

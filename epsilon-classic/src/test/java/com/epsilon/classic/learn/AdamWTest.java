package com.epsilon.classic.learn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 自前の AdamW は DJL の AdamW（Optimizer.adamW）と同じ値に更新する。書き出した状態を新しい AdamW に戻すと、続きも同じ値になる。
 */
public class AdamWTest {

  private static final double RATE = 3e-3;
  private static final double DECAY = 0.01;
  private static final int STEPS = 6;

  @Test(groups = "native")
  public void updatesLikeDjlAndResumesFromItsState() {
    try (NDManager manager = NDManager.newBaseManager("PyTorch")) {
      Random random = new Random(7);
      Shape[] shapes = {new Shape(3, 4), new Shape(5)};
      float[][] start = new float[shapes.length][];
      for (int index = 0; index < shapes.length; index++) {
        start[index] = gaussians(random, (int) shapes[index].size());
      }
      float[][][] gradients = new float[STEPS][shapes.length][];
      for (int step = 0; step < STEPS; step++) {
        for (int index = 0; index < shapes.length; index++) {
          gradients[step][index] = gaussians(random, (int) shapes[index].size());
        }
      }

      NDArray[] ours = arrays(manager, start, shapes);
      NDArray[] djl = arrays(manager, start, shapes);
      Optimizer reference =
          Optimizer.adamW()
              .optLearningRateTracker(Tracker.fixed((float) RATE))
              .optWeightDecays((float) DECAY)
              .build();
      AdamW adamW = new AdamW(List.of(ours), RATE, DECAY);
      int half = STEPS / 2;
      for (int step = 0; step < half; step++) {
        adamW.step(arrays(manager, gradients[step], shapes));
        update(reference, djl, arrays(manager, gradients[step], shapes));
      }
      assertSameValues(ours, djl, "after " + half + " steps");

      // 状態を書き出して、新しい重みと新しい AdamW で続ける
      NDArray[] resumed = arrays(manager, values(ours), shapes);
      AdamW again = new AdamW(List.of(resumed), RATE, DECAY);
      again.restore(adamW.steps(), adamW.means(), adamW.variances());
      Assert.assertEquals(again.steps(), half);
      for (int step = half; step < STEPS; step++) {
        again.step(arrays(manager, gradients[step], shapes));
        update(reference, djl, arrays(manager, gradients[step], shapes));
      }
      assertSameValues(resumed, djl, "after " + STEPS + " steps");
    }
  }

  private static void update(Optimizer optimizer, NDArray[] weights, NDArray[] gradients) {
    for (int index = 0; index < weights.length; index++) {
      optimizer.update("p" + index, weights[index], gradients[index]);
    }
  }

  private static void assertSameValues(NDArray[] actual, NDArray[] expected, String label) {
    for (int index = 0; index < expected.length; index++) {
      float[] a = actual[index].toFloatArray();
      float[] e = expected[index].toFloatArray();
      Assert.assertEquals(a.length, e.length);
      for (int value = 0; value < e.length; value++) {
        Assert.assertEquals(a[value], e[value], 1e-7f * Math.max(1f, Math.abs(e[value])),
            label + " tensor " + index + " value " + value);
      }
    }
  }

  private static NDArray[] arrays(NDManager manager, float[][] values, Shape[] shapes) {
    NDArray[] arrays = new NDArray[values.length];
    for (int index = 0; index < values.length; index++) {
      arrays[index] = manager.create(values[index], shapes[index]);
    }
    return arrays;
  }

  private static float[][] values(NDArray[] arrays) {
    float[][] values = new float[arrays.length][];
    for (int index = 0; index < arrays.length; index++) {
      values[index] = arrays[index].toFloatArray();
    }
    return values;
  }

  private static float[] gaussians(Random random, int count) {
    float[] values = new float[count];
    for (int index = 0; index < count; index++) {
      values[index] = (float) random.nextGaussian();
    }
    return values;
  }
}

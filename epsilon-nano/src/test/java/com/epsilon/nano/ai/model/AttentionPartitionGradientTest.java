package com.epsilon.nano.ai.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class AttentionPartitionGradientTest {

  @DataProvider
  public Object[][] partitions() {
    return new Object[][] {{2, -1}, {3, -1}, {3, 1}};
  }

  @Test(groups = "native", dataProvider = "partitions")
  public void splitMatchesSlicesThroughAttentionHeadLayout(int parts, int unusedPart) {
    int batch = 2;
    int tokens = 5;
    int heads = 4;
    int headSize = 3;
    int width = heads * headSize;
    float[] values = new float[batch * tokens * parts * width];
    for (int index = 0; index < values.length; index++) {
      values[index] = ((index % 29) - 14) * 0.01f;
    }
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDArray splitInput = manager.create(values, new Shape(batch, tokens, parts * width));
        NDArray sliceInput = splitInput.duplicate()) {
      splitInput.setRequiresGradient(true);
      sliceInput.setRequiresGradient(true);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDList split = splitInput.split(parts, 2);
        NDArray splitLoss = null;
        NDArray sliceLoss = null;
        for (int part = 0; part < parts; part++) {
          if (part == unusedPart) {
            continue;
          }
          NDArray actual = split.get(part).reshape(batch, tokens, heads, headSize).swapAxes(1, 2);
          NDArray expected =
              sliceInput
                  .get("...,{}:{}", part * width, (part + 1) * width)
                  .reshape(batch, tokens, heads, headSize)
                  .swapAxes(1, 2);
          Assert.assertEquals(actual.toFloatArray(), expected.toFloatArray());
          NDArray actualTerm = actual.square().mul(part + 1).sum();
          NDArray expectedTerm = expected.square().mul(part + 1).sum();
          splitLoss = splitLoss == null ? actualTerm : splitLoss.add(actualTerm);
          sliceLoss = sliceLoss == null ? expectedTerm : sliceLoss.add(expectedTerm);
        }
        collector.backward(splitLoss.add(sliceLoss));
      }
      try (NDArray actualGradient = splitInput.getGradient();
          NDArray expectedGradient = sliceInput.getGradient()) {
        Assert.assertEquals(actualGradient.toFloatArray(), expectedGradient.toFloatArray());
      }
    }
  }
}

package com.epsilon.ai.decision;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DecisionBranchLossTest {
  @Test(groups = "native")
  public void detachmentPreservesValuesAndOnlyBlocksTargetGate() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      NDArray scores = manager.ones(new Shape(3, 10));
      scores.setRequiresGradient(true);
      NDArray targets =
          manager.create(new float[][] {{3, 0, 0, 0, 0}, {1, .5f, 1, -1, 1}, {0, 0, 0, 0, 0}});
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray detached = DecisionBranchLoss.detachComparedGates(scores, targets);
        Assert.assertEquals(detached.toFloatArray(), scores.toFloatArray());
        collector.backward(detached.sum());
      }
      for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 10; column++) {
          float expected = row == 0 && column == 2 || row == 1 && column == 0 ? 0 : 1;
          Assert.assertEquals(scores.getGradient().getFloat(row, column), expected);
        }
      }
    }
  }

  @Test(groups = "native")
  public void pairedGradientUsesDifferenceAndUnfinishedRowsHaveZeroGradient() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      NDArray scores = manager.zeros(new Shape(2, 10));
      scores.setRequiresGradient(true);
      NDArray targets = manager.create(new float[][] {{3, .5f, 1, -1, 1}, {3, 0, 0, 0, 0}});
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        collector.backward(
            DecisionBranchLoss.loss(scores, targets, manager.ones(new Shape(2)), .2f));
      }
      for (int row = 0; row < 2; row++) {
        for (int column = 0; column < 10; column++) {
          Assert.assertEquals(
              scores.getGradient().getFloat(row, column),
              row == 0 && column == 2 ? -.5f : 0,
              1e-6f);
        }
      }
    }
  }
}

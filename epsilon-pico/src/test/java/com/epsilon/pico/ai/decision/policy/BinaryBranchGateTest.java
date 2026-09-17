package com.epsilon.pico.ai.decision.policy;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class BinaryBranchGateTest {
  @DataProvider
  public Object[][] widths() {
    return new Object[][] {{8}, {128}};
  }

  /** Pico の標準幅とテスト用の小さな幅で、有効な候補だけを採点した結果が、全候補を採点した結果と値・勾配ともに一致することを検証する。 */
  @Test(groups = "native", dataProvider = "widths")
  public void compactActionsPreserveDenseScoresAndInputGradients(int width) {
    try (Model model = Model.newInstance("binary_gate", Device.cpu(), "PyTorch")) {
      var gate = new EpsilonBinaryBranchGate(width);
      model.setBlock(gate);
      NDManager manager = model.getNDManager();
      gate.initialize(manager, DataType.FLOAT32, new Shape(-1, width));
      ParameterStore parameters = new ParameterStore(manager, true);
      NDList compactInputs = inputs(manager, width);
      NDList denseInputs = inputs(manager, width);
      NDArray active = manager.create(new int[] {0, 2, 4});
      NDArray mask = manager.create(new float[] {1, 0, 1, 0, 1, 0}).reshape(2, 3);
      NDArray compact;
      NDArray dense;
      try (var collector = manager.getEngine().newGradientCollector()) {
        compact =
            gate.scoreActions(
                parameters,
                compactInputs.get(0),
                compactInputs.get(1),
                compactInputs.get(2),
                active,
                2,
                3,
                true,
                new PairList<>());
        dense =
            gate.scoreBroadcastState(
                    parameters,
                    denseInputs.get(0),
                    denseInputs.get(1),
                    denseInputs.get(2),
                    true,
                    new PairList<>())
                .reshape(2, 3)
                .mul(mask);
        collector.backward(compact.sum().add(dense.sum()));
      }
      Assert.assertEquals(compact.toFloatArray(), dense.toFloatArray(), 1e-5f);
      for (int i = 0; i < compactInputs.size(); i++) {
        NDArray gradient = compactInputs.get(i).getGradient();
        Assert.assertEquals(
            gradient.toFloatArray(), denseInputs.get(i).getGradient().toFloatArray(), 1e-5f);
        Assert.assertTrue(gradient.abs().sum().getFloat() > 1e-7f);
      }
      for (var parameter : gate.getParameters().values()) {
        Assert.assertTrue(parameter.getArray().getGradient().abs().sum().getFloat() > 1e-7f);
      }
    }
  }

  private static NDList inputs(NDManager manager, int width) {
    NDList inputs = new NDList();
    for (int component = 0; component < 3; component++) {
      int candidates = component == 0 ? 1 : 3;
      float[] values = new float[2 * candidates * width];
      for (int i = 0; i < values.length; i++) values[i] = ((i + component * 3) % 17 - 8) * 0.02f;
      NDArray input = manager.create(values).reshape(2, candidates, width);
      input.setRequiresGradient(true);
      inputs.add(input);
    }
    return inputs;
  }
}

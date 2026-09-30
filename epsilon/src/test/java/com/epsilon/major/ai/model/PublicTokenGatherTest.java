package com.epsilon.major.ai.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.core.GameState;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class PublicTokenGatherTest {

  @DataProvider
  public Object[][] precisions() {
    return new Object[][] {
      {Device.cpu(), DataType.FLOAT32},
      {Device.gpu(), DataType.FLOAT32},
      {Device.gpu(), DataType.BFLOAT16}
    };
  }

  /** 河と面子の並び、家の境界、複数行の出力と逆伝播を従来の分割・連結と比較する。 */
  @Test(groups = "rocm", dataProvider = "precisions")
  public void matchesSplitConcatenationAndGradient(Device device, DataType dataType) {
    for (int rows : new int[] {1, 3}) {
      try (NDManager manager = NDManager.newBaseManager(device, "PyTorch")) {
        Shape shape =
            new Shape(
                rows,
                GameState.NUM_PLAYERS,
                EpsilonMahjongStateEncoder.PLAYER_MEMORY_TOKEN_COUNT,
                8);
        float[] values = new float[Math.toIntExact(shape.size())];
        for (int index = 0; index < values.length; index++) {
          values[index] = (index % 251 - 125) * 0.125f;
        }
        NDArray actualInput = manager.create(values, shape).toType(dataType, false);
        NDArray expectedInput = actualInput.duplicate();
        NDArray indices = manager.create(EpsilonMahjongStateEncoder.publicTokenIndices());
        actualInput.setRequiresGradient(true);
        expectedInput.setRequiresGradient(true);
        try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
          NDArray actual = EpsilonMahjongStateEncoder.gatherPublicTokens(actualInput, indices);
          NDList sections =
              expectedInput.split(
                  new long[] {1, 1 + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER}, 2);
          NDArray expected =
              NDArrays.concat(
                  new NDList(
                      sections.get(1).reshape(rows, -1, 8), sections.get(2).reshape(rows, -1, 8)),
                  1);
          Assert.assertEquals(actual.getShape(), expected.getShape());
          Assert.assertEquals(actual.getDataType(), dataType);
          Assert.assertEquals(floats(actual), floats(expected));
          NDArray weights = manager.arange(actual.size()).mod(17).reshape(actual.getShape());
          collector.backward(actual.mul(weights).sum().add(expected.mul(weights).sum()));
        }
        Assert.assertEquals(floats(actualInput.getGradient()), floats(expectedInput.getGradient()));
      }
    }
  }

  private static float[] floats(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }
}

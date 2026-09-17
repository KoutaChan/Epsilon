package com.epsilon.ai.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class EpsilonLinearTest {
  /** 学習の入力・重み・バイアスへ既存Linearと同じ解析的な勾配を返す。 */
  @Test(groups = "native")
  public void trainingKeepsInputWeightAndBiasGradients() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Linear linear = initializedLinear(manager, 2, 1);
      NDArray weight = linear.getDirectParameters().get("weight").getArray();
      NDArray bias = linear.getDirectParameters().get("bias").getArray();
      weight.set(new float[] {2.0f, -1.0f});
      bias.set(new float[] {0.5f});
      NDArray input = manager.create(new float[] {3.0f, 4.0f, 5.0f, -2.0f}, new Shape(2, 1, 2));
      input.setRequiresGradient(true);
      NDArray output;
      try (var collector = manager.getEngine().newGradientCollector()) {
        output =
            EpsilonLinear.projectBatchSlices(
                linear, new ParameterStore(manager, true), input, true, new PairList<>());
        collector.backward(output.sum());
      }
      Assert.assertEquals(output.toFloatArray(), new float[] {2.5f, 12.5f});
      Assert.assertEquals(input.getGradient().toFloatArray(), new float[] {2, -1, 2, -1});
      Assert.assertEquals(weight.getGradient().toFloatArray(), new float[] {8, 2});
      Assert.assertEquals(bias.getGradient().toFloatArray(), new float[] {2});
      Assert.assertEquals(input.toFloatArray(), new float[] {3, 4, 5, -2});
      Assert.assertEquals(weight.toFloatArray(), new float[] {2, -1});
      Assert.assertEquals(bias.toFloatArray(), new float[] {0.5f});
    }
  }

  /** CPU推論は元のLinearを使い、共有パラメーターをバッチの寿命へ移さない。 */
  @Test(groups = "native")
  public void cpuInferenceRetainsTheOriginalParameterOwner() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Linear linear = initializedLinear(manager, 2, 1);
      NDArray weight = linear.getDirectParameters().get("weight").getArray();
      NDArray bias = linear.getDirectParameters().get("bias").getArray();
      weight.set(new float[] {2, -1});
      bias.set(new float[] {0.5f});
      for (int repetition = 0; repetition < 3; repetition++) {
        try (NDManager batch = manager.newSubManager()) {
          NDArray input = batch.create(new float[] {3, 4}, new Shape(1, 1, 2));
          NDArray output =
              EpsilonLinear.projectBatchSlices(
                  linear, new ParameterStore(batch, false), input, false, new PairList<>());
          Assert.assertEquals(output.toFloatArray(), new float[] {2.5f});
        }
        Assert.assertSame(weight.getManager(), manager);
        Assert.assertSame(bias.getManager(), manager);
        Assert.assertEquals(weight.toFloatArray(), new float[] {2, -1});
        Assert.assertEquals(bias.toFloatArray(), new float[] {0.5f});
      }
    }
  }

  @DataProvider
  public Object[][] parameterTypes() {
    return new Object[][] {{false, 1}, {false, 2}, {false, 3}, {true, 1}, {true, 2}, {true, 3}};
  }

  /**
   * 複数バッチはBF16の中点を二回丸める旧非連続Linearと一致し、単バッチは元の融合バイアスと一致する。 FP32で保持する学習用パラメーター
   * バイアスの一時変換はバッチ終了時に解放し、元パラメーターを保持する。
   */
  @Test(groups = "rocm", dataProvider = "parameterTypes")
  public void gpuInferencePreservesProductRoundingAndParameterOwnership(
      boolean frozenBfloat16, int rows) {
    Device device = Device.gpu();
    int inputWidth = 128;
    int outputWidth = 96;
    try (NDManager manager = NDManager.newBaseManager(device)) {
      Linear linear = initializedLinear(manager, inputWidth, outputWidth);
      float[] weights = new float[inputWidth * outputWidth];
      float[] biases = new float[outputWidth];
      for (int unit = 0; unit < outputWidth; unit++) {
        weights[unit * inputWidth] = 1.0f;
        weights[unit * inputWidth + 1] = 1.0f / 256;
      }
      Arrays.fill(biases, 1.0f / 256);
      linear.getDirectParameters().get("weight").getArray().set(weights);
      linear.getDirectParameters().get("bias").getArray().set(biases);
      if (frozenBfloat16) {
        linear.getDirectParameters().get("weight").castArray(DataType.BFLOAT16);
        linear.getDirectParameters().get("bias").castArray(DataType.BFLOAT16);
      }
      NDArray weight = linear.getDirectParameters().get("weight").getArray();
      NDArray bias = linear.getDirectParameters().get("bias").getArray();
      float[] inputValues = new float[rows * 34 * inputWidth];
      for (int row = 0; row < rows * 34; row++) {
        inputValues[row * inputWidth] = 1;
        inputValues[row * inputWidth + 1] = 1;
      }
      for (int repetition = 0; repetition < 3; repetition++) {
        try (NDManager batch = manager.newSubManager();
            var inference = manager.getEngine().newInferenceMode();
            var autocast = manager.getEngine().newAutocast(device, DataType.BFLOAT16, true)) {
          NDArray input =
              batch
                  .create(inputValues, new Shape(rows, 34, inputWidth))
                  .toType(DataType.BFLOAT16, false);
          NDArray padded =
              NDArrays.concat(
                  new NDList(batch.zeros(new Shape(rows, 1, inputWidth), DataType.BFLOAT16), input),
                  1);
          NDArray oldStrided = padded.get(":,1:35,:");
          var store = new ParameterStore(batch, false);
          NDArray old =
              linear
                  .forward(store, new NDList(oldStrided), false, new PairList<>())
                  .singletonOrThrow();
          NDArray actual =
              EpsilonLinear.projectBatchSlices(linear, store, input, false, new PairList<>());
          NDArray fused = Linear.linear(input, weight, bias).singletonOrThrow();
          Assert.assertEquals(actual.getDataType(), DataType.BFLOAT16);
          float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
          Assert.assertEquals(actualValues, old.toType(DataType.FLOAT32, false).toFloatArray());
          for (float value : actualValues)
            Assert.assertEquals(value, rows == 1 ? 1.0f + 1.0f / 128 : 1.0f);
          for (float value : fused.toType(DataType.FLOAT32, false).toFloatArray())
            Assert.assertEquals(value, 1.0f + 1.0f / 128);
        }
        Assert.assertSame(weight.getManager(), manager);
        Assert.assertSame(bias.getManager(), manager);
        Assert.assertEquals(weight.toType(DataType.FLOAT32, false).toFloatArray(), weights);
        Assert.assertEquals(bias.toType(DataType.FLOAT32, false).toFloatArray(), biases);
        Assert.assertEquals(
            bias.getDataType(), frozenBfloat16 ? DataType.BFLOAT16 : DataType.FLOAT32);
      }
    }
  }

  private static Linear initializedLinear(NDManager manager, int inputWidth, int outputWidth) {
    Linear linear = Linear.builder().setUnits(outputWidth).build();
    linear.initialize(manager, DataType.FLOAT32, new Shape(-1, 34, inputWidth));
    return linear;
  }
}

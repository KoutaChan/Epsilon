package com.epsilon.major.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import org.testng.Assert;
import org.testng.annotations.Test;

public final class EpsilonFeatureFusionTypedConcatTest {

  @Test(groups = "native")
  public void float32InferenceMatchesOriginalConcatenation() {
    assertMatchesOriginal(Device.cpu(), DataType.FLOAT32);
  }

  @Test(groups = "rocm")
  public void mixedInputsMatchOriginalBfloat16Autocast() {
    assertMatchesOriginal(Device.gpu(), DataType.BFLOAT16);
  }

  private static void assertMatchesOriginal(Device device, DataType parameterType) {
    Engine engine = Engine.getEngine("PyTorch");
    engine.setRandomSeed(5731);
    try (NDManager manager = engine.newBaseManager(device);
        var inference = engine.newInferenceMode()) {
      EpsilonFeatureFusion fusion = new EpsilonFeatureFusion(32, 4, 17, 13, 7, 9, 5);
      fusion.initialize(manager, parameterType, new Shape(3, 4, 1, 17));
      fusion.freezeParameters(true);
      ParameterStore store = new ParameterStore(manager, false);
      NDArray[] components = new NDArray[fusion.componentCount()];
      for (int component = 0; component < components.length; component++) {
        DataType inputType = component == 1 || component == 2 ? DataType.FLOAT32 : parameterType;
        components[component] =
            manager.randomUniform(
                -1,
                1,
                new Shape(3, component == 4 ? 1 : 4, 1, fusion.inputSize(component)),
                inputType);
      }
      try (var autocast =
          parameterType == DataType.FLOAT32
              ? null
              : engine.newAutocast(device, DataType.BFLOAT16, false)) {
        NDArray expected = originalFusion(fusion, store, components);
        NDArray actual = fusion.fuse(store, false, new PairList<>(), components);
        Assert.assertEquals(actual.getDataType(), parameterType);
        Assert.assertEquals(actual.getDataType(), expected.getDataType());
        Assert.assertEquals(actual.getShape(), expected.getShape());
        Assert.assertEquals(values(actual), values(expected));
      }
    }
  }

  private static NDArray originalFusion(
      EpsilonFeatureFusion fusion, ParameterStore store, NDArray[] components) {
    NDList inputs = new NDList();
    NDList weights = new NDList();
    for (int component = 0; component < fusion.fusedPrefixComponents(); component++) {
      inputs.add(components[component]);
      weights.add(
          store.getValue(
              fusion.projection(component).getDirectParameters().get("weight"),
              components[0].getDevice(),
              false));
    }
    NDArray bias =
        store.getValue(
            fusion.projection(0).getDirectParameters().get("bias"),
            components[0].getDevice(),
            false);
    NDArray expected =
        Linear.linear(NDArrays.concat(inputs, 3), NDArrays.concat(weights, 1), bias)
            .singletonOrThrow();
    for (int component = fusion.fusedPrefixComponents();
        component < fusion.componentCount();
        component++) {
      expected.addi(
          fusion
              .projection(component)
              .forward(store, new NDList(components[component]), false, new PairList<>())
              .singletonOrThrow());
    }
    return expected;
  }

  private static float[] values(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }
}

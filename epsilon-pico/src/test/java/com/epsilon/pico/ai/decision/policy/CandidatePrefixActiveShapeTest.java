package com.epsilon.pico.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.pico.ai.decision.policy.fusion.DecisionPolicyFusionCandidatePrefixExecution;
import com.epsilon.pico.ai.model.EpsilonMahjongStateEncoder;
import org.testng.Assert;
import org.testng.annotations.Test;

public final class CandidatePrefixActiveShapeTest {

  @Test(groups = "native")
  public void differentActiveShapesPreserveOutstandingOutputAndSlots() throws Exception {
    Engine engine = Engine.getEngine("PyTorch");
    engine.setRandomSeed(5731);
    try (NDManager manager = engine.newBaseManager(Device.cpu());
        var inference = engine.newInferenceMode()) {
      EpsilonFeatureFusion fusion = new EpsilonFeatureFusion(8, 4, 5, 6, 7, 4, 8);
      fusion.initialize(manager, DataType.FLOAT32, new Shape(3, 7, 5));
      fusion.freezeParameters(true);
      ParameterStore store = new ParameterStore(manager, false);
      try (var execution =
              new DecisionPolicyFusionCandidatePrefixExecution(
                  manager, store, fusion, DataType.FLOAT32, 3, 2);
          NDManager heldManager = manager.newSubManager();
          var heldForward = execution.beginForward(heldManager)) {
        NDArray[] heldInputs = inputs(heldManager, fusion, 3, 7, DataType.FLOAT32);
        NDArray held = project(heldForward, store, heldInputs);
        assertClose(held, expected(fusion, store, heldInputs), 1e-5f);
        float[] heldValues = held.toFloatArray();
        heldForward.seal();

        try (NDManager working = manager.newSubManager();
            var forward = execution.beginForward(working)) {
          Assert.assertThrows(IllegalStateException.class, () -> execution.beginForward(working));
          NDArray[] smallInputs = inputs(working, fusion, 1, 2, DataType.FLOAT32);
          assertClose(project(forward, store, smallInputs), expected(fusion, store, smallInputs), 1e-5f);
          forward.seal();
          Assert.assertEquals(held.toFloatArray(), heldValues);
        }

        try (NDManager working = manager.newSubManager();
            var forward = execution.beginForward(working)) {
          NDArray[] singleInputs = inputs(working, fusion, 2, 1, DataType.FLOAT32);
          assertClose(project(forward, store, singleInputs), expected(fusion, store, singleInputs), 1e-5f);
          forward.seal();
          Assert.assertEquals(held.toFloatArray(), heldValues);
        }
        Assert.assertFalse(execution.hasIncompleteWork());
        Assert.assertNull(execution.failure());
      }
    }
  }

  @Test(groups = "rocm")
  public void mixedInputTypesMatchEagerBfloat16Projection() throws Exception {
    Engine engine = Engine.getEngine("PyTorch");
    engine.setRandomSeed(5731);
    try (NDManager manager = engine.newBaseManager(Device.gpu());
        var inference = engine.newInferenceMode();
        var autocast = engine.newAutocast(Device.gpu(), DataType.BFLOAT16, false)) {
      EpsilonFeatureFusion fusion = new EpsilonFeatureFusion(8, 4, 5, 6, 7, 4, 8);
      fusion.initialize(manager, DataType.BFLOAT16, new Shape(2, 3, 5));
      fusion.freezeParameters(true);
      ParameterStore store = new ParameterStore(manager, false);
      try (var execution =
              new DecisionPolicyFusionCandidatePrefixExecution(
                  manager, store, fusion, DataType.BFLOAT16, 2, 1);
          NDManager working = manager.newSubManager();
          var forward = execution.beginForward(working)) {
        NDArray[] values = inputs(working, fusion, 2, 3, DataType.BFLOAT16);
        values[1] = values[1].toType(DataType.FLOAT32, false);
        values[2] = values[2].toType(DataType.FLOAT32, false);
        NDArray actual = project(forward, store, values);
        Assert.assertEquals(actual.getDataType(), DataType.BFLOAT16);
        assertClose(actual, expected(fusion, store, values), 0.02f);
        forward.seal();
      }
    }
  }

  private static NDArray[] inputs(
      NDManager manager, EpsilonFeatureFusion fusion, int rows, int actions, DataType type) {
    NDArray[] values = new NDArray[fusion.componentCount()];
    for (int index = 0; index < values.length; index++) {
      values[index] =
          manager.randomUniform(
              -1, 1, new Shape(rows, index == 4 ? 1 : actions, fusion.inputSize(index)), type);
    }
    return values;
  }

  private static NDArray project(
      DecisionPolicyCandidatePrefixExecution.Forward forward,
      ParameterStore store,
      NDArray[] inputs) {
    return forward.project(
        store,
        new PairList<>(),
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4]);
  }

  private static NDArray expected(
      EpsilonFeatureFusion fusion, ParameterStore store, NDArray[] inputs) {
    return EpsilonMahjongStateEncoder.silu(fusion.fuse(store, false, new PairList<>(), inputs));
  }

  private static void assertClose(NDArray actual, NDArray expected, float tolerance) {
    Assert.assertEquals(actual.getShape(), expected.getShape());
    float[] got = actual.toType(DataType.FLOAT32, false).toFloatArray();
    float[] want = expected.toType(DataType.FLOAT32, false).toFloatArray();
    for (int index = 0; index < want.length; index++) {
      Assert.assertEquals(got[index], want[index], tolerance, "element=" + index);
    }
  }
}

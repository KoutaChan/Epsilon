package com.epsilon.nano.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionCandidatePrefixExecution;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public final class CandidatePrefixActiveShapeTest {

  @Test(groups = "native")
  public void compactActionsPreserveProjectedStateMappingAndOwnership() throws Exception {
    verifyCompactActions(Device.cpu(), DataType.FLOAT32);
  }

  @DataProvider
  public Object[][] compactTypes() {
    return new Object[][] {{DataType.BFLOAT16}, {DataType.FLOAT16}};
  }

  @Test(groups = "rocm", dataProvider = "compactTypes")
  public void compactActionsPreserveRoundedProjection(DataType type) throws Exception {
    verifyCompactActions(Device.gpu(0), type);
  }

  private static void verifyCompactActions(Device device, DataType type) throws Exception {
    var engine = Engine.getEngine("PyTorch");
    engine.setRandomSeed(5731);
    try (var manager = engine.newBaseManager(device);
        var inference = engine.newInferenceMode();
        var autocast = type == DataType.FLOAT32 ? null : engine.newAutocast(device, type, false)) {
      int width = device.isGpu() ? 256 : 16;
      var fusion = new EpsilonFeatureFusion(width, 4, 5, 6, 7, 4, width);
      fusion.initialize(manager, type, new Shape(4, 5, 5));
      fusion.freezeParameters(true);
      var store = new ParameterStore(manager, false);
      try (var execution =
              new DecisionPolicyFusionCandidatePrefixExecution(manager, store, fusion, type, 4, 1);
          var working = manager.newSubManager()) {
        NDArray held = null;
        float[] heldValues = null;
        for (int[] selection :
            new int[][] {{0, 2, 5, 10, 11, 12}, {1, 6}, {0, 1, 3, 5, 7, 8, 10, 14}}) {
          try (var forward = execution.beginForward(working)) {
            var values = inputs(working, fusion, 4, 5, type);
            values[4].get(3).fillI(Float.NaN);
            values[0].get(3).fillI(Float.NaN);
            NDArray ids = working.create(selection);
            int[] rows = new int[selection.length];
            for (int index = 0; index < rows.length; index++) rows[index] = selection[index] / 5;
            NDArray batchIds = working.create(rows);
            NDArray expected =
                EpsilonMaskedRows.gather(expected(fusion, store, values).reshape(-1, width), ids)
                    .reshape(selection.length, 1, width);
            var compact = new NDArray[4];
            for (int component = 0; component < compact.length; component++)
              compact[component] =
                  EpsilonMaskedRows.gather(
                          values[component].reshape(-1, fusion.inputSize(component)), ids)
                      .reshape(selection.length, 1, fusion.inputSize(component));
            NDArray eager =
                EpsilonMahjongStateEncoder.silu(
                    fusion.fuseIndexedState(store, new PairList<>(), values[4], batchIds, compact));
            NDArray actual =
                forward.projectCompact(
                    store,
                    new PairList<>(),
                    compact[0],
                    compact[1],
                    compact[2],
                    compact[3],
                    values[4],
                    batchIds);
            float[] reference = expected.toType(DataType.FLOAT32, false).toFloatArray();
            TransitionStateProjectionTest.assertValues(
                eager.toType(DataType.FLOAT32, false).toFloatArray(),
                reference,
                type,
                "compact eager");
            TransitionStateProjectionTest.assertValues(
                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                reference,
                type,
                "compact prefix");
            Assert.assertEquals(actual.getShape(), new Shape(selection.length, 1, width));
            Assert.assertEquals(actual.getDataType(), type);
            if (held != null)
              Assert.assertEquals(held.toType(DataType.FLOAT32, false).toFloatArray(), heldValues);
            held = actual;
            heldValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
            Assert.assertThrows(
                IllegalStateException.class,
                () ->
                    forward.projectCompact(
                        store, new PairList<>(), null, null, null, null, null, null));
            forward.seal();
            Assert.assertThrows(
                IllegalStateException.class,
                () ->
                    forward.projectCompact(
                        store, new PairList<>(), null, null, null, null, null, null));
          }
        }
      }
    }
  }

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
        NDArray[] another = inputs(heldManager, fusion, 1, 4, DataType.FLOAT32);
        Assert.assertThrows(
            IllegalStateException.class, () -> project(heldForward, store, another));
        heldForward.seal();
        Assert.assertThrows(
            IllegalStateException.class, () -> project(heldForward, store, another));

        try (NDManager working = manager.newSubManager();
            var forward = execution.beginForward(working)) {
          Assert.assertThrows(IllegalStateException.class, () -> execution.beginForward(working));
          NDArray[] smallInputs = inputs(working, fusion, 1, 2, DataType.FLOAT32);
          assertClose(
              project(forward, store, smallInputs), expected(fusion, store, smallInputs), 1e-5f);
          forward.seal();
          Assert.assertEquals(held.toFloatArray(), heldValues);
        }

        try (NDManager working = manager.newSubManager();
            var forward = execution.beginForward(working)) {
          NDArray[] singleInputs = inputs(working, fusion, 2, 1, DataType.FLOAT32);
          assertClose(
              project(forward, store, singleInputs), expected(fusion, store, singleInputs), 1e-5f);
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
        float[] held = actual.toType(DataType.FLOAT32, false).toFloatArray();
        NDArray[] smaller = inputs(working, fusion, 1, 1, DataType.BFLOAT16);
        Assert.assertThrows(IllegalStateException.class, () -> project(forward, store, smaller));
        Assert.assertEquals(actual.toType(DataType.FLOAT32, false).toFloatArray(), held);
        forward.seal();
        Assert.assertThrows(IllegalStateException.class, () -> project(forward, store, smaller));
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
        store, new PairList<>(), inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
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

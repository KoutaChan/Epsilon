package com.epsilon.nano.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public final class TransitionStateProjectionTest {
  @Test(groups = "native")
  public void projectedStatesPreserveRowMappingWeightsAndRepeatedExecution() {
    verify(Device.cpu(), DataType.FLOAT32, false);
  }

  @DataProvider
  public Object[][] computeTypes() {
    return new Object[][] {
      {DataType.BFLOAT16, false}, {DataType.FLOAT16, false}, {DataType.FLOAT16, true}
    };
  }

  @Test(groups = "rocm", dataProvider = "computeTypes")
  public void projectedStatesPreserveFrozenRounding(DataType type, boolean fusedActivation) {
    if (Engine.getEngine("PyTorch").getGpuCount() == 0)
      throw new SkipException("This test requires a ROCm GPU");
    verify(Device.gpu(0), type, fusedActivation);
  }

  @Test(groups = "native")
  public void stateAdditionPreservesCpuActivationAndResidual() {
    verifyStateAddition(Device.cpu(), DataType.FLOAT32);
  }

  @DataProvider
  public Object[][] residualTypes() {
    return new Object[][] {{DataType.FLOAT32}, {DataType.BFLOAT16}, {DataType.FLOAT16}};
  }

  @Test(groups = "rocm", dataProvider = "residualTypes")
  public void stateAdditionPreservesOwnedBufferAndRoundedActivation(DataType type) {
    if (Engine.getEngine("PyTorch").getGpuCount() == 0)
      throw new SkipException("This test requires a ROCm GPU");
    verifyStateAddition(Device.gpu(0), type);
  }

  private static void verifyStateAddition(Device device, DataType type) {
    var engine = Engine.getEngine("PyTorch");
    try (var manager = engine.newBaseManager(device);
        var inference = engine.newInferenceMode()) {
      int width = 16;
      float unit = type == DataType.BFLOAT16 ? 1f / 128 : 1f / 1024;
      float[] pattern = {1, -1, .25f, -.25f, 3, -3, .0625f, -.0625f};
      for (int rows : new int[] {1, 3, 37}) {
        try (var scope = manager.newSubManager()) {
          float[] values = new float[rows * width], residual = new float[values.length];
          for (int index = 0; index < values.length; index++) {
            values[index] = pattern[index % pattern.length];
            // 加算後に入力型へ丸める位置が変わると、低精度の半ULP付近で差が出る。
            residual[index] = unit * (index % 3 == 0 ? .5f : index % 3 == 1 ? -.5f : 1.5f);
          }
          Shape shape = new Shape(rows, 1, 1, width);
          NDArray projected = scope.create(values, shape).toType(type, false);
          NDArray state = scope.create(residual, shape).toType(type, false);
          float[] savedState = values(state);
          NDArray added = projected.add(state);
          float[] expectedAdded = values(added);
          float[] expected = values(EpsilonMahjongStateEncoder.silu(added));
          NDArray actual =
              DecisionPolicyAffineExecution.addStateProjectionAndActivate(projected, state, width);
          Assert.assertEquals(actual.getShape(), shape);
          Assert.assertEquals(actual.getDataType(), type);
          if (device.isGpu()) Assert.assertSame(actual, projected);
          else Assert.assertEquals(values(projected), expectedAdded);
          Assert.assertEquals(values(state), savedState, "The shared state must not be modified");
          Assert.assertEquals(
              values(actual),
              expected,
              type == DataType.FLOAT32 ? 2e-4f : 0f,
              "Rounded state addition " + type + " rows=" + rows);
        }
      }
    }
  }

  private static void verify(Device device, DataType type, boolean fusedActivation) {
    var engine = Engine.getEngine("PyTorch");
    engine.setRandomSeed(9421);
    try (var manager = engine.newBaseManager(device);
        var inference = engine.newInferenceMode()) {
      int width = device.isGpu() ? 256 : 16;
      var fusion = new EpsilonFeatureFusion(width, 4, width, 16, width, 16, width);
      fusion.initialize(manager, type, new Shape(4, 1, 1, width));
      fusion.freezeParameters(true);
      var store = new ParameterStore(manager, false);
      var before = parameters(fusion);
      try (var projection =
          new DecisionPolicyAffineExecution.PackedTransitionProjection(
              manager, store, fusion, type, fusedActivation)) {
        for (int count : new int[] {37, 3, 1, 61, 37}) {
          try (var scope = manager.newSubManager();
              var autocast =
                  type == DataType.FLOAT32 ? null : engine.newAutocast(device, type, false)) {
            NDArray states =
                scope.randomUniform(
                    -1, 1, new Shape(4, width), count == 3 ? DataType.FLOAT32 : type);
            // 未参照の盤面はNaNでも、他の盤面へ漏れず同じ対応行だけが使われる。
            states.get(3).fillI(Float.NaN);
            int[] ids = new int[count];
            for (int index = 0; index < count; index++) ids[index] = (index % 5) % 3;
            NDArray stateRows = scope.create(ids).reshape(1, count);
            NDArray[] components = new NDArray[fusion.componentCount()];
            for (int index = 0; index < components.length - 1; index++)
              components[index] =
                  scope.randomUniform(
                      -1,
                      1,
                      new Shape(count, 1, 1, fusion.inputSize(index)),
                      index == 1 || index == 2 ? DataType.FLOAT32 : type);
            components[4] = EpsilonMaskedRows.gather(states, stateRows).reshape(count, 1, 1, width);
            float[] expected =
                values(
                    EpsilonMahjongStateEncoder.silu(
                        fusion.fuse(
                            new ParameterStore(scope, false),
                            false,
                            new PairList<>(),
                            components)));
            components[4] = states;
            NDArray actual = projection.project(stateRows, components);
            Assert.assertEquals(actual.getShape(), new Shape(count, 1, 1, width));
            Assert.assertEquals(actual.getDataType(), type);
            assertValues(
                values(actual), expected, type, "rows=" + count + " fused=" + fusedActivation);
          }
        }
        projection.close();
        Assert.expectThrows(IllegalStateException.class, () -> projection.project(null));
      }
      var after = parameters(fusion);
      Assert.assertEquals(after.size(), before.size());
      for (int index = 0; index < before.size(); index++)
        Assert.assertEquals(after.get(index), before.get(index), "Frozen parameter " + index);
    }
  }

  private static List<float[]> parameters(EpsilonFeatureFusion fusion) {
    List<float[]> result = new ArrayList<>();
    for (var parameter : fusion.getParameters())
      result.add(values(parameter.getValue().getArray()));
    return result;
  }

  private static float[] values(NDArray value) {
    return value.toType(DataType.FLOAT32, false).toFloatArray();
  }

  static void assertValues(float[] actual, float[] expected, DataType type, String label) {
    if (type == DataType.FLOAT32) {
      Assert.assertEquals(actual, expected, 2e-4f, label);
      return;
    }
    double squaredError = 0, expectedSquares = 0, actualSquares = 0, dot = 0;
    double maximumError = 0, maximumExpected = 0;
    for (int index = 0; index < expected.length; index++) {
      Assert.assertTrue(Float.isFinite(actual[index]), label + " must be finite");
      double reference = expected[index], value = actual[index], difference = value - reference;
      squaredError += difference * difference;
      expectedSquares += reference * reference;
      actualSquares += value * value;
      dot += value * reference;
      maximumError = Math.max(maximumError, Math.abs(difference));
      maximumExpected = Math.max(maximumExpected, Math.abs(reference));
    }
    double relativeRms = Math.sqrt(squaredError / Math.max(expectedSquares, 1e-30));
    double cosine = dot / Math.sqrt(Math.max(expectedSquares * actualSquares, 1e-30));
    double resolution = type == DataType.BFLOAT16 ? 1d / 128 : 1d / 1024;
    String metrics =
        type.name()
            + " "
            + label
            + " maxAbs="
            + maximumError
            + " relativeRms="
            + relativeRms
            + " cosine="
            + cosine;
    System.out.println(metrics);
    // 共有特徴テストと同じ基準で、型の相対精度2単位と最大座標4単位を確認する。
    Assert.assertTrue(relativeRms <= 2 * resolution, metrics);
    Assert.assertTrue(cosine >= .9998, metrics);
    Assert.assertTrue(maximumError <= Math.max(1, maximumExpected) * 4 * resolution, metrics);
  }
}

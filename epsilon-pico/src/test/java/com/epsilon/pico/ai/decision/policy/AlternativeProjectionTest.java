package com.epsilon.pico.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AlternativeProjectionTest {
  /** 分岐オフセットを活性化前に加え、状態・候補・オフセットへ参照式どおりに勾配を流す。 */
  @Test(groups = "native")
  public void directOffsetsMatchDenseProjectionAndItsPerBranchDerivative() {
    int width = 4;
    int alternatives = DecisionAlternative.NETWORK_SIZE;
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      var head = new EpsilonDecisionPolicyHead(width);
      head.initialize(manager, DataType.FLOAT32, new Shape(-1, width));
      Linear hidden = alternativeHidden(head);
      Parameter offsets = head.getDirectParameters().get("alternativeOffsets");
      float[] offsetValues = new float[alternatives * width];
      for (int i = 0; i < offsetValues.length; i++) offsetValues[i] = i * 0.01f - 0.2f;
      offsets.getArray().set(offsetValues);
      float[] weights = new float[width * width * 2];
      for (int i = 0; i < weights.length; i++) weights[i] = (i % 7 - 3) * 0.03f;
      hidden.getDirectParameters().get("weight").getArray().set(weights);
      NDArray state = manager.ones(new Shape(2, 1, width)).mul(0.3f);
      NDArray candidates = manager.ones(new Shape(2, alternatives, width)).mul(0.2f);
      state.setRequiresGradient(true);
      candidates.setRequiresGradient(true);
      NDArray dense =
          Linear.linear(
                  state.broadcast(2, alternatives, width).concat(candidates, 2),
                  hidden.getDirectParameters().get("weight").getArray(),
                  null)
              .singletonOrThrow()
              .add(offsets.getArray());
      NDArray expected = dense.mul(Activation.sigmoid(dense));
      NDArray actual;
      try (var collector = manager.getEngine().newGradientCollector()) {
        actual =
            DecisionPolicyAffineExecution.alternativeContexts(
                hidden, offsets, new ParameterStore(manager, true), true, state, candidates);
        collector.backward(actual.sum());
      }
      Assert.assertEquals(actual.toFloatArray(), expected.toFloatArray(), 1e-5f);
      Assert.assertTrue(state.getGradient().abs().sum().getFloat() > 1e-6f);
      Assert.assertTrue(candidates.getGradient().abs().sum().getFloat() > 1e-6f);
      Assert.assertTrue(
          hidden.getDirectParameters().get("weight").getArray().getGradient().abs().sum().getFloat()
              > 1e-6f);
      NDArray sigmoid = Activation.sigmoid(dense);
      NDArray expectedOffsetGradient =
          sigmoid.add(dense.mul(sigmoid).mul(sigmoid.neg().add(1))).sum(new int[] {0});
      Assert.assertEquals(
          offsets.getArray().getGradient().toFloatArray(),
          expectedOffsetGradient.toFloatArray(),
          1e-5f);
    }
  }

  /** BF16 の一時オフセットをバッチの終了時に解放し、FP32 の学習パラメータとその勾配を維持する。 */
  @Test(groups = "native")
  public void mixedPrecisionBatchesReleaseConvertedOffsetsAndRetainMasterGradients() {
    int width = 4;
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      var head = new EpsilonDecisionPolicyHead(width);
      head.initialize(manager, DataType.FLOAT32, new Shape(-1, width));
      Linear hidden = alternativeHidden(head);
      Parameter offsets = head.getDirectParameters().get("alternativeOffsets");
      int retainedArrays = manager.getManagedArrays().size();
      for (int i = 0; i < 3; i++) {
        try (NDManager batch = manager.newSubManager();
            var collector = manager.getEngine().newGradientCollector()) {
          batch.tempAttachAll(hidden.getDirectParameters().get("weight").getArray());
          NDArray state = batch.ones(new Shape(2, 1, width));
          NDArray candidates = batch.ones(new Shape(2, DecisionAlternative.NETWORK_SIZE, width));
          NDArray output;
          try (Autocast ignored =
              manager.getEngine().newAutocast(Device.cpu(), DataType.BFLOAT16, true)) {
            output =
                DecisionPolicyAffineExecution.alternativeContexts(
                    hidden, offsets, new ParameterStore(batch, true), true, state, candidates);
            Assert.assertEquals(output.getDataType(), DataType.BFLOAT16);
          }
          collector.backward(output.sum());
        }
        Assert.assertEquals(manager.getManagedArrays().size(), retainedArrays);
        Assert.assertEquals(offsets.getArray().getDataType(), DataType.FLOAT32);
      }
      Assert.assertTrue(offsets.getArray().getGradient().abs().sum().getFloat() > 1e-6f);
    }
  }

  private static Linear alternativeHidden(EpsilonDecisionPolicyHead head) {
    for (var child : head.getChildren()) {
      if (child.getKey().endsWith("alternativeHidden")) return (Linear) child.getValue();
    }
    throw new AssertionError("alternative projection is absent");
  }
}

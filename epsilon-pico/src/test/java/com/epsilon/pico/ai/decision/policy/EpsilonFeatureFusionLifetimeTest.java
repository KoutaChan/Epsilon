package com.epsilon.pico.ai.decision.policy;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import org.testng.Assert;
import org.testng.annotations.Test;

public final class EpsilonFeatureFusionLifetimeTest {
  @Test(groups = "native")
  public void releasedPackingPreservesForwardAndBackward() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      EpsilonFeatureFusion fusion = new EpsilonFeatureFusion(5, 2, 3, 2);
      fusion.initialize(manager, DataType.FLOAT32, new Shape(2, 3), new Shape(2, 2));
      NDArray first = manager.create(new float[][] {{1, -2, 3}, {-4, 5, 6}});
      NDArray second = manager.create(new float[][] {{2, 3}, {-1, 4}});
      first.setRequiresGradient(true);
      second.setRequiresGradient(true);
      ParameterStore store = new ParameterStore(manager, true);
      float[] expected;
      float[] firstGradient;
      float[] secondGradient;
      float[][] parameterGradients = new float[fusion.getParameters().size()][];
      try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
        NDArray reference =
            first
                .matMul(fusion.projection(0).getParameters().get("weight").getArray().transpose())
                .add(
                    second.matMul(
                        fusion.projection(1).getParameters().get("weight").getArray().transpose()))
                .add(fusion.projection(0).getParameters().get("bias").getArray());
        expected = reference.toFloatArray();
        collector.backward(reference.square().sum());
        firstGradient = first.getGradient().toFloatArray();
        secondGradient = second.getGradient().toFloatArray();
        for (int i = 0; i < parameterGradients.length; i++) {
          parameterGradients[i] =
              fusion.getParameters().valueAt(i).getArray().getGradient().toFloatArray();
        }
      }
      first.getGradient().fillI(0);
      second.getGradient().fillI(0);
      for (var parameter : fusion.getParameters())
        parameter.getValue().getArray().getGradient().fillI(0);
      try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
        NDArray actual = fusion.fuse(store, true, new PairList<>(), first, second);
        assertClose(actual.toFloatArray(), expected);
        collector.backward(actual.square().sum());
        assertClose(first.getGradient().toFloatArray(), firstGradient);
        assertClose(second.getGradient().toFloatArray(), secondGradient);
        for (int i = 0; i < parameterGradients.length; i++) {
          assertClose(
              fusion.getParameters().valueAt(i).getArray().getGradient().toFloatArray(),
              parameterGradients[i]);
        }
      }
    }
  }

  private static void assertClose(float[] actual, float[] expected) {
    Assert.assertEquals(actual.length, expected.length);
    for (int i = 0; i < actual.length; i++)
      Assert.assertEquals(actual[i], expected[i], 1e-4f, "element=" + i);
  }
}

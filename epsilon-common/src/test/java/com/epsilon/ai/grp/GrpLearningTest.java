package com.epsilon.ai.grp;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 保存したAdamWの状態を復元すると、学習を中断せず続けた場合と同じ次の更新を得られることを検証する。 */
public class GrpLearningTest {
  @Test(groups = "native")
  public void resumedAdamStateProducesTheSameNextUpdate() throws Exception {
    Path directory = Files.createTempDirectory("grp-resume-");
    var samples =
        List.of(
            new EpsilonGrpExample(
                7,
                EpsilonGrpFeature.fromProgress(0, 0, 0, new int[] {25000, 25000, 25000, 25000}),
                0));
    try (Model continuous = model();
        Model resumed = model();
        var originalTrainer = trainer(continuous)) {
      float initialNll =
          new EpsilonGrpInference(continuous, 1).evaluateExamples(samples).marginalNll();
      originalTrainer.trainExamples(samples, 3);
      continuous.save(directory, "grp");
      originalTrainer.saveOptimizerState(directory.resolve("optimizer.state"));
      resumed.load(directory, "grp");
      try (var restoredTrainer = trainer(resumed)) {
        restoredTrainer.loadOptimizerState(directory.resolve("optimizer.state"));
        originalTrainer.trainExamples(samples, 1);
        restoredTrainer.trainExamples(samples, 1);
      }
      var expected = continuous.getBlock().getParameters();
      var actual = resumed.getBlock().getParameters();
      Assert.assertEquals(actual.keys(), expected.keys());
      for (int parameter = 0; parameter < expected.size(); parameter++) {
        float[] left = expected.valueAt(parameter).getArray().toFloatArray();
        float[] right = actual.valueAt(parameter).getArray().toFloatArray();
        Assert.assertEquals(right.length, left.length);
        for (int i = 0; i < left.length; i++) Assert.assertEquals(right[i], left[i], 1e-6f);
      }
      Assert.assertTrue(
          new EpsilonGrpInference(continuous, 1).evaluateExamples(samples).marginalNll()
              < initialNll);
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  private static Model model() {
    Model model = Model.newInstance("grp", Device.cpu(), "PyTorch");
    var block = new EpsilonGrpNetwork(16, 1);
    model.setBlock(block);
    block.initialize(
        model.getNDManager(),
        DataType.FLOAT32,
        new Shape(-1, -1, EpsilonGrpFeature.FEATURE_SIZE),
        new Shape(-1));
    return model;
  }

  private static EpsilonGrpTrainer trainer(Model model) {
    return new EpsilonGrpTrainer(model, 1, .001f, .0001f, 1f);
  }
}

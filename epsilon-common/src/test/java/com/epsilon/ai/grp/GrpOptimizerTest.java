package com.epsilon.ai.grp;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** パラメーターとの対応を一意に復元できない旧形式のAdamW状態を拒否することを検証する。 */
public class GrpOptimizerTest {
  @DataProvider
  public Object[][] invalidStates() {
    return new Object[][] {
      {"partial", "Incomplete GRP optimizer counters"},
      {"unknown", "Unknown or ambiguous GRP optimizer parameter ID"},
      {"shape", "Incompatible GRP optimizer moment"}
    };
  }

  @Test(groups = "native", dataProvider = "invalidStates")
  public void ambiguousLegacyMomentsAreRejected(String defect, String message) throws Exception {
    Path state = Files.createTempFile("grp-invalid-", ".state");
    try (Model model = Model.newInstance("grp", Device.cpu(), "PyTorch")) {
      var block = new EpsilonGrpNetwork(16, 1);
      model.setBlock(block);
      block.initialize(
          model.getNDManager(),
          DataType.FLOAT32,
          new Shape(-1, -1, EpsilonGrpFeature.FEATURE_SIZE),
          new Shape(-1));
      Optimizer legacy = Optimizer.adamW().build();
      var parameters = block.getParameters();
      try (var manager = model.getNDManager().newSubManager()) {
        for (int i = 0; i < parameters.size(); i++) {
          if (defect.equals("partial") && i == 0) continue;
          var parameter = parameters.valueAt(i);
          String id = defect.equals("unknown") && i == 0 ? "unknown-parameter" : parameter.getId();
          var weight =
              defect.equals("shape") && i == 0 ? manager.zeros(new Shape(1)) : parameter.getArray();
          legacy.update(id, weight, manager.ones(weight.getShape()));
        }
      }
      legacy.saveState(state);
      try (var trainer = new EpsilonGrpTrainer(model, 1, .001f, .0001f, 1f)) {
        var failure =
            Assert.expectThrows(IOException.class, () -> trainer.loadOptimizerState(state));
        Assert.assertTrue(failure.getMessage().contains(message), failure.getMessage());
      }
    } finally {
      Files.deleteIfExists(state);
    }
  }
}

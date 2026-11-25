package com.epsilon.prism.app;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.util.List;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class SplitDuelGpuTest {
  @Test(groups = "rocm")
  public void hipFrozenNetworksRepeatAcrossWorkerCounts() throws Exception {
    if (Engine.getInstance().getGpuCount() == 0) throw new SkipException("Requires a GPU");
    SplitDuelTest.frozenComparison("gpu0", "BFLOAT16", "HIP", PrismShape.standard(2));
  }

  @Test(groups = "rocm")
  public void mixedShapesHonorBothRequestedPrecisions() throws Exception {
    if (Engine.getInstance().getGpuCount() == 0) throw new SkipException("Requires a GPU");
    var root = Files.createTempDirectory("split-precision");
    try {
      var a = root.resolve("a");
      var b = root.resolve("b");
      try (var manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(a, HipFixtures.network(manager, 2, 3), PolicyParams.of(1));
        Snapshot.write(
            b,
            HipFixtures.network(manager, new PrismShape.V3(32, 128, 64, 128, 128, 128, true), 7),
            PolicyParams.of(2));
      }
      for (String precision : List.of("FLOAT32", "BFLOAT16")) {
        var fields = SplitDuelTest.overall(SplitDuelTest.compare(root, a, b, "gpu0", precision, 2));
        for (String role : List.of("a", "b", "mu")) {
          Assert.assertEquals(fields.get(role + "Precision"), precision);
          Assert.assertEquals(
              fields.get(role + "Backend"),
              precision.equals("BFLOAT16") && !role.equals("b") ? "HIP" : "EAGER");
        }
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}

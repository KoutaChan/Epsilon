package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * TEMPO の錨の HIP の道（rocm の組。GPU 1 枚）。保存した同じ網と同じ塊で、HIP の錨（収集の方策と同じ自前の推論）と eager の錨（GPU の
 * BF16）の s が、行ごとの先頭を通して同じ手に並び、相対 RMS ≤ 2e-2 で合う。
 */
@Test(groups = "rocm")
public class AnchorGpuTest {
  private static final Logger log = LoggerFactory.getLogger(AnchorGpuTest.class);

  private static final double LIMIT = 2e-2;

  public void hipAnchorMatchesEagerAnchor() throws Exception {
    Path directory = Files.createTempDirectory("prism-anchor");
    try {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        HipFixtures.network(manager, 2, 91).save(directory);
      }
      HostBatch batch = HipFixtures.encode(HipFixtures.mixedRows(48, 900));
      HostInputs inputs = new HostInputs(batch, null);
      BatchPlan.Capacity capacity = StepFeed.capacity(FitPlan.TOKEN_BUDGET);
      Device gpu = Device.gpu(0);
      PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
      try (PtStream stream = engine.newStream(gpu);
          Anchor hip = Anchor.open(directory, InferencePath.HIP, gpu, stream, null, capacity);
          Anchor eager = Anchor.open(directory, InferencePath.EAGER, gpu, stream, null, capacity)) {
        Assert.assertTrue(hip instanceof Anchor.Hip, "HIP path on the GPU");
        Assert.assertTrue(eager instanceof Anchor.Eager, "eager path on request");
        Anchor.Scores fast = hip.scores(inputs);
        Anchor.Scores slow = eager.scores(inputs);
        BatchPlan plan = batch.plan();
        Assert.assertEquals(fast.starts().length, plan.rows());
        Assert.assertEquals(slow.starts().length, plan.rows());
        double squared = 0;
        double reference = 0;
        int compared = 0;
        for (int row = 0; row < plan.rows(); row++) {
          for (int k = 0; k < plan.actions(row); k++) {
            double difference = fast.values()[fast.starts()[row] + k] - slow.values()[slow.starts()[row] + k];
            double value = slow.values()[slow.starts()[row] + k];
            squared += difference * difference;
            reference += value * value;
            compared++;
          }
        }
        double relative = Math.sqrt(squared / reference);
        log.info(
            "PRISM-ANCHOR HIP vs eager BF16: {} scores in {} rows, relative RMS {}",
            compared,
            plan.rows(),
            String.format(Locale.ROOT, "%.3e", relative));
        Assert.assertTrue(relative <= LIMIT, "anchor scores differ: relative RMS " + relative);
      }
    } finally {
      Snapshot.deleteTree(directory);
    }
  }
}

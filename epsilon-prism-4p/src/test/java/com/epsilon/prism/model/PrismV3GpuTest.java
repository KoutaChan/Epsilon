package com.epsilon.prism.model;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.backend.HipBackend;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.backend.WeightLayout;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.input.HostBatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 型別 FFN と再読を持つ HIP 計画の順伝播。 */
public class PrismV3GpuTest {
  private static final Logger log = LoggerFactory.getLogger(PrismV3GpuTest.class);

  /**
   * 型ごとの全てのアームで、HIP の計画を作れ、重みを写せ、順伝播が最後まで回り（s が有限）、カーネルを 1 本ずつでも回せる（GPU 1 枚）。
   */
  @Test(groups = "rocm", dataProvider = "typedArms", dataProviderClass = PrismV3Test.class)
  public void typedArmsRunEveryKernel(ShapeArm arm) {
    PrismShape.TokenTrunk shape = arm.shape(ShapeArm.V3_DEPTH);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream =
            ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0))) {
      manager.getEngine().setRandomSeed(17);
      PrismNetwork typed = new PrismNetwork(FOUR_PLAYER, shape);
      typed.initialize(manager);
      Batches.perturb(typed, 0.05f, 19);
      WeightLayout layout = WeightLayout.of(typed);
      HostBatch batch = HipFixtures.encode(HipFixtures.mixedRows(6, 300));
      HipBatch hipBatch = HipBackend.batchOf(batch);
      try (HipBackend backend =
          HipBackend.open(HipLibrary.load(), stream, layout, PrismV3Test.capacity(6))) {
        backend.publish(WeightPack.build(layout, typed, 2));
        for (float score : backend.forward(batch, 2)) {
          Assert.assertTrue(Float.isFinite(score), "score " + score);
        }
        log.info("PRISM-V3-HIP {} runs the whole forward", arm);
        backend
            .plan()
            .copyIn(HipRegion.INPUT, batch.blob().duplicate().position(0), hipBatch.bytes());
        backend.plan().run(HipKernel.RELATION, 0, hipBatch);
        backend.plan().run(HipKernel.MLP_ALL, 0, hipBatch);
        backend.plan().run(HipKernel.ATTENTION_FIXED, 1, hipBatch);
        backend.plan().run(HipKernel.MLP_FIXED, 1, hipBatch);
        backend.plan().run(HipKernel.ATTENTION_FIXED, 2, hipBatch);
        backend.plan().run(HipKernel.MLP_LAST, 2, hipBatch);
        backend.plan().synchronize();
        if (shape.reread()) {
          backend.plan().run(HipKernel.REREAD_KEY_VALUE, 0, hipBatch);
          backend.plan().run(HipKernel.CANDIDATE, 0, hipBatch);
          backend.plan().run(HipKernel.REREAD, 0, hipBatch);
          backend.plan().synchronize();
        }
      }
    }
  }
}

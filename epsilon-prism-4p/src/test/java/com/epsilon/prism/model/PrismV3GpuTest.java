package com.epsilon.prism.model;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.backend.NativeLane;
import com.epsilon.prism.input.HostBatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 型ごとの FFN と候補層の 2 段目を持つ形の、製品の入口での順伝播。 */
public class PrismV3GpuTest {
  private static final Logger log = LoggerFactory.getLogger(PrismV3GpuTest.class);

  /** 型ごとの全てのアームで、製品の入口で計画を開いて重みを公開でき、順伝播が最後まで回る（s が有限。GPU 1 枚）。 */
  @Test(groups = "rocm", dataProvider = "typedArms", dataProviderClass = PrismV3Test.class)
  public void typedArmsRunTheWholeForward(ShapeArm arm) {
    PrismShape.TokenTrunk shape = arm.shape(ShapeArm.V3_DEPTH);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      manager.getEngine().setRandomSeed(17);
      PrismNetwork typed = new PrismNetwork(FOUR_PLAYER, shape);
      typed.initialize(manager);
      Batches.perturb(typed, 0.05f, 19);
      HostBatch batch = HipFixtures.encode(HipFixtures.mixedRows(6, 300));
      try (NativeLane lane = NativeLane.open(typed, Device.gpu(0), PrismV3Test.capacity(6), 2)) {
        for (float score : lane.forward(batch)) {
          Assert.assertTrue(Float.isFinite(score), "score " + score);
        }
        log.info("PRISM-V3-NATIVE {} runs the whole forward on {}", arm, lane.backend());
      }
    }
  }
}

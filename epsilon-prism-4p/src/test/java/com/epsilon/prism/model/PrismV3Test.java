package com.epsilon.prism.model;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.backend.WeightLayout;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.kernel.CollectionArchitecture;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 型別 FFN の記述と遷移の再読を確かめる。 */
public class PrismV3Test {
  /** 型ごとの全てのアームで、HIP の形の記述・重みの束・計画の形を作れる。 */
  @Test(dataProvider = "typedArms")
  public void typedArmsBuildTheHipDescriptionPackAndPlanShape(ShapeArm arm) {
    PrismShape.TokenTrunk shape = arm.shape(ShapeArm.V3_DEPTH);
    Assert.assertTrue(WeightLayout.supports(shape));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(FOUR_PLAYER, shape);
      network.initialize(manager);
      WeightLayout layout = WeightLayout.of(network);
      CollectionArchitecture architecture = layout.architecture();
      Assert.assertEquals(architecture.layers(), shape.layers());
      Assert.assertEquals(architecture.reread(), shape.reread());
      Assert.assertEquals(architecture.candidateFfn(), shape.candidateFfn());
      WeightPack pack = WeightPack.build(layout, network, 1);
      Assert.assertEquals(pack.buffer().remaining(), layout.bytes());
    }
  }

  @DataProvider
  public static Object[][] typedArms() {
    return new Object[][] {{ShapeArm.V3}, {ShapeArm.V3_ONE_STAGE}, {ShapeArm.V3_CONTEXT_512}};
  }

  /** {@code rows} 行のバッチを入れる容量。 */
  static BatchPlan.Capacity capacity(int rows) {
    return new BatchPlan.Capacity(rows, rows * 110, rows * 192, rows * 32);
  }
}

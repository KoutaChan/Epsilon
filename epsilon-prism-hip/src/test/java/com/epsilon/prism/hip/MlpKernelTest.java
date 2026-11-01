package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;

import java.nio.ByteBuffer;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/** K3 を GPU 1 枚で CPU の参照と比べる。 */
@Test(groups = "rocm")
public class MlpKernelTest {

  private static final Logger log = LoggerFactory.getLogger(MlpKernelTest.class);

  private static final int FFN = 512;
  private static final int CAPACITY = 1024;

  private PtStream stream;
  private HipPlan plan;
  private long version;

  @BeforeClass(groups = "rocm")
  public void openPlan() {
    HipLibrary library = HipLibrary.load();
    stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
    plan = HipPlan.create(library, stream, MlpCase.shape(FFN, CAPACITY));
  }

  @AfterClass(groups = "rocm", alwaysRun = true)
  public void closePlan() {
    if (plan != null) {
      plan.close();
    }
    if (stream != null) {
      stream.close();
    }
  }

  /** 本番の形（H=256・F=512）で、64 の倍数でないトークンの数（最後の塊が半端）でも参照と合う。 */
  public void matchesReference() {
    MlpCase c = MlpCase.random(1, FFN, 1000, CAPACITY);
    MlpReference.Comparison comparison = MlpReference.Comparison.of(MlpReference.run(c), run(c));
    log.info(
        "K3 vs reference: relative RMS {}, within 1 ulp {}, max ulp {}",
        String.format("%.3e", comparison.relativeRms()),
        String.format("%.5f", comparison.withinOneUlp()),
        comparison.maxUlp());
    assertTrue(comparison.relativeRms() <= 1e-2, "relative RMS " + comparison.relativeRms());
    assertTrue(comparison.withinOneUlp() >= 0.99, "within 1 ulp " + comparison.withinOneUlp());
  }

  /** 残差の枝が 0 なら、x はビットで変わらない（x1 = ⌊x + 0⌉ = x、x = ⌊x1 + 0⌉）。 */
  public void zeroBranchKeepsResidual() {
    MlpCase c = MlpCase.random(2, FFN, 333, CAPACITY).withoutResidualBranch();
    assertEquals(run(c), c.x);
  }

  /** 同じトークンは、塊の中の位置（WG の中の行）が変わってもビットで同じ。 */
  public void independentOfBatchPosition() {
    MlpCase alone = MlpCase.random(3, FFN, 300, CAPACITY);
    MlpCase shifted = alone.prepend(MlpCase.random(4, FFN, 37, CAPACITY));
    short[] expected = run(alone);
    short[] actual = run(shifted);
    int h = alone.shape.hidden();
    assertEquals(Arrays.copyOfRange(actual, 37 * h, actual.length), expected);
  }

  private short[] run(MlpCase c) {
    plan.uploadWeights(c.pack(), ++version);
    plan.copyIn(HipRegion.RESIDUAL, c.input(HipRegion.RESIDUAL), c.bytes(HipRegion.RESIDUAL));
    plan.copyIn(HipRegion.ATTENTION, c.input(HipRegion.ATTENTION), c.bytes(HipRegion.ATTENTION));
    plan.run(HipKernel.MLP0, 0, c.batch());
    ByteBuffer out = TestPack.direct(c.bytes(HipRegion.RESIDUAL));
    plan.copyOut(HipRegion.RESIDUAL, out, c.bytes(HipRegion.RESIDUAL));
    plan.synchronize();
    return MlpReference.shorts(out, c.tokens * c.shape.hidden());
  }
}

package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.model.Batches.Batch;
import java.util.List;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 順伝播の前に送る教師（{@link LossInputs}）の損失と、s・補助の頭の出力への勾配が、GPU でも損失の中で教師を作る参照の道（{@link ReferenceLoss}）
 * とビットで同じこと（rocm の組。GPU 1 枚）。標準の形・深さ 2 の乱した網の BF16 の順伝播の出力を葉にし、出来事の数（S）の
 * 違う 4 種の行の塊で、補助の頭のある順伝播とない順伝播を比べる。送った区画は型ごとの 1 本の配列の途中から始まるので、GPU の演算が
 * 配列の位置の揃い方で値を変えないことも確かめる。比べ方は {@link PrismLossTest#assertStagedLossMatchesTheReference}。
 */
@Test(groups = "rocm")
public class PrismLossGpuTest {

  private static final int ROWS = 96;

  public void stagedLossIsBitwiseTheInlineLossOnGpu() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0))) {
      manager.getEngine().setRandomSeed(3);
      PrismNetwork network = new PrismNetwork(PrismShape.standard(2));
      network.initialize(manager);
      Batches.perturb(network, 0.05f, 7);
      Batch batch = Batches.build(PrismLossTest.rows(ROWS), 5);
      List<ValueLabel> labels = PrismLossTest.labels(batch, 11, -0.5f);
      PrismLossTest.assertStagedLossMatchesTheReference(
          network, manager, Precision.BFLOAT16, batch, labels, Batches.auxiliary(batch, 13));
      PrismLossTest.assertStagedLossMatchesTheReference(
          network, manager, Precision.BFLOAT16, batch, labels, null);
    }
  }
}

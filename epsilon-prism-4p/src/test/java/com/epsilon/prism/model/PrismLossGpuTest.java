package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism4p.input.InputSchema;
import java.util.List;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 順伝播の前に送る教師（{@link LossInputs}）で求めた損失と、s・補助のヘッドの出力への勾配が、GPU でも、損失の中で教師を作る参照の経路
 * （{@link ReferenceLoss}）とビットで同じことを確かめる（rocm のグループ。GPU 1 枚）。標準の形・深さ 2 の乱したネットワークで BF16 の
 * 順伝播をし、その出力を葉にする。イベントの数（S）の違う 4 種の行のバッチで、補助のヘッドのある順伝播とない順伝播を比べる。送った領域は
 * 型ごとの 1 本の配列の途中から始まるので、配列の上の位置のアラインメントで GPU の演算が値を変えないことも確かめる。比べ方は
 * {@link PrismLossTest#assertStagedLossMatchesTheReference}。
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
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, PrismShape.standard(2));
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

package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.backend.ThreePlayerRows;
import com.epsilon.prism.input.HostBatch;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 三人打ちの {@link AttentionTape} を確かめる（CPU、FP32）。バックボーンの Attention と正規化を、自動微分の外にある参照実装
 * （{@link ReferenceAttention}）に置き換えて、逆伝播を区切る。その損失・s・全てのパラメーターの勾配は、djl の経路（自動微分の中）と、
 * 和の順の違いを除いて同じになる（s ≤ 1e-5、テンソルごとの勾配の相対の差 ≤ 1e-4）。ネットワークは三人打ちの入力形式で乱数で初期化した、
 * v2 の小さな形（深さ 2 と 3）、型ごとの FFN の小さな形（候補層の 2 段目の有無）と V3 の形。行はエンジンで打った三人打ちの判断（全ての
 * 手の種類と、公開のイベントが多い行を含む）。どちらの経路でも、全てのパラメーターに勾配が届く。順伝播の区切りで途中の配列を閉じても、
 * 閉じないときと損失・s・勾配がビット単位で同じ。記録しない形（{@link AttentionTape#forwardOnly}）では、s が推論の経路と同じ。
 */
public class ThreePlayerAttentionTapeTest {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerAttentionTapeTest.class);

  /** ふつうの判断の行の数。 */
  private static final int ROWS = 22;

  /** 公開のイベントが多い行の数。 */
  private static final int LONG_ROWS = 2;

  /** 型ごとの FFN の小さな形（CPU で速く実行できる）。 */
  private static final PrismShape TYPED =
      new PrismShape(64, 128, 64, 3, new PrismShape.Typed(128, 256, 128, true));

  /** 型ごとの FFN の小さな形のうち、候補層の 2 段目がない形。 */
  private static final PrismShape TYPED_ONE_STAGE =
      new PrismShape(64, 128, 64, 3, new PrismShape.Typed(128, 256, 128, false));

  private HostBatch batch;

  @BeforeClass
  public void rows() {
    List<long[]> rows = new ArrayList<>(ThreePlayerRows.mixedRows(ROWS, 900));
    rows.addAll(ThreePlayerRows.longRows(LONG_ROWS));
    batch = ThreePlayerRows.encode(rows);
  }

  /** v2 の小さな形。 */
  private static PrismShape small(int depth) {
    return new PrismShape(64, 128, 64, depth, null);
  }

  @DataProvider
  public Object[][] shapes() {
    return new Object[][] {
      {small(PrismShape.MIN_DEPTH)},
      {small(PrismShape.MAX_DEPTH)},
      {TYPED},
      {TYPED_ONE_STAGE},
      {ShapeArm.V3.shape(ShapeArm.V3_DEPTH)}
    };
  }

  @Test(timeOut = 600_000, dataProvider = "shapes")
  public void tapeKeepsTheGradients(PrismShape shape) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = ThreePlayerRows.network(manager, shape, 3);
      ThreePlayerSteps.Outcome reference =
          ThreePlayerSteps.run(network, manager, batch, null, Precision.FLOAT32, Context.Intermediates.CLOSE);
      ThreePlayerSteps.Outcome tape =
          ThreePlayerSteps.run(
              network, manager, batch, new ReferenceAttention(), Precision.FLOAT32, Context.Intermediates.CLOSE);
      double scoreError = ThreePlayerSteps.relative(reference.scores(), tape.scores());
      List<Map.Entry<String, Double>> ratios = new ArrayList<>();
      for (Map.Entry<String, float[]> entry : reference.gradients().entrySet()) {
        float[] expected = entry.getValue();
        double norm = Math.sqrt(ThreePlayerSteps.squared(expected));
        double error = Math.sqrt(ThreePlayerSteps.difference(expected, tape.gradients().get(entry.getKey())));
        ratios.add(Map.entry(entry.getKey(), norm > 1e-12 ? error / norm : error));
      }
      ratios.sort(Map.Entry.<String, Double>comparingByValue().reversed());
      String worstName = ratios.getFirst().getKey();
      double worst = ratios.getFirst().getValue();
      log.info(
          "PRISM-3P-TAPE-CPU shape={} rows={} loss {} vs {} scores rel={} gradients rel={} worst tensors {}",
          shape,
          batch.plan().rows(),
          String.format("%.7f", reference.loss()),
          String.format("%.7f", tape.loss()),
          String.format("%.3g", scoreError),
          String.format("%.3g", ThreePlayerSteps.gradients(reference, tape)),
          ratios.subList(0, Math.min(8, ratios.size())).stream()
              .map(entry -> entry.getKey() + "=" + String.format("%.3g", entry.getValue()))
              .toList());
      Assert.assertTrue(ThreePlayerSteps.maxAbs(reference.scores()) > 1e-3f, "scores must be non-trivial");
      Assert.assertEquals(tape.loss(), reference.loss(), 1e-5f * Math.abs(reference.loss()), shape + " loss");
      Assert.assertTrue(scoreError <= 1e-5, shape + " scores rel " + scoreError);
      Assert.assertTrue(worst <= 1e-4, shape + " gradient " + worstName + " rel " + worst);
      Assert.assertEquals(ThreePlayerSteps.silent(reference), List.of(), shape + " djl gradients that are all zero");
      Assert.assertEquals(ThreePlayerSteps.silent(tape), List.of(), shape + " tape gradients that are all zero");
    }
  }

  /** 記録しない形: 推論の s が、djl の演算での推論と和の順の違いを除いて同じになる。 */
  @Test(timeOut = 600_000, dataProvider = "shapes")
  public void forwardOnlyKeepsTheInferenceScores(PrismShape shape) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = ThreePlayerRows.network(manager, shape, 5);
      float[] reference = infer(network, manager, null);
      float[] kernels = infer(network, manager, AttentionTape.forwardOnly(new ReferenceAttention()));
      double error = ThreePlayerSteps.relative(reference, kernels);
      log.info("PRISM-3P-TAPE-INFERENCE shape={} scores rel={}", shape, String.format("%.3g", error));
      Assert.assertTrue(ThreePlayerSteps.maxAbs(reference) > 1e-3f, "scores must be non-trivial");
      Assert.assertTrue(error <= 1e-5, shape + " scores rel " + error);
    }
  }

  /**
   * 途中の配列を区切りで閉じる順伝播と、区切りを作らない順伝播（{@link Context.Intermediates#KEEP}）で、損失・s・全ての勾配がビット単位で
   * 同じになる。記録の有無の両方で、v2 の小さな形と型ごとの FFN の小さな形は FP32 と BF16、V3 の形は FP32 で確かめる。どちらも作った
   * ばかりの同じネットワークで 1 回だけ実行する（勾配のバッファの有無で、0 の符号が変わらないように）。
   */
  @Test(timeOut = 900_000)
  public void closingIntermediatesKeepsEveryBit() {
    List<Object[]> cases = new ArrayList<>();
    for (Precision precision : Precision.values()) {
      cases.add(new Object[] {small(PrismShape.MIN_DEPTH), precision});
      cases.add(new Object[] {TYPED, precision});
    }
    cases.add(new Object[] {ShapeArm.V3.shape(ShapeArm.V3_DEPTH), Precision.FLOAT32});
    for (Object[] c : cases) {
      PrismShape shape = (PrismShape) c[0];
      Precision precision = (Precision) c[1];
      for (boolean recorded : new boolean[] {false, true}) {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
          ReferenceAttention kernels = recorded ? new ReferenceAttention() : null;
          ThreePlayerSteps.Outcome closed =
              ThreePlayerSteps.run(
                  ThreePlayerRows.network(manager, shape, 7),
                  manager,
                  batch,
                  kernels,
                  precision,
                  Context.Intermediates.CLOSE);
          ThreePlayerSteps.Outcome kept =
              ThreePlayerSteps.run(
                  ThreePlayerRows.network(manager, shape, 7),
                  manager,
                  batch,
                  kernels,
                  precision,
                  Context.Intermediates.KEEP);
          String label = shape + " " + precision + (recorded ? " tape" : " djl");
          log.info(
              "PRISM-3P-TAPE-SCOPES {} loss {} vs {}",
              label,
              String.format("%.7f", closed.loss()),
              String.format("%.7f", kept.loss()));
          Assert.assertEquals(
              Float.floatToRawIntBits(closed.loss()), Float.floatToRawIntBits(kept.loss()), label + " loss");
          Assert.assertTrue(Arrays.equals(closed.scores(), kept.scores()), label + " scores");
          for (Map.Entry<String, float[]> entry : kept.gradients().entrySet()) {
            Assert.assertTrue(
                Arrays.equals(entry.getValue(), closed.gradients().get(entry.getKey())),
                label + " gradient " + entry.getKey());
          }
        }
      }
    }
  }

  private float[] infer(PrismNetwork network, NDManager manager, AttentionTape attention) {
    try (NDManager step = manager.newSubManager()) {
      return network
          .forward(
              new ParameterStore(manager, false),
              NetworkInputs.of(step, PinnedUploads.on(step), batch),
              Pass.inference(Precision.FLOAT32, attention))
          .scores()
          .toFloatArray();
    }
  }
}

package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism4p.input.InputSchema;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * {@link AttentionTape} のつなぎ込みを確かめる（CPU）。バックボーンの Attention と正規化を自動微分の外の参照の実装（{@link ReferenceAttention}）
 * に替え、そこで区切って逆伝播したときの損失・s・全てのパラメーターの勾配が、eager の経路（全て自動微分の中）と、和の順の違いを除いて同じ。
 * 形は v2 の深さ 2 と 3、型ごとの FFN の小さな形（候補層の 2 段目の有無）、V3 で、イベントの数の違う行を混ぜる。どちらの経路でも全ての
 * パラメーターに勾配が届く。順伝播の区切り（{@link Context#scoped}）で途中の配列を閉じても、閉じないときと損失・s・勾配がビットで同じ。
 * 記録しない使い方（{@link AttentionTape#forwardOnly}、アンカーの推論）では、s が推論の経路と同じ。
 */
public class AttentionTapeTest {
  private static final Logger log = LoggerFactory.getLogger(AttentionTapeTest.class);

  private static final int[] EVENTS = {0, 3, 9, 17};

  /** 型ごとの FFN の小さな形（CPU で速く回す）。 */
  private static final PrismShape TYPED =
      new PrismShape(64, 128, 64, 3, new PrismShape.Typed(128, 256, 128, true));

  /** 型ごとの FFN の小さな形で、候補層の 2 段目のない形。 */
  private static final PrismShape TYPED_ONE_STAGE =
      new PrismShape(64, 128, 64, 3, new PrismShape.Typed(128, 256, 128, false));

  @DataProvider
  public Object[][] shapes() {
    return new Object[][] {
      {Batches.small(PrismShape.MIN_DEPTH)},
      {Batches.small(PrismShape.MAX_DEPTH)},
      {TYPED},
      {TYPED_ONE_STAGE},
      {ShapeArm.V3.shape(ShapeArm.V3_DEPTH)}
    };
  }

  @Test(timeOut = 600_000, dataProvider = "shapes")
  public void tapeKeepsTheGradients(PrismShape shape) {
    compare(shape, Precision.FLOAT32);
  }

  /**
   * BF16（GPU の学習と同じ丸め）でも、つなぎ込みが勾配を落とさない。2 つの経路の差は和の順と、そこから来る BF16 の丸めの違い（正規化の
   * 出力にまれに出る 1 ulp がネットワークで広がる）だけで、どちらの経路も FP32 から丸めの分だけずれる（型ごとの小さな形で s が約 2.5%）。
   * そこで、記録する経路の s の FP32 からのずれが djl の経路の 1.25 倍以内で、テンソルごとの勾配の差が 5e-2 未満であることを見る
   * （つなぎ込みで勾配の経路が抜ければ 1 の桁になる）。
   */
  @Test(timeOut = 600_000)
  public void tapeKeepsTheGradientsInBfloat16() {
    compare(Batches.small(PrismShape.MIN_DEPTH), Precision.BFLOAT16);
    compare(Batches.small(PrismShape.MAX_DEPTH), Precision.BFLOAT16);
    compare(TYPED, Precision.BFLOAT16);
  }

  /** 記録しない使い方: 推論の s が、djl-rocm の演算での推論と、和の順の違いを除いて同じ。 */
  @Test(timeOut = 600_000, dataProvider = "shapes")
  public void forwardOnlyKeepsTheInferenceScores(PrismShape shape) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = network(manager, shape);
      Batch batch = Batches.build(rows(), 5);
      float[] reference = infer(network, manager, batch, null);
      float[] kernels =
          infer(network, manager, batch, AttentionTape.forwardOnly(new ReferenceAttention()));
      double error = relative(reference, kernels);
      log.info("PRISM-TAPE-INFERENCE shape={} scores rel={}", shape, String.format("%.3g", error));
      Assert.assertTrue(Batches.maxAbs(reference) > 1e-3f, "scores must be non-trivial");
      Assert.assertTrue(error < 1e-5, shape + " scores rel " + error);
    }
  }

  /** 記録する使い方は推論では拒み、1 つの記録は 1 回の順伝播だけを記録する。 */
  @Test
  public void aRecordingTapeTrainsOneForwardPass() {
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Pass.inference(Precision.FLOAT32, new AttentionTape(new ReferenceAttention())));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      AttentionTape tape = new AttentionTape(new ReferenceAttention());
      tape.open(manager);
      Assert.assertThrows(IllegalStateException.class, () -> tape.open(manager));
    }
  }

  /**
   * 区切りで途中の配列を閉じる順伝播と、区切りを作らない順伝播（{@link Context.Intermediates#KEEP}）で、損失・s・全ての勾配がビットで
   * 同じ（記録の有無、v2 の深さ 2 と型ごとの FFN の小さな形、FP32 と BF16）。どちらも作ったばかりの同じネットワークで 1 回だけ回す
   * （勾配のバッファの有無で 0 の符号が変わらないように）。
   */
  @Test(timeOut = 600_000)
  public void closingIntermediatesKeepsEveryBit() {
    for (Precision precision : Precision.values()) {
      for (PrismShape shape : List.of(Batches.small(PrismShape.MIN_DEPTH), TYPED)) {
        for (boolean recorded : new boolean[] {false, true}) {
          try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            Batch batch = Batches.build(rows(), 5);
            Outcome closed =
                run(network(manager, shape), manager, batch, recorded, precision, Context.Intermediates.CLOSE);
            Outcome kept =
                run(network(manager, shape), manager, batch, recorded, precision, Context.Intermediates.KEEP);
            String label = shape + " " + precision + (recorded ? " tape" : " djl");
            log.info(
                "PRISM-TAPE-SCOPES {} loss {} vs {}",
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
  }

  private record Outcome(float loss, float[] scores, Map<String, float[]> gradients) {}

  /** 乱したネットワーク（種は形に依らず同じ）。 */
  private static PrismNetwork network(NDManager manager, PrismShape shape) {
    manager.getEngine().setRandomSeed(3);
    PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
    network.initialize(manager);
    Batches.perturb(network, 0.05f, 7);
    return network;
  }

  /** 手番・応答・鳴いた後の行を、イベントの数を変えて 12 行作る。 */
  private static List<Batches.Row> rows() {
    List<Batches.Row> rows = new ArrayList<>();
    for (int r = 0; r < 12; r++) {
      int events = EVENTS[r % EVENTS.length];
      rows.add(
          switch (r % 4) {
            case 0, 1 -> Batches.turn(events);
            case 2 -> Batches.response(events);
            default -> Batches.postCall(events);
          });
    }
    return rows;
  }

  private static float[] infer(
      PrismNetwork network, NDManager manager, Batch batch, AttentionTape attention) {
    try (NDManager step = manager.newSubManager()) {
      return network
          .forward(
              new ParameterStore(manager, false),
              batch.inputs(step),
              Pass.inference(Precision.FLOAT32, attention))
          .scores()
          .toFloatArray();
    }
  }

  private static void compare(PrismShape shape, Precision precision) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = network(manager, shape);
      Batch batch = Batches.build(rows(), 5);
      Outcome reference = run(network, manager, batch, false, precision, Context.Intermediates.CLOSE);
      Outcome tape = run(network, manager, batch, true, precision, Context.Intermediates.CLOSE);
      boolean exact = precision == Precision.FLOAT32;
      Assert.assertEquals(
          tape.loss(), reference.loss(), (exact ? 1e-5f : 1e-2f) * Math.abs(reference.loss()));
      double scoreError = relative(reference.scores(), tape.scores());
      List<Map.Entry<String, Double>> ratios = new ArrayList<>();
      for (Map.Entry<String, float[]> entry : reference.gradients().entrySet()) {
        float[] expected = entry.getValue();
        double norm = Math.sqrt(squared(expected));
        double error = Math.sqrt(difference(expected, tape.gradients().get(entry.getKey())));
        ratios.add(Map.entry(entry.getKey(), norm > 1e-12 ? error / norm : error));
      }
      ratios.sort(Map.Entry.<String, Double>comparingByValue().reversed());
      String worstName = ratios.getFirst().getKey();
      double worst = ratios.getFirst().getValue();
      // BF16 では 2 つの経路がそれぞれ FP32 から丸めの分だけずれるので、FP32 の s からのずれも比べる
      double djlRounding = 0;
      double tapeRounding = 0;
      if (!exact) {
        Outcome fp32 = run(network, manager, batch, false, Precision.FLOAT32, Context.Intermediates.CLOSE);
        djlRounding = relative(fp32.scores(), reference.scores());
        tapeRounding = relative(fp32.scores(), tape.scores());
      }
      log.info(
          "PRISM-TAPE-CPU shape={} {} loss {} vs {} scores rel={} from FP32 djl={} tape={} worst gradients {}",
          shape,
          precision,
          String.format("%.7f", reference.loss()),
          String.format("%.7f", tape.loss()),
          String.format("%.3g", scoreError),
          String.format("%.3g", djlRounding),
          String.format("%.3g", tapeRounding),
          ratios.subList(0, Math.min(8, ratios.size())).stream()
              .map(entry -> entry.getKey() + "=" + String.format("%.3g", entry.getValue()))
              .toList());
      Assert.assertTrue(Batches.maxAbs(reference.scores()) > 1e-3f, "scores must be non-trivial");
      Assert.assertTrue(scoreError < (exact ? 1e-5 : 1e-2), "scores rel " + scoreError);
      Assert.assertTrue(worst < (exact ? 1e-4 : 5e-2), "gradient " + worstName + " rel " + worst);
      Assert.assertTrue(
          tapeRounding <= 1.25 * djlRounding,
          "the tape's scores are further from FP32 (" + tapeRounding + ") than djl's (" + djlRounding + ")");
      Assert.assertEquals(silent(reference.gradients()), List.of(), shape + " djl gradients that are all zero");
      Assert.assertEquals(silent(tape.gradients()), List.of(), shape + " tape gradients that are all zero");
    }
  }

  /** 勾配が全て 0 のパラメーター。 */
  private static List<String> silent(Map<String, float[]> gradients) {
    List<String> names = new ArrayList<>();
    for (Map.Entry<String, float[]> entry : gradients.entrySet()) {
      if (Batches.maxAbs(entry.getValue()) == 0) {
        names.add(entry.getKey());
      }
    }
    return names;
  }

  private static Outcome run(
      PrismNetwork network,
      NDManager manager,
      Batch batch,
      boolean recorded,
      Precision precision,
      Context.Intermediates intermediates) {
    List<ValueLabel> labels = Batches.values(batch, 11);
    AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 13);
    boolean[] measured = ValueLabel.measured(labels);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    float value;
    float[] scores;
    try (NDManager step = manager.newSubManager();
        GradientCollector collector = manager.getEngine().newGradientCollector()) {
      AttentionTape tape = recorded ? new AttentionTape(new ReferenceAttention()) : null;
      Outputs outputs =
          network.forward(
              new ParameterStore(manager, false),
              batch.inputs(step),
              Pass.training(precision, measured, auxiliary, tape),
              intermediates);
      NDArray total = loss.total(step, outputs, labels, auxiliary, 1f, labels.size(), 1f);
      if (tape == null) {
        collector.backward(total);
      } else {
        tape.backward(collector, total);
      }
      value = total.getFloat();
      scores = outputs.scores().toFloatArray();
    }
    return new Outcome(value, scores, Batches.takeGradients(network));
  }

  private static double relative(float[] expected, float[] actual) {
    return Math.sqrt(difference(expected, actual) / Math.max(squared(expected), 1e-30));
  }

  private static double squared(float[] values) {
    double sum = 0;
    for (float value : values) {
      sum += (double) value * value;
    }
    return sum;
  }

  private static double difference(float[] expected, float[] actual) {
    double sum = 0;
    for (int i = 0; i < expected.length; i++) {
      double d = (double) expected[i] - actual[i];
      sum += d * d;
    }
    return sum;
  }
}

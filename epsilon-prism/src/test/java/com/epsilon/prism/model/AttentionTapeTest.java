package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.model.Batches.Batch;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * {@link AttentionTape} のつなぎ（CPU の FP32）: 背骨の注意を自動微分の外の参照の実装（{@link ReferenceAttention}）にして区切った逆伝播の
 * 損失・s・全てのパラメーターの勾配が、djl-rocm の演算の道（自動微分の中）と和の順の違いを除いて同じこと。深さ 2 と 3、出来事の数の違う行と、
 * 最後の層の問い合わせを絞る形を含む。記録しない形（{@link AttentionTape#forwardOnly}、TEMPO の錨の推論）は s が推論の道と同じこと。
 */
public class AttentionTapeTest {
  private static final Logger log = LoggerFactory.getLogger(AttentionTapeTest.class);

  private static final int[] EVENTS = {0, 3, 9, 17};

  @Test(timeOut = 600_000)
  public void tapeKeepsTheGradients() {
    for (int depth = PrismShape.MIN_DEPTH; depth <= PrismShape.MAX_DEPTH; depth++) {
      compare(depth, Precision.FLOAT32);
    }
  }

  /** BF16（GPU の学習と同じ丸め）でも、つなぎが勾配を落とさないこと。差は丸めの順だけなので緩く見る。 */
  @Test(timeOut = 600_000)
  public void tapeKeepsTheGradientsInBfloat16() {
    compare(PrismShape.MIN_DEPTH, Precision.BFLOAT16);
  }

  /** 記録しない形: 推論の s が djl-rocm の演算の推論と和の順の違いを除いて同じこと。記録する形は推論で拒む。 */
  @Test(timeOut = 600_000)
  public void forwardOnlyKeepsTheInferenceScores() {
    for (int depth = PrismShape.MIN_DEPTH; depth <= PrismShape.MAX_DEPTH; depth++) {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        manager.getEngine().setRandomSeed(3);
        PrismNetwork network = new PrismNetwork(Batches.small(depth));
        network.initialize(manager);
        Batches.perturb(network, 0.05f, 7);
        Batch batch = Batches.build(rows(), 5);
        float[] reference = infer(network, manager, batch, null);
        float[] kernels =
            infer(network, manager, batch, AttentionTape.forwardOnly(new ReferenceAttention()));
        double error = relative(reference, kernels);
        log.info("PRISM-TAPE-INFERENCE depth={} scores rel={}", depth, String.format("%.3g", error));
        Assert.assertTrue(Batches.maxAbs(reference) > 1e-3f, "scores must be non-trivial");
        Assert.assertTrue(error < 1e-5, "depth " + depth + " scores rel " + error);
      }
    }
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Pass.inference(Precision.FLOAT32, new AttentionTape(new ReferenceAttention())));
  }

  private record Outcome(float loss, float[] scores, Map<String, float[]> gradients) {}

  /** 手番・応答・鳴いた後の行を、出来事の数を変えて 12 行。 */
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

  private static void compare(int depth, Precision precision) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      manager.getEngine().setRandomSeed(3);
      PrismNetwork network = new PrismNetwork(Batches.small(depth));
      network.initialize(manager);
      Batches.perturb(network, 0.05f, 7);
      Batch batch = Batches.build(rows(), 5);
      Outcome reference = run(network, manager, batch, false, precision);
      Outcome tape = run(network, manager, batch, true, precision);
      boolean exact = precision == Precision.FLOAT32;
      Assert.assertEquals(
          tape.loss(), reference.loss(), (exact ? 1e-5f : 1e-2f) * Math.abs(reference.loss()));
      double scoreError = relative(reference.scores(), tape.scores());
      double worst = 0;
      String worstName = "";
      for (Map.Entry<String, float[]> entry : reference.gradients().entrySet()) {
        float[] expected = entry.getValue();
        double norm = Math.sqrt(squared(expected));
        double error = Math.sqrt(difference(expected, tape.gradients().get(entry.getKey())));
        double ratio = norm > 1e-12 ? error / norm : error;
        if (ratio > worst) {
          worst = ratio;
          worstName = entry.getKey();
        }
      }
      log.info(
          "PRISM-TAPE-CPU depth={} {} loss {} vs {} scores rel={} worst gradient {} rel={}",
          depth,
          precision,
          String.format("%.7f", reference.loss()),
          String.format("%.7f", tape.loss()),
          String.format("%.3g", scoreError),
          worstName,
          String.format("%.3g", worst));
      Assert.assertTrue(Batches.maxAbs(reference.scores()) > 1e-3f, "scores must be non-trivial");
      Assert.assertTrue(scoreError < (exact ? 1e-5 : 1e-3), "scores rel " + scoreError);
      Assert.assertTrue(worst < (exact ? 1e-4 : 1e-2), "gradient " + worstName + " rel " + worst);
    }
  }

  private static Outcome run(
      PrismNetwork network, NDManager manager, Batch batch, boolean tape, Precision precision) {
    List<ValueLabel> labels = Batches.values(batch, 11);
    AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 13);
    boolean[] measured = ValueLabel.measured(labels);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    float value;
    float[] scores;
    try (NDManager step = manager.newSubManager();
        GradientCollector collector = manager.getEngine().newGradientCollector()) {
      AttentionTape recorded = tape ? new AttentionTape(new ReferenceAttention()) : null;
      Outputs outputs =
          network.forward(
              new ParameterStore(manager, false),
              batch.inputs(step),
              Pass.training(precision, measured, auxiliary, recorded));
      NDArray total = loss.total(step, outputs, labels, auxiliary, 1f, labels.size(), 1f);
      if (recorded == null) {
        collector.backward(total);
      } else {
        recorded.backward(collector, total);
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

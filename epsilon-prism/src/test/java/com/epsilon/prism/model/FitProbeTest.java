package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism.model.FitProbe.Stage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 段の時計の印（{@link FitProbe}）: 計測のときに計測の葉（頭の中・埋め込みの欄の並べ）を作って逆伝播を段に分けても、損失は印のない学習と
 * ビットで同じで、全てのパラメーターの勾配は和の順の差の内で同じこと（CPU、自前の注意と正規化の記録に参照の実装
 * {@link ReferenceAttention}、FP32 と BF16、深さ 2 と 3）。印は 1 微小バッチの順伝播と逆伝播の区切りを決まった順で通る。
 *
 * <p>勾配が式で同じでもビットで同じとは限らないのは、背骨の出力の葉のように寄与が 3 つ以上ある葉では、葉ごとの注入で寄与を足す順が 1 回の
 * 逆伝播の順と変わるため。差はパラメーターごとの最大の |勾配| に対する比で測る（{@link #relativeDifference}）。
 */
public class FitProbeTest {
  private static final Logger log = LoggerFactory.getLogger(FitProbeTest.class);

  private static final int[] EVENTS = {0, 3, 9, 17};

  /** 印の段と層を順に覚え、計測の葉も作る印。 */
  private static final class Recorder implements FitProbe {

    private final List<String> marks = new ArrayList<>();

    @Override
    public void mark(Stage stage, int layer) {
      marks.add(name(stage, layer));
    }

    @Override
    public boolean cuts() {
      return true;
    }
  }

  private record Outcome(float loss, Map<String, float[]> gradients) {}

  /** 勾配の差の上限（パラメーターごとの最大の |勾配| に対する比。FP32 の和の順の丸めの差だけを許す）。 */
  private static final double TOLERANCE = 1e-5;

  @Test(timeOut = 600_000)
  public void splitBackwardKeepsTheLossAndTheGradients() {
    for (Precision precision : Precision.values()) {
      for (int depth = PrismShape.MIN_DEPTH; depth <= PrismShape.MAX_DEPTH; depth++) {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
          manager.getEngine().setRandomSeed(3);
          PrismNetwork network = new PrismNetwork(Batches.small(depth));
          network.initialize(manager);
          Batches.perturb(network, 0.05f, 7);
          Batch batch = Batches.build(rows(), 5);
          Outcome plain = run(network, manager, batch, FitProbe.NONE, precision);
          Recorder recorder = new Recorder();
          Outcome split = run(network, manager, batch, recorder, precision);
          int differing = 0;
          double worst = 0;
          String worstName = "";
          for (Map.Entry<String, float[]> entry : plain.gradients().entrySet()) {
            float[] other = split.gradients().get(entry.getKey());
            Assert.assertNotNull(other, "no gradient for " + entry.getKey());
            double difference = relativeDifference(entry.getValue(), other);
            if (difference > 0) {
              differing++;
            }
            if (difference > worst) {
              worst = difference;
              worstName = entry.getKey();
            }
          }
          log.info(
              "PRISM-PROBE depth={} {} loss {} vs {} gradients differing={} of {} worst={} ({}) marks={}",
              depth,
              precision,
              String.format("%.7f", plain.loss()),
              String.format("%.7f", split.loss()),
              differing,
              plain.gradients().size(),
              String.format("%.3g", worst),
              worstName,
              recorder.marks);
          Assert.assertEquals(
              Float.floatToRawIntBits(split.loss()), Float.floatToRawIntBits(plain.loss()), "loss bits");
          Assert.assertTrue(worst <= TOLERANCE, "gradient difference " + worst);
          Assert.assertEquals(recorder.marks, expectedMarks(depth));
        }
      }
    }
  }

  /** 推論の順伝播には計測の印を渡せない。 */
  @Test
  public void inferenceRejectsAProbe() {
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new Pass(Precision.FLOAT32, null, null, null, new Recorder()));
  }

  /**
   * 深さ L の網の 1 微小バッチの印の順（順伝播、損失の逆伝播、葉と注意を積んだ順の逆）。頭の中の葉は作った順が y_T[τ] の gather・t0・
   * 交差注意の q・K/V・出力・補助の頭の決定の行と読みの行の gather で、埋め込みの欄の並べは型の順に 4 つ。
   */
  private static List<String> expectedMarks(int depth) {
    List<String> marks = new ArrayList<>();
    marks.add(name(Stage.BATCH_INDEX, 0));
    marks.add(name(Stage.EMBED, 0));
    marks.add(name(Stage.RELATION, 0));
    for (int layer = 0; layer < depth; layer++) {
      marks.add(name(Stage.NORM_QKV, layer));
      marks.add(name(Stage.ATTENTION, layer));
      marks.add(name(Stage.O_FFN, layer));
    }
    marks.add(name(Stage.FINAL_NORM, 0));
    marks.add(name(Stage.CANDIDATE, 0));
    marks.add(name(Stage.VALUE_GRAPH, 0));
    marks.add(name(Stage.AUX_HEADS, 0));
    marks.add(name(Stage.HEADS_BACKWARD, 0));
    marks.add(name(Stage.GATHER_BACKWARD, 0));
    marks.add(name(Stage.GATHER_BACKWARD, 0));
    marks.add(name(Stage.CROSS_ATTENTION_BACKWARD, 0));
    marks.add(name(Stage.CROSS_PROJECTION_BACKWARD, 0));
    marks.add(name(Stage.CROSS_PROJECTION_BACKWARD, 0));
    marks.add(name(Stage.CANDIDATE_START_BACKWARD, 0));
    marks.add(name(Stage.GATHER_BACKWARD, 0));
    for (int layer = depth - 1; layer >= 0; layer--) {
      // 最後の正規化（最後の層）か次の層の入力の葉、FFN の前の正規化、残差の葉
      for (int injection = 0; injection < 3; injection++) {
        marks.add(name(Stage.O_FFN_BACKWARD, layer));
      }
      marks.add(name(Stage.ATTENTION_BACKWARD, layer));
      marks.add(name(Stage.NORM_QKV_BACKWARD, layer));
    }
    marks.add(name(Stage.EMBED_BACKWARD, 0));
    for (int type = 0; type < TokenType.values().length; type++) {
      marks.add(name(Stage.EMBED_PACK_BACKWARD, 0));
    }
    return marks;
  }

  private static String name(Stage stage, int layer) {
    return stage.layered() ? stage + " L" + layer : stage.name();
  }

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

  /** 自前の注意の記録で 1 回の順伝播と逆伝播をする（印は順伝播の種類と記録に同じものを渡す）。勾配の写しを取り、0 に戻す。 */
  private static Outcome run(
      PrismNetwork network, NDManager manager, Batch batch, FitProbe probe, Precision precision) {
    List<ValueLabel> labels = Batches.values(batch, 11);
    AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 13);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    float value;
    try (NDManager step = manager.newSubManager();
        GradientCollector collector = manager.getEngine().newGradientCollector()) {
      AttentionTape tape = new AttentionTape(new ReferenceAttention(), probe);
      Outputs outputs =
          network.forward(
              new ParameterStore(manager, false),
              batch.inputs(step),
              Pass.training(precision, ValueLabel.measured(labels), auxiliary, tape, probe));
      NDArray total = loss.total(step, outputs, labels, auxiliary, 1f, labels.size(), 1f);
      tape.backward(collector, total);
      value = total.getFloat();
    }
    return new Outcome(value, Batches.takeGradients(network));
  }

  /** max |a − b| / max |a|（形が違うか NaN が出れば無限大、a が全て 0 なら差の最大）。ビットで同じなら 0。 */
  private static double relativeDifference(float[] expected, float[] actual) {
    if (expected.length != actual.length) {
      return Double.POSITIVE_INFINITY;
    }
    double scale = 0;
    double difference = 0;
    for (int i = 0; i < expected.length; i++) {
      scale = Math.max(scale, Math.abs(expected[i]));
      difference = Math.max(difference, Math.abs((double) expected[i] - actual[i]));
    }
    double relative = scale == 0 ? difference : difference / scale;
    // Math.max は NaN を通すので、NaN の勾配が呼び手の比べ（>）をすり抜けないように無限大にする
    return Double.isNaN(relative) ? Double.POSITIVE_INFINITY : relative;
  }
}

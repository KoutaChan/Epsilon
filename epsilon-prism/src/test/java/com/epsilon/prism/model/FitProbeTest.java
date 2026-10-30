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
 * 段の時計の印（{@link FitProbe}）: 計測のときに背骨の出力も葉にして逆伝播を頭の側と背骨の側に分けても、損失と全てのパラメーターの勾配が
 * 印のない学習とビットで同じこと（CPU、自前の注意の記録に参照の実装 {@link ReferenceAttention}、FP32 と BF16、深さ 2 と 3）。印は
 * 1 微小バッチの順伝播と逆伝播の区切りを決まった順で通る。
 */
public class FitProbeTest {
  private static final Logger log = LoggerFactory.getLogger(FitProbeTest.class);

  private static final int[] EVENTS = {0, 3, 9, 17};

  /** 印の段と層を順に覚え、背骨の出力も葉にする印。 */
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

  @Test(timeOut = 600_000)
  public void splitBackwardKeepsTheGradientsBitwise() {
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
          for (Map.Entry<String, float[]> entry : plain.gradients().entrySet()) {
            if (!sameBits(entry.getValue(), split.gradients().get(entry.getKey()))) {
              differing++;
            }
          }
          log.info(
              "PRISM-PROBE depth={} {} loss {} vs {} gradients differing={} of {} marks={}",
              depth,
              precision,
              String.format("%.7f", plain.loss()),
              String.format("%.7f", split.loss()),
              differing,
              plain.gradients().size(),
              recorder.marks);
          Assert.assertEquals(
              Float.floatToRawIntBits(split.loss()), Float.floatToRawIntBits(plain.loss()), "loss bits");
          Assert.assertEquals(differing, 0, "gradients that differ in bits");
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

  /** 深さ L の網の 1 微小バッチの印の順（順伝播、損失の逆伝播、葉と注意を積んだ順の逆）。 */
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
    marks.add(name(Stage.O_FFN_BACKWARD, depth - 1));
    for (int layer = depth - 1; layer >= 0; layer--) {
      marks.add(name(Stage.ATTENTION_BACKWARD, layer));
      marks.add(name(Stage.NORM_QKV_BACKWARD, layer));
      marks.add(layer > 0 ? name(Stage.O_FFN_BACKWARD, layer - 1) : name(Stage.EMBED_BACKWARD, 0));
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

  private static boolean sameBits(float[] expected, float[] actual) {
    if (expected.length != actual.length) {
      return false;
    }
    for (int i = 0; i < expected.length; i++) {
      if (Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(actual[i])) {
        return false;
      }
    }
    return true;
  }
}

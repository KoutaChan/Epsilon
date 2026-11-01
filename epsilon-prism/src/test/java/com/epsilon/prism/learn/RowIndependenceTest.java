package com.epsilon.prism.learn;

import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.Uploads;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.rows.Row;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 書き手（Tokenizer・BatchPlan）の塊の並びを網が正しく読むこと。実際の判断の O を何行も 1 つの塊に書いたときの s が、1 行ずつの塊の s と
 * 最後のビットの違いを除いて同じ（FP32 の CPU）。関係の属性（型ごとの区画の並び）の読み違いや、行をまたぐ取り違えがあれば崩れる。
 */
public class RowIndependenceTest {
  private static final Logger log = LoggerFactory.getLogger(RowIndependenceTest.class);

  private static final PrismShape SHAPE = new PrismShape(64, 128, 64, 2, false);

  @Test(timeOut = 600_000)
  public void scoresDoNotDependOnTheOtherRowsInTheBatch() {
    List<Row> rows = SyntheticCohort.rows(900, 1, 5);
    assertTrue(rows.size() >= 8, "rows " + rows.size());
    FitPlan plan =
        new FitPlan(
            new FitSettings(1, 4096, 3e-3, 3e-4, 1f, 0.01, 0.02),
            SHAPE,
            AuxiliaryLoss.standard(),
            1_000_000,
            5);
    Encoder encoder = new Encoder(plan, Teacher.RAW);
    Observation observation = new Observation();

    HostInputs together = batch(rows.size());
    together.plan().clear();
    for (Row row : rows) {
      assertTrue(together.plan().add(observation.bind(row.observation(), 0)) >= 0);
    }
    together.plan().seal();
    Microbatch all = encoder.encode(together, rows);

    try (Model model = Fit.blank(SHAPE, Device.cpu(), 7);
        NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = (PrismNetwork) model.getBlock();
      perturb(network, 0.05f, 13);
      float[] joint = scores(network, all, manager);
      int jointWidth = joint.length / rows.size();
      HostInputs single = batch(1);
      int compared = 0;
      double worst = 0;
      double largest = 0;
      for (int r = 0; r < rows.size(); r++) {
        Row row = rows.get(r);
        int legal = observation.bind(row.observation(), 0).legalActions();
        single.plan().clear();
        single.plan().add(observation);
        single.plan().seal();
        float[] alone = scores(network, encoder.encode(single, List.of(row)), manager);
        for (int a = 0; a < legal; a++) {
          float expected = alone[a];
          float actual = joint[r * jointWidth + a];
          double difference = Math.abs(actual - expected);
          worst = Math.max(worst, difference);
          largest = Math.max(largest, Math.abs(expected));
          // CPU の行列積は行の数で計算の分け方が変わるので、最後のビットの違いだけを許す（並びの読み違いは桁が違う）
          assertTrue(
              difference <= 1e-5 + 1e-4 * Math.abs(expected),
              "row " + r + " action " + a + ": " + actual + " vs " + expected);
          compared++;
        }
      }
      assertTrue(compared > 0);
      assertTrue(largest > 1e-3, "scores must be non-trivial: " + largest);
      log.info(
          "PRISM-ROWS rows={} actions={} max|diff|={} max|s|={}",
          rows.size(),
          compared,
          String.format("%.3g", worst),
          String.format("%.3g", largest));
    }
  }

  private static HostInputs batch(int rows) {
    BatchPlan.Capacity capacity =
        new BatchPlan.Capacity(
            rows,
            rows * Limits.MAX_EVENT_TOKENS,
            rows * Limits.MAX_TRANSITIONS,
            rows * Limits.MAX_LEGAL_ACTIONS);
    return HostInputs.allocate(capacity, false, ByteBuffer::allocateDirect);
  }

  private static float[] scores(PrismNetwork network, Microbatch part, NDManager manager) {
    try (NDManager scope = manager.newSubManager()) {
      return network
          .forward(
              new ParameterStore(scope, false),
              part.inputs(scope, Uploads.on(scope)),
              Pass.inference(Precision.FLOAT32))
          .scores()
          .toFloatArray();
    }
  }

  /** 白紙の網は s ≡ 0 なので、全てのパラメーターを少し動かして s を 0 でなくする。 */
  private static void perturb(PrismNetwork network, float scale, long seed) {
    Random random = new Random(seed);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      float[] values = array.toFloatArray();
      for (int i = 0; i < values.length; i++) {
        values[i] += (float) random.nextGaussian() * scale;
      }
      array.set(values);
    }
  }
}

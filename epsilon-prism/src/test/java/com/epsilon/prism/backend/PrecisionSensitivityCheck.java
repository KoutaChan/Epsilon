package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 学んだ版（-Dprism.snapshot）の s の BF16 の丸めへの弱さを CPU で測る（診断。性質がなければ飛ばす）。本物の半荘の判断の行で
 * eager の s を FP32 と BF16 で出し、|s|・行の中で中心化した |s|・1 位と 2 位の差・FP32 と BF16 の差・最善手の一致率を出す。
 * HIP と eager の s の食い違いが、網の BF16 の雑音か HIP の不具合かを分けるのに使う。
 */
public class PrecisionSensitivityCheck {
  private static final Logger log = LoggerFactory.getLogger(PrecisionSensitivityCheck.class);

  /** CPU の packed 注意が [行, S, S, 頭 + A] を作るので、塊は小さくする。 */
  private static final int BATCH_ROWS = 24;

  private static final int ROWS = 480;

  @Test
  public void measure() throws IOException {
    String given = System.getProperty("prism.snapshot");
    if (given == null) {
      throw new SkipException("needs -Dprism.snapshot");
    }
    List<long[]> rows = new ArrayList<>();
    rows.addAll(HipFixtures.observations(31, 6, DecisionWalk.Policy.GREEDY, ROWS / 2));
    rows.addAll(HipFixtures.observations(37, 6, DecisionWalk.Policy.CALLS, ROWS / 2));
    List<float[]> exact = new ArrayList<>();
    List<float[]> rounded = new ArrayList<>();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = PrismNetwork.load(Snapshot.model(Path.of(given)), manager);
      ParameterStore store = new ParameterStore(manager, false);
      for (int from = 0; from < rows.size(); from += BATCH_ROWS) {
        HostBatch batch = HipFixtures.encode(rows.subList(from, Math.min(rows.size(), from + BATCH_ROWS)));
        exact.addAll(scores(network, store, manager, batch, Precision.FLOAT32));
        rounded.addAll(scores(network, store, manager, batch, Precision.BFLOAT16));
      }
    }
    double absolute = 0, centered = 0, gapSum = 0;
    int actions = 0, considered = 0, agree = 0, narrow = 0, confident = 0, confidentAgree = 0;
    List<Double> differences = new ArrayList<>();
    List<Double> centeredDifferences = new ArrayList<>();
    for (int row = 0; row < exact.size(); row++) {
      float[] e = exact.get(row);
      float[] r = rounded.get(row);
      if (e.length < 2) {
        continue;
      }
      double meanE = mean(e), meanR = mean(r);
      for (int a = 0; a < e.length; a++) {
        absolute += Math.abs(e[a]);
        centered += Math.abs(e[a] - meanE);
        differences.add(Math.abs((double) e[a] - r[a]));
        centeredDifferences.add(Math.abs((e[a] - meanE) - (r[a] - meanR)));
        actions++;
      }
      int bestE = argmax(e), bestR = argmax(r);
      double gap = gap(e);
      considered++;
      gapSum += gap;
      if (gap < 0.005) {
        narrow++;
      }
      if (bestE == bestR) {
        agree++;
      }
      if (gap >= 0.005) {
        confident++;
        if (bestE == bestR) {
          confidentAgree++;
        }
      }
    }
    log.info(
        "PRISM-PRECISION snapshot={} rows={} actions={} mean|s|={} mean|s-rowmean|={} meanGap={} narrow(<0.005)={}"
            + " |FP32-BF16| median={} p99={} centered median={} p99={} argmax={} confident={} (rows {})",
        given,
        considered,
        actions,
        String.format(Locale.ROOT, "%.4f", absolute / actions),
        String.format(Locale.ROOT, "%.4f", centered / actions),
        String.format(Locale.ROOT, "%.4f", gapSum / considered),
        String.format(Locale.ROOT, "%.3f", (double) narrow / considered),
        String.format(Locale.ROOT, "%.3e", quantile(differences, 0.5)),
        String.format(Locale.ROOT, "%.3e", quantile(differences, 0.99)),
        String.format(Locale.ROOT, "%.3e", quantile(centeredDifferences, 0.5)),
        String.format(Locale.ROOT, "%.3e", quantile(centeredDifferences, 0.99)),
        String.format(Locale.ROOT, "%.4f", (double) agree / considered),
        String.format(Locale.ROOT, "%.4f", confident == 0 ? Double.NaN : (double) confidentAgree / confident),
        confident);
  }

  /** 塊の行ごとの s（行の手の数の長さ）。 */
  private static List<float[]> scores(
      PrismNetwork network, ParameterStore store, NDManager manager, HostBatch batch, Precision precision) {
    try (NDManager scope = manager.newSubManager()) {
      Outputs outputs = network.forward(store, EagerStages.inputs(scope, batch), Pass.inference(precision));
      float[] flat = outputs.scores().toFloatArray();
      int capacity = Math.toIntExact(outputs.scores().getShape().get(1));
      List<float[]> rows = new ArrayList<>(outputs.actions().length);
      for (int row = 0; row < outputs.actions().length; row++) {
        rows.add(Arrays.copyOfRange(flat, row * capacity, row * capacity + outputs.actions()[row]));
      }
      return rows;
    }
  }

  private static double mean(float[] values) {
    double sum = 0;
    for (float value : values) {
      sum += value;
    }
    return sum / values.length;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int a = 1; a < values.length; a++) {
      if (values[a] > values[best]) {
        best = a;
      }
    }
    return best;
  }

  private static double gap(float[] values) {
    float[] sorted = values.clone();
    Arrays.sort(sorted);
    return sorted[sorted.length - 1] - sorted[sorted.length - 2];
  }

  private static double quantile(List<Double> values, double q) {
    double[] sorted = values.stream().mapToDouble(Double::doubleValue).sorted().toArray();
    return sorted.length == 0 ? Double.NaN : sorted[Math.min(sorted.length - 1, (int) (q * sorted.length))];
  }
}

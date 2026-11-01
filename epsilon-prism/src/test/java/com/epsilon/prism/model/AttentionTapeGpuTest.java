package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.model.Batches.Batch;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 自前の HIP の注意（{@link AttentionTape}）で学ぶときの損失・s・全てのパラメーターの勾配が、djl-rocm の演算の道と同じこと（rocm の組。
 * GPU 1 枚）。標準の形の深さ 2 と 3、出来事の数（S）の違う行を混ぜた塊で、乱した網を比べる。
 *
 * <ul>
 *   <li>BF16: HIP の道（関係の符号の T0・注意の t1、勾配の VJP の注入）。違いは注意の FP32 の和の順と、そこから来る BF16 の丸めの違いだけ
 *       なので、テンソルごとの相対の L2 の差で見る。つなぎと演算を分けるため、ホストの参照の実装（{@link ReferenceAttention}）を差した道も
 *       出す。1 歩の時間も djl と HIP の道で出す。
 *   <li>FP32: 参照の実装を差した道。HIP の演算は BF16 だけなので、記録のつなぎ（葉と VJP の注入）だけを和の順の違いの幅で見る。
 *   <li>精度の非劣性（{@link #hipIsAsAccurateAsTheReference}）: 種を変えた多くの網で、BF16 の HIP の道を約束どおりの参照の道（FP64 の
 *       和を 1 回丸める {@link ReferenceAttention}）と、FP32 の真の値からの差で対にして比べる。BF16 の 2 つの道の間の差は精度の尺度で
 *       ないので、上の比べの幅は目安で、判定はこちら。網の全体の差は注意の外の BF16 の丸めが大きいので、op の段の FP64 の検査
 *       （{@link HipAttentionGpuTest}）の代わりにはならない。
 * </ul>
 */
public class AttentionTapeGpuTest {
  private static final Logger log = LoggerFactory.getLogger(AttentionTapeGpuTest.class);

  private static final int ROWS = 96;
  private static final int[] EVENTS = {0, 12, 35, 60, 90, 117};
  private static final int SEEDS = 16;

  /** 非劣性の幅: 差 e が 5% 悪いまで（対数の比 ln 1.05）。 */
  private static final double MARGIN = Math.log(1.05);

  /** 注意に接するパラメーター（層ごとの qkv の射影・関係の表 ρ・O の射影）。 */
  private static final Pattern ATTENTION = Pattern.compile("encoder\\d+_(qkv|qkvBias|relation|output|outputBias)$");

  /** 判定する差の種類（s・全ての勾配・注意に接する勾配）。 */
  private enum Metric {
    SCORES,
    GRADIENTS,
    ATTENTION
  }

  @Test(groups = "rocm")
  public void tapeMatchesTheDjlOperator() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    for (int depth = PrismShape.MIN_DEPTH; depth <= PrismShape.MAX_DEPTH; depth++) {
      compare(depth, Precision.BFLOAT16);
    }
  }

  @Test(groups = "rocm")
  public void tapeMatchesTheDjlOperatorInFloat32() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    for (int depth = PrismShape.MIN_DEPTH; depth <= PrismShape.MAX_DEPTH; depth++) {
      compare(depth, Precision.FLOAT32);
    }
  }

  /**
   * BF16 の HIP の道が約束どおりの参照の道より不正確でないこと（非劣性）。種 0〜{@value #SEEDS} − 1 の網（初期化・乱し・塊が種で変わり、
   * ラベルは塊で変わる）を深さ 2 と 3 で作り、FP32 の djl の道を真の値として（host-fp32 との差は BF16 の道の差より桁違いに小さい）、
   * BF16 の djl・参照（host）・HIP の道の差 e（s、全ての勾配、注意に接する勾配の相対の L2）を出す。
   *
   * <p>前もって決めた判定（{@link Metric} ごと）: 種ごとに x_s = 深さ 2 と 3 の ln(e_hip/e_host) の平均（同じ種の 2 つの深さは塊を
   * 共有するので 1 つの単位）、n = {@value #SEEDS}。平均 + 2·標準誤差 ≤ ln 1.05 なら PASS、平均 − 2·標準誤差 > ln 1.05 なら FAIL、
   * ほかは INCONCLUSIVE（幅が揺れより狭くて決まらない）。試験は FAIL だけで止める。対照に ln(e_hip/e_djl) と、約束に近い 2 つの道の
   * 間の揺れの目安 ln(e_djl/e_host) も出す。行は種ごとと注意のテンソルごと。
   */
  @Test(groups = "rocm")
  public void hipIsAsAccurateAsTheReference() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    int depths = PrismShape.MAX_DEPTH - PrismShape.MIN_DEPTH + 1;
    Map<Metric, double[]> hipHost = new EnumMap<>(Metric.class);
    Map<Metric, double[]> hipDjl = new EnumMap<>(Metric.class);
    Map<Metric, double[]> djlHost = new EnumMap<>(Metric.class);
    for (Metric metric : Metric.values()) {
      hipHost.put(metric, new double[SEEDS]);
      hipDjl.put(metric, new double[SEEDS]);
      djlHost.put(metric, new double[SEEDS]);
    }
    Device gpu = Device.gpu(0);
    for (int seed = 0; seed < SEEDS; seed++) {
      for (int depth = PrismShape.MIN_DEPTH; depth <= PrismShape.MAX_DEPTH; depth++) {
        try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
            PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
            PtStreamScope ignored = stream.openScope()) {
          PrismNetwork network = network(manager, depth, seed);
          Batch batch = batch(seed);
          Outcome truth = run(network, manager, batch, null, Precision.FLOAT32);
          Outcome djl = run(network, manager, batch, null, Precision.BFLOAT16);
          Outcome host = run(network, manager, batch, new ReferenceAttention(), Precision.BFLOAT16);
          Outcome hip = run(network, manager, batch, HipAttention.on(HipLibrary.load(), stream), Precision.BFLOAT16);
          Map<Metric, Double> eDjl = errors(truth, djl);
          Map<Metric, Double> eHost = errors(truth, host);
          Map<Metric, Double> eHip = errors(truth, hip);
          log.info(
              "PRISM-TAPE-ACCURACY depth={} seed={} scores djl={} host={} hip={} gradients djl={} host={}"
                  + " hip={} attention djl={} host={} hip={} hip-host scores={} loss fp32={} djl={}"
                  + " host={} hip={}",
              depth,
              seed,
              String.format("%.6f", eDjl.get(Metric.SCORES)),
              String.format("%.6f", eHost.get(Metric.SCORES)),
              String.format("%.6f", eHip.get(Metric.SCORES)),
              String.format("%.6f", eDjl.get(Metric.GRADIENTS)),
              String.format("%.6f", eHost.get(Metric.GRADIENTS)),
              String.format("%.6f", eHip.get(Metric.GRADIENTS)),
              String.format("%.6f", eDjl.get(Metric.ATTENTION)),
              String.format("%.6f", eHost.get(Metric.ATTENTION)),
              String.format("%.6f", eHip.get(Metric.ATTENTION)),
              String.format("%.6f", relative(host.scores(), hip.scores())),
              String.format("%.6f", truth.loss()),
              String.format("%.6f", djl.loss()),
              String.format("%.6f", host.loss()),
              String.format("%.6f", hip.loss()));
          for (String name : truth.gradients().keySet()) {
            if (ATTENTION.matcher(name).find()) {
              float[] expected = truth.gradients().get(name);
              log.info(
                  "PRISM-TAPE-TENSOR depth={} seed={} {} host={} hip={}",
                  depth,
                  seed,
                  name,
                  String.format("%.6f", relative(expected, host.gradients().get(name))),
                  String.format("%.6f", relative(expected, hip.gradients().get(name))));
            }
          }
          for (Metric metric : Metric.values()) {
            hipHost.get(metric)[seed] += Math.log(eHip.get(metric) / eHost.get(metric)) / depths;
            hipDjl.get(metric)[seed] += Math.log(eHip.get(metric) / eDjl.get(metric)) / depths;
            djlHost.get(metric)[seed] += Math.log(eDjl.get(metric) / eHost.get(metric)) / depths;
          }
        }
      }
    }
    StringBuilder failures = new StringBuilder();
    for (Metric metric : Metric.values()) {
      String verdict = verdict(hipHost.get(metric));
      log.info(
          "PRISM-TAPE-ACCURACY-SUMMARY metric={} ln(e_hip/e_host) {} verdict={} | ln(e_hip/e_djl) {} |"
              + " ln(e_djl/e_host) {}",
          metric,
          summary(hipHost.get(metric)),
          verdict,
          summary(hipDjl.get(metric)),
          summary(djlHost.get(metric)));
      if (verdict.equals("FAIL")) {
        failures.append(' ').append(metric);
      }
    }
    Assert.assertTrue(failures.isEmpty(), "HIP less accurate than the reference attention:" + failures);
  }

  private record Outcome(float loss, float[] scores, Map<String, float[]> gradients, double millis) {}

  /** 種 seed の乱した網（種 0 は {@link #tapeMatchesTheDjlOperator} の網）。 */
  private static PrismNetwork network(NDManager manager, int depth, int seed) {
    manager.getEngine().setRandomSeed(3 + seed);
    PrismNetwork network = new PrismNetwork(PrismShape.standard(depth));
    network.initialize(manager);
    Batches.perturb(network, 0.05f, 7 + seed);
    return network;
  }

  /** 出来事の数の違う行と 3 つの判断の種類を混ぜた種 seed の塊（種 0 は {@link #tapeMatchesTheDjlOperator} の塊）。 */
  private static Batch batch(int seed) {
    List<Batches.Row> rows = new ArrayList<>();
    for (int r = 0; r < ROWS; r++) {
      int events = EVENTS[r % EVENTS.length];
      rows.add(
          switch (r % 4) {
            case 0, 1 -> Batches.turn(events);
            case 2 -> Batches.response(events);
            default -> Batches.postCall(events);
          });
    }
    return Batches.build(rows, 5 + seed);
  }

  private static void compare(int depth, Precision precision) {
    Device gpu = Device.gpu(0);
    try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
        PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
        PtStreamScope ignored = stream.openScope()) {
      PrismNetwork network = network(manager, depth, 0);
      Batch batch = batch(0);
      Outcome reference = run(network, manager, batch, null, precision);
      Outcome host = run(network, manager, batch, new ReferenceAttention(), precision);
      Assert.assertTrue(Batches.maxAbs(reference.scores()) > 1e-3f, "scores must be non-trivial");
      if (precision == Precision.FLOAT32) {
        double total = report(depth, "host-fp32", reference, host);
        Assert.assertEquals(host.loss(), reference.loss(), 1e-5f * Math.abs(reference.loss()));
        Assert.assertTrue(relative(reference.scores(), host.scores()) < 1e-5, "scores");
        Assert.assertTrue(total < 1e-4, "gradients rel " + total);
        return;
      }
      HipAttention attention = HipAttention.on(HipLibrary.load(), stream);
      Outcome hip = run(network, manager, batch, attention, precision);
      // 2 回目は温まった後の時間
      double referenceMillis = run(network, manager, batch, null, precision).millis();
      double hipMillis = run(network, manager, batch, attention, precision).millis();

      report(depth, "host", reference, host);
      double total = report(depth, "hip", reference, hip);
      log.info(
          "PRISM-TAPE depth={} rows={} ms/step djl={} hip={}",
          depth,
          ROWS,
          String.format("%.1f", referenceMillis),
          String.format("%.1f", hipMillis));
      Assert.assertEquals(hip.loss(), reference.loss(), 1e-3f * Math.abs(reference.loss()) + 1e-5f);
      Assert.assertTrue(relative(reference.scores(), hip.scores()) < 1e-2, "scores");
      Assert.assertTrue(total < 2e-2, "gradients rel " + total);
    }
  }

  /** 道の損失・s・勾配を djl の道と比べて出す（パラメーターごとの差の大きい順に 12 個と、勾配の全体の相対の差）。 */
  private static double report(int depth, String name, Outcome reference, Outcome other) {
    double error = 0;
    double norm = 0;
    List<String> lines = new ArrayList<>();
    List<Double> ratios = new ArrayList<>();
    for (Map.Entry<String, float[]> entry : reference.gradients().entrySet()) {
      float[] expected = entry.getValue();
      float[] actual = other.gradients().get(entry.getKey());
      double tensorNorm = squared(expected);
      double tensorError = squaredDifference(expected, actual);
      norm += tensorNorm;
      error += tensorError;
      double ratio = Math.sqrt(tensorError / Math.max(tensorNorm, 1e-30));
      lines.add(
          String.format(
              "%s rel=%.3g |ref|=%.3g |other|=%.3g",
              entry.getKey(), ratio, Math.sqrt(tensorNorm), Math.sqrt(squared(actual))));
      ratios.add(ratio);
    }
    double total = Math.sqrt(error / Math.max(norm, 1e-30));
    log.info(
        "PRISM-TAPE depth={} {} loss {} vs {} scores rel={} gradients rel={}",
        depth,
        name,
        String.format("%.6f", reference.loss()),
        String.format("%.6f", other.loss()),
        String.format("%.3g", relative(reference.scores(), other.scores())),
        String.format("%.3g", total));
    Integer[] order = new Integer[lines.size()];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    Arrays.sort(order, (a, b) -> Double.compare(ratios.get(b), ratios.get(a)));
    for (int k = 0; k < Math.min(12, order.length); k++) {
      log.info("PRISM-TAPE   depth={} {} {}", depth, name, lines.get(order[k]));
    }
    return total;
  }

  /** 1 回の順伝播と逆伝播（attention が null なら djl-rocm の演算の道）。勾配の写しを取り、0 に戻す。 */
  private static Outcome run(
      PrismNetwork network,
      NDManager manager,
      Batch batch,
      TrainingKernels attention,
      Precision precision) {
    List<ValueLabel> labels = Batches.values(batch, 11);
    AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 13);
    boolean[] measured = ValueLabel.measured(labels);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    long started = System.nanoTime();
    float value;
    float[] scores;
    try (NDManager step = manager.newSubManager();
        GradientCollector collector = manager.getEngine().newGradientCollector()) {
      AttentionTape tape = attention == null ? null : new AttentionTape(attention);
      Outputs outputs =
          network.forward(
              new ParameterStore(manager, false),
              batch.inputs(step),
              Pass.training(precision, measured, auxiliary, tape));
      NDArray total = loss.total(step, outputs, labels, auxiliary, 1f, labels.size(), 1f);
      if (tape == null) {
        collector.backward(total);
      } else {
        tape.backward(collector, total);
      }
      value = total.getFloat();
      scores = outputs.scores().toFloatArray();
    }
    Map<String, float[]> gradients = Batches.takeGradients(network);
    return new Outcome(value, scores, gradients, (System.nanoTime() - started) / 1e6);
  }

  /** 真の値 truth からの道 other の差 e（s、全ての勾配を合わせた相対の L2、注意に接する勾配を合わせた相対の L2）。 */
  private static Map<Metric, Double> errors(Outcome truth, Outcome other) {
    double[] error = new double[2];
    double[] norm = new double[2];
    int tensors = 0;
    for (Map.Entry<String, float[]> entry : truth.gradients().entrySet()) {
      double tensorNorm = squared(entry.getValue());
      double tensorError = squaredDifference(entry.getValue(), other.gradients().get(entry.getKey()));
      norm[0] += tensorNorm;
      error[0] += tensorError;
      if (ATTENTION.matcher(entry.getKey()).find()) {
        norm[1] += tensorNorm;
        error[1] += tensorError;
        tensors++;
      }
    }
    Assert.assertTrue(tensors > 0, "no attention parameters in " + truth.gradients().keySet());
    Map<Metric, Double> errors = new EnumMap<>(Metric.class);
    errors.put(Metric.SCORES, relative(truth.scores(), other.scores()));
    errors.put(Metric.GRADIENTS, Math.sqrt(error[0] / Math.max(norm[0], 1e-30)));
    errors.put(Metric.ATTENTION, Math.sqrt(error[1] / Math.max(norm[1], 1e-30)));
    return errors;
  }

  /** 前もって決めた三つの判定（平均 ± 2·標準誤差を幅 ln 1.05 と比べる）。 */
  private static String verdict(double[] values) {
    double mean = mean(values);
    double error = standardError(values);
    if (mean + 2 * error <= MARGIN) {
      return "PASS";
    }
    return mean - 2 * error > MARGIN ? "FAIL" : "INCONCLUSIVE";
  }

  private static String summary(double[] values) {
    double mean = mean(values);
    double error = standardError(values);
    return String.format(
        "n=%d mean=%+.4f se=%.4f sd=%.4f mean+2se=%+.4f mean-2se=%+.4f margin=%.4f min=%+.4f max=%+.4f",
        values.length,
        mean,
        error,
        error * Math.sqrt(values.length),
        mean + 2 * error,
        mean - 2 * error,
        MARGIN,
        Arrays.stream(values).min().orElse(0),
        Arrays.stream(values).max().orElse(0));
  }

  private static double mean(double[] values) {
    return Arrays.stream(values).average().orElse(0);
  }

  /** 標本の標準偏差（n − 1）/ √n。 */
  private static double standardError(double[] values) {
    double mean = mean(values);
    double sum = 0;
    for (double value : values) {
      sum += (value - mean) * (value - mean);
    }
    return Math.sqrt(sum / Math.max(values.length - 1, 1) / values.length);
  }

  private static double relative(float[] expected, float[] actual) {
    return Math.sqrt(squaredDifference(expected, actual) / Math.max(squared(expected), 1e-30));
  }

  private static double squared(float[] values) {
    double sum = 0;
    for (float value : values) {
      sum += (double) value * value;
    }
    return sum;
  }

  private static double squaredDifference(float[] expected, float[] actual) {
    double sum = 0;
    for (int i = 0; i < expected.length; i++) {
      double difference = (double) expected[i] - actual[i];
      sum += difference * difference;
    }
    return sum;
  }
}

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
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism4p.input.InputSchema;
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
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 自前の HIP の Attention（{@link AttentionTape}）で学習するときの損失・s・全てのパラメーターの勾配が、djl-rocm の演算を使う経路と
 * 同じことを確かめる（rocm のグループ。GPU 1 枚）。標準の形の深さ 2 と 3、型ごとの FFN のアーム（{@link ShapeArm} の v3 のアーム）で、
 * 乱したネットワークを比べる。バッチにはイベントの数（S）の違う行を混ぜる。どの経路でも、勾配がちょうど 0 のパラメーターはない。
 *
 * <ul>
 *   <li>BF16: HIP の経路（関係のコードの T0、Attention の t1、正規化の t2、勾配の VJP の注入）。違うのは Attention と正規化の FP32 の
 *       和の順と、そこから来る BF16 の丸めだけなので、テンソルごとの相対 L2 の差で見る。つなぎ込みの誤りと演算の誤りを見分けるため、
 *       ホストの参照の実装（{@link ReferenceAttention}）を差し込んだ経路も出す。1 ステップの時間も djl と HIP の経路で出す。型ごとの FFN の
 *       アームは、丸めのずれの揺れを均すため、複数の種の平均で見る（{@link #typedTapeMatchesTheDjlOperator}）。
 *   <li>FP32: 参照の実装を差し込んだ経路。HIP の演算は BF16 だけなので、記録のつなぎ込み（葉と VJP の注入）だけを、和の順の違いの幅で見る。
 *   <li>精度の非劣性（{@link #hipIsAsAccurateAsTheReference}）: 種を変えた多くのネットワークで、BF16 の HIP の経路と、契約どおりの参照の
 *       経路（FP64 の和を 1 回だけ丸める {@link ReferenceAttention}）を、FP32 の真の値からの差で対にして比べる。BF16 の 2 つの経路の間の差は
 *       精度の尺度ではないので、上の比べの幅は目安で、判定はこちらで行う。ネットワーク全体の差では Attention の外の BF16 の丸めが大きいので、
 *       演算単位の FP64 の検査（{@link HipAttentionGpuTest}）の代わりにはならない。
 * </ul>
 */
public class AttentionTapeGpuTest {
  private static final Logger log = LoggerFactory.getLogger(AttentionTapeGpuTest.class);

  private static final int ROWS = 96;
  private static final int[] EVENTS = {0, 12, 35, 60, 90, 117};
  private static final int SEEDS = 16;

  /** VRAM のピークを比べるバッチの行の数（本番のマイクロバッチに近いトークンの数になる）。 */
  private static final int PEAK_ROWS = 1024;

  /** 型ごとの FFN のアームで比べるネットワークの種の数。 */
  private static final int TYPED_SEEDS = 8;

  /** 非劣性の幅: 差 e が 5% 悪いまで（対数の比 ln 1.05）。 */
  private static final double MARGIN = Math.log(1.05);

  /** Attention に接するパラメーター（層ごとの qkv の射影、関係の表 ρ、O の射影）。 */
  private static final Pattern ATTENTION = Pattern.compile("encoder\\d+_(qkv|qkvBias|relation|output|outputBias)$");

  /** 判定する差の種類（s、全ての勾配、Attention に接する勾配）。 */
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
      compare(PrismShape.standard(depth), Precision.BFLOAT16);
    }
  }

  @Test(groups = "rocm")
  public void tapeMatchesTheDjlOperatorInFloat32() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    for (int depth = PrismShape.MIN_DEPTH; depth <= PrismShape.MAX_DEPTH; depth++) {
      compare(PrismShape.standard(depth), Precision.FLOAT32);
    }
  }

  /**
   * 型ごとの FFN のアーム（BF16 の HIP の経路を djl-rocm の経路と比べる）。2 つの経路の差は BF16 の丸めのずれの揺れで、1 つのネットワーク
   * では 1e-2 を超えることもある（v2 でも同じ）。そこで {@value #TYPED_SEEDS} 個の種の平均をしきい値と比べる: 損失 ≤ 1e-3、s ≤ 1e-2、
   * 勾配の全体 ≤ 2e-2。FP32 の djl の経路を真の値として、HIP の経路の s のずれの平均が djl の経路の 1.25 倍以内であることも見る。どの種でも、
   * 勾配がちょうど 0 のパラメーターはない。
   */
  @Test(groups = "rocm", dataProvider = "typedArms")
  public void typedTapeMatchesTheDjlOperator(ShapeArm arm) {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    PrismShape shape = arm.shape(ShapeArm.V3_DEPTH);
    double[] loss = new double[TYPED_SEEDS];
    double[] scores = new double[TYPED_SEEDS];
    double[] gradients = new double[TYPED_SEEDS];
    double[] djlRounding = new double[TYPED_SEEDS];
    double[] hipRounding = new double[TYPED_SEEDS];
    Device gpu = Device.gpu(0);
    for (int seed = 0; seed < TYPED_SEEDS; seed++) {
      try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
          PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
          PtStreamScope ignored = stream.openScope()) {
        PrismNetwork network = network(manager, shape, seed);
        Batch batch = batch(seed);
        HipAttention attention = HipAttention.on(HipLibrary.load(), stream);
        Outcome truth = run(network, manager, batch, null, Precision.FLOAT32);
        Outcome djl = run(network, manager, batch, null, Precision.BFLOAT16);
        Outcome hip = run(network, manager, batch, attention, Precision.BFLOAT16);
        Assert.assertEquals(silent(hip), List.of(), arm + " seed " + seed + " HIP gradients that are exactly zero");
        Assert.assertEquals(silent(djl), List.of(), arm + " seed " + seed + " djl gradients that are exactly zero");
        loss[seed] = Math.abs(hip.loss() - djl.loss()) / Math.abs(djl.loss());
        scores[seed] = relative(djl.scores(), hip.scores());
        gradients[seed] = errors(djl, hip).get(Metric.GRADIENTS);
        djlRounding[seed] = relative(truth.scores(), djl.scores());
        hipRounding[seed] = relative(truth.scores(), hip.scores());
        log.info(
            "PRISM-TAPE-TYPED {} seed={} loss {} vs {} rel={} s rel={} gradients rel={} from FP32 s djl={} hip={}",
            arm,
            seed,
            String.format("%.6f", djl.loss()),
            String.format("%.6f", hip.loss()),
            String.format("%.3g", loss[seed]),
            String.format("%.4f", scores[seed]),
            String.format("%.4f", gradients[seed]),
            String.format("%.4f", djlRounding[seed]),
            String.format("%.4f", hipRounding[seed]));
      }
    }
    log.info(
        "PRISM-TAPE-TYPED-SUMMARY {} seeds={} mean loss rel={} s rel={} (max {}) gradients rel={} (max {}) from FP32 s djl={} hip={}",
        arm,
        TYPED_SEEDS,
        String.format("%.3g", mean(loss)),
        String.format("%.4f", mean(scores)),
        String.format("%.4f", Arrays.stream(scores).max().orElse(0)),
        String.format("%.4f", mean(gradients)),
        String.format("%.4f", Arrays.stream(gradients).max().orElse(0)),
        String.format("%.4f", mean(djlRounding)),
        String.format("%.4f", mean(hipRounding)));
    Assert.assertTrue(mean(loss) <= 1e-3, "mean loss rel " + mean(loss));
    Assert.assertTrue(mean(scores) <= 1e-2, "mean scores rel " + mean(scores));
    Assert.assertTrue(mean(gradients) <= 2e-2, "mean gradients rel " + mean(gradients));
    Assert.assertTrue(
        mean(hipRounding) <= 1.25 * mean(djlRounding),
        "the HIP path is further from FP32 (" + mean(hipRounding) + ") than djl (" + mean(djlRounding) + ")");
  }

  /** 型ごとの FFN の V3 のアーム（FP32 の参照の実装の経路で、記録のつなぎ込みを比べる）。 */
  @Test(groups = "rocm")
  public void typedTapeMatchesTheDjlOperatorInFloat32() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    compare(ShapeArm.V3.shape(ShapeArm.V3_DEPTH), Precision.FLOAT32);
  }

  @DataProvider
  public Object[][] typedArms() {
    return new Object[][] {{ShapeArm.V3}, {ShapeArm.V3_ONE_STAGE}, {ShapeArm.V3_CONTEXT_512}};
  }

  /**
   * 学習の順伝播の区切りで途中の配列を閉じると、1 回の順伝播と逆伝播の VRAM のピーク（PyTorch のメモリ確保の記録で見た、ステップの前からの
   * 増え）が、区切りを作らないとき（{@link Context.Intermediates#KEEP}）より下がる（BF16 の HIP の経路、v2 の深さ 2 と V3、
   * {@value #PEAK_ROWS} 行のバッチ）。順伝播の値はビットで同じ。勾配は GPU の逆伝播の和の順（アトミックな足し込み）のせいで走らせるたびに
   * 少し揺れるので、同じ形で 2 回走らせた間の揺れと比べ、それと同じ大きさなら同じとみなす（CPU でビットで同じことは {@link AttentionTapeTest}
   * が見る）。
   */
  @Test(groups = "rocm")
  public void closingIntermediatesLowersTheStepPeak() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    Device gpu = Device.gpu(0);
    for (PrismShape shape : List.of(PrismShape.standard(2), ShapeArm.V3.shape(ShapeArm.V3_DEPTH))) {
      try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
          PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
          PtStreamScope ignored = stream.openScope()) {
        PtEngine engine = (PtEngine) manager.getEngine();
        PrismNetwork network = network(manager, shape, 0);
        Batch batch = batch(0, PEAK_ROWS);
        HipAttention attention = HipAttention.on(HipLibrary.load(), stream);
        // 1 回目は、勾配のバッファとカーネルを用意するためのウォームアップ
        run(network, manager, batch, attention, Precision.BFLOAT16, Context.Intermediates.CLOSE);
        Outcome kept = run(network, manager, batch, attention, Precision.BFLOAT16, Context.Intermediates.KEEP);
        Outcome closed = run(network, manager, batch, attention, Precision.BFLOAT16, Context.Intermediates.CLOSE);
        Outcome again = run(network, manager, batch, attention, Precision.BFLOAT16, Context.Intermediates.CLOSE);
        String label = shape.typed() == null ? "depth=2" : "typed";
        double difference = report(label, "keep-vs-close", kept, closed);
        double wobble = report(label, "close-vs-close", closed, again);
        log.info(
            "PRISM-TAPE-PEAK shape={} rows={} step peak keep={}GB close={}GB gradients rel keep-close={} close-close={}",
            shape,
            PEAK_ROWS,
            String.format("%.3f", kept.peakBytes() / 1e9),
            String.format("%.3f", closed.peakBytes() / 1e9),
            String.format("%.3g", difference),
            String.format("%.3g", wobble));
        Assert.assertEquals(relative(kept.scores(), closed.scores()), 0d, "scores of the same forward");
        Assert.assertTrue(closed.peakBytes() < kept.peakBytes(), "closing must lower the step peak");
        Assert.assertTrue(difference <= Math.max(2 * wobble, 2e-3), "gradients rel " + difference);
        Assert.assertTrue(engine.getMemoryStats(gpu).getOutOfMemoryCount() == 0, "no allocation failed");
      }
    }
  }

  /**
   * BF16 の HIP の経路が、契約どおりの参照の経路より不正確でない（非劣性）。種 0〜{@value #SEEDS} − 1 のネットワーク（初期化・乱し方・
   * バッチが種で変わり、ラベルはバッチで変わる）を深さ 2 と 3 で作る。FP32 の djl の経路を真の値として（host-fp32 との差は BF16 の経路の差
   * より桁違いに小さい）、BF16 の djl・参照（host）・HIP の各経路の差 e（s、全ての勾配、Attention に接する勾配の、それぞれ相対 L2）を出す。
   *
   * <p>判定は前もって決めてある（{@link Metric} ごと）。種ごとに x_s = 深さ 2 と 3 の ln(e_hip/e_host) の平均とする（同じ種の 2 つの深さは
   * バッチを共有するので 1 つの単位）。n = {@value #SEEDS}。平均 + 2·標準誤差 ≤ ln 1.05 なら PASS、平均 − 2·標準誤差 > ln 1.05 なら FAIL、
   * ほかは INCONCLUSIVE（揺れに比べて幅が狭く、決まらない）。テストは FAIL のときだけ止める。対照として ln(e_hip/e_djl) と、契約に近い
   * 2 つの経路の間の揺れの目安 ln(e_djl/e_host) も出す。出力の行は種ごとと Attention のテンソルごと。
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
          PrismNetwork network = network(manager, PrismShape.standard(depth), seed);
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

  /** 1 回の順伝播と逆伝播の結果。{@code peakBytes} は、ステップの前の確保量からの VRAM のピークの増え。 */
  private record Outcome(
      float loss, float[] scores, Map<String, float[]> gradients, double millis, long peakBytes) {}

  /** 種で決まる、乱したネットワーク（種 0 は {@link #tapeMatchesTheDjlOperator} と同じネットワーク）。 */
  private static PrismNetwork network(NDManager manager, PrismShape shape, int seed) {
    manager.getEngine().setRandomSeed(3 + seed);
    PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
    network.initialize(manager);
    Batches.perturb(network, 0.05f, 7 + seed);
    return network;
  }

  /** イベントの数の違う行と 3 つの判断の種類を混ぜた、種で決まるバッチ（種 0 は {@link #tapeMatchesTheDjlOperator} と同じバッチ）。 */
  private static Batch batch(int seed) {
    return batch(seed, ROWS);
  }

  /** イベントの数の違う行と 3 つの判断の種類を混ぜた、種で決まる {@code count} 行のバッチ。 */
  private static Batch batch(int seed, int count) {
    List<Batches.Row> rows = new ArrayList<>();
    for (int r = 0; r < count; r++) {
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

  private static void compare(PrismShape shape, Precision precision) {
    Device gpu = Device.gpu(0);
    String label = shape.typed() == null ? "depth=" + shape.depth() : "typed=" + shape.typed();
    try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
        PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
        PtStreamScope ignored = stream.openScope()) {
      PrismNetwork network = network(manager, shape, 0);
      Batch batch = batch(0);
      Outcome reference = run(network, manager, batch, null, precision);
      Outcome host = run(network, manager, batch, new ReferenceAttention(), precision);
      Assert.assertTrue(Batches.maxAbs(reference.scores()) > 1e-3f, "scores must be non-trivial");
      Assert.assertEquals(silent(reference), List.of(), label + " djl gradients that are exactly zero");
      Assert.assertEquals(silent(host), List.of(), label + " host gradients that are exactly zero");
      if (precision == Precision.FLOAT32) {
        double total = report(label, "host-fp32", reference, host);
        Assert.assertEquals(host.loss(), reference.loss(), 1e-5f * Math.abs(reference.loss()));
        Assert.assertTrue(relative(reference.scores(), host.scores()) < 1e-5, "scores");
        Assert.assertTrue(total < 1e-4, "gradients rel " + total);
        return;
      }
      HipAttention attention = HipAttention.on(HipLibrary.load(), stream);
      Outcome hip = run(network, manager, batch, attention, precision);
      // 2 回目に走らせて、ウォームアップ後の時間を取る
      double referenceMillis = run(network, manager, batch, null, precision).millis();
      double hipMillis = run(network, manager, batch, attention, precision).millis();

      report(label, "host", reference, host);
      double total = report(label, "hip", reference, hip);
      log.info(
          "PRISM-TAPE {} rows={} ms/step djl={} hip={}",
          label,
          ROWS,
          String.format("%.1f", referenceMillis),
          String.format("%.1f", hipMillis));
      Assert.assertEquals(hip.loss(), reference.loss(), 1e-3f * Math.abs(reference.loss()) + 1e-5f);
      Assert.assertTrue(relative(reference.scores(), hip.scores()) < 1e-2, "scores");
      Assert.assertTrue(total < 2e-2, "gradients rel " + total);
      Assert.assertEquals(silent(hip), List.of(), label + " HIP gradients that are exactly zero");
    }
  }

  /** 勾配がちょうど 0 のパラメーター。 */
  private static List<String> silent(Outcome outcome) {
    List<String> names = new ArrayList<>();
    for (Map.Entry<String, float[]> entry : outcome.gradients().entrySet()) {
      if (Batches.maxAbs(entry.getValue()) == 0) {
        names.add(entry.getKey());
      }
    }
    return names;
  }

  /** ある経路の損失・s・勾配を djl の経路と比べてログに出し、勾配の全体の相対の差を返す（ログにはパラメーターごとの差の大きい順に 12 個も出す）。 */
  private static double report(String label, String name, Outcome reference, Outcome other) {
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
        "PRISM-TAPE {} {} loss {} vs {} scores rel={} gradients rel={}",
        label,
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
      log.info("PRISM-TAPE   {} {} {}", label, name, lines.get(order[k]));
    }
    return total;
  }

  /** 1 回の順伝播と逆伝播（{@code attention} が null なら djl-rocm の演算の経路）。勾配の写しを取り、勾配を 0 に戻す。 */
  private static Outcome run(
      PrismNetwork network,
      NDManager manager,
      Batch batch,
      TrainingKernels attention,
      Precision precision) {
    return run(network, manager, batch, attention, precision, Context.Intermediates.CLOSE);
  }

  /** 区切りで途中の配列をどう扱うかを選んで行う、1 回の順伝播と逆伝播。 */
  private static Outcome run(
      PrismNetwork network,
      NDManager manager,
      Batch batch,
      TrainingKernels attention,
      Precision precision,
      Context.Intermediates intermediates) {
    List<ValueLabel> labels = Batches.values(batch, 11);
    AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 13);
    boolean[] measured = ValueLabel.measured(labels);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    PtEngine engine = (PtEngine) manager.getEngine();
    Device device = manager.getDevice();
    engine.resetPeakMemoryStats(device);
    long before = engine.getMemoryStats(device).getAllocatedBytes();
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
    long peak = engine.getMemoryStats(device).getPeakAllocatedBytes() - before;
    Map<String, float[]> gradients = Batches.takeGradients(network);
    return new Outcome(value, scores, gradients, (System.nanoTime() - started) / 1e6, peak);
  }

  /** 真の値からの、ある経路の差 e（s、全ての勾配を合わせた相対 L2、Attention に接する勾配を合わせた相対 L2）。 */
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

  /** 前もって決めた 3 つの判定のどれか（平均 ± 2·標準誤差を幅 ln 1.05 と比べる）。 */
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

  /** 標準誤差: 標本の標準偏差（n − 1 で割る）/ √n。 */
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

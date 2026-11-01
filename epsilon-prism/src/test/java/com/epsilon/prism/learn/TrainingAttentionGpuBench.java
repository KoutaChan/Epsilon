package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.RelationAttention;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.model.AttentionTape;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.Uploads;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 学習の注意 t1（{@link HipAttention}）のカーネルの時間の測り（通常の試験には入れず、サーバの GPU 1 枚で
 * {@code -Dtest=TrainingAttentionGpuBench -Dprism.bench.rows=<rows-v2/t>} で回す）。本物の行を本番の詰め方（{@link StepFeed}、
 * {@value #LANES} レーン、トークンの上限 {@link FitPlan#TOKEN_BUDGET}、最後の塊はレーンの数の部分に分ける）で
 * 微小バッチにし、網の順伝播で背骨の注意に渡る配列（形・在否・関係の符号）をそのまま取って、層ごとに順伝播と逆伝播を温めの後
 * {@value #LAUNCHES} 回ずつ積み、ストリームを待って 1 回あたりの ms を出す。層 0 は問い合わせが S 個、最後の層は局・席・牌の 39 個。
 * レーンごとの歩の和（全ての層の順伝播と逆伝播）と、歩ごとに大きい方のレーンの和の平均も出す。
 */
public class TrainingAttentionGpuBench {
  private static final Logger log = LoggerFactory.getLogger(TrainingAttentionGpuBench.class);

  private static final int THREADS = 16;
  private static final int STEP_ROWS = 4096;
  private static final int STEPS = 4;
  private static final int LANES = 2;
  private static final int WARMUP = 5;
  private static final int LAUNCHES = 50;

  /** 網の順伝播で背骨の注意に渡った配列（層の順）。 */
  private record Call(
      RelationAttention.Dimensions dimensions,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      NDArray out,
      NDArray outFloat,
      NDArray logSumExp) {}

  /** 注意と正規化の演算を通し、注意の順伝播に渡った配列を覚える（記録しない推論の順伝播だけに使う）。 */
  private static final class Capture implements TrainingKernels {

    private final TrainingKernels kernels;
    private final List<Call> calls = new ArrayList<>();

    private Capture(TrainingKernels kernels) {
      this.kernels = kernels;
    }

    @Override
    public void forward(
        Dimensions d,
        NDArray query,
        NDArray keyValue,
        NDArray mask,
        NDArray codes,
        NDArray table,
        NDArray out,
        NDArray outFloat,
        NDArray logSumExp) {
      kernels.forward(d, query, keyValue, mask, codes, table, out, outFloat, logSumExp);
      calls.add(new Call(d, query, keyValue, mask, codes, table, out, outFloat, logSumExp));
    }

    @Override
    public void backward(
        Dimensions d,
        NDArray query,
        NDArray keyValue,
        NDArray mask,
        NDArray codes,
        NDArray table,
        NDArray outFloat,
        NDArray logSumExp,
        NDArray gradOut,
        NDArray gradQuery,
        NDArray gradKeyValue,
        NDArray gradTable) {
      throw new UnsupportedOperationException("the capture records forward passes only");
    }

    @Override
    public void forward(NormShape d, NDArray input, NDArray gain, NDArray out, NDArray inverseRms) {
      kernels.forward(d, input, gain, out, inverseRms);
    }

    @Override
    public void backward(
        NormShape d,
        NDArray input,
        NDArray gain,
        NDArray inverseRms,
        NDArray gradOut,
        NDArray gradInput,
        NDArray gradGain) {
      throw new UnsupportedOperationException("the capture records forward passes only");
    }
  }

  @Test(timeOut = 1_800_000)
  public void measureTrainingAttention() throws Exception {
    String directory = System.getProperty("prism.bench.rows");
    if (directory == null || Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("needs -Dprism.bench.rows and a GPU");
    }
    Device gpu = Device.gpu(0);
    FitPlan plan =
        new FitPlan(
            new FitSettings(1, STEP_ROWS, 2e-4, 1e-5, 1f, 0.01, 0.02),
            PrismShape.standard(),
            AuxiliaryLoss.standard(),
            FitPlan.TOKEN_BUDGET,
            3);
    try (CohortStore store = new CohortStore();
        Model model = Fit.blank(plan.shape(), gpu, plan.seed());
        PtStream stream = ((PtEngine) model.getNDManager().getEngine()).newStream(gpu);
        PtStreamScope ignored = stream.openScope();
        PtEvent event = stream.newEvent();
        StepFeed feed = new StepFeed(store, plan, LANES, THREADS)) {
      store.addCohort(Path.of(directory));
      int cohort = store.cohorts().get(0);
      int[] rows = Fit.shuffle(store.select(cohort, cohort, Selection.LABELS), 5);
      HipAttention kernels = HipAttention.on(HipLibrary.load(), stream);
      double[][] totals = new double[LANES][2];
      double slowest = 0;
      for (int step = 0; step < STEPS; step++) {
        StepFeed.Step prepared =
            feed.prepare(Arrays.copyOfRange(rows, step * STEP_ROWS, (step + 1) * STEP_ROWS), Teacher.RAW)
                .join();
        double[] lanes = new double[LANES];
        for (int lane = 0; lane < LANES; lane++) {
          double[] sums = new double[2];
          for (Microbatch part : prepared.lanes().get(lane)) {
            for (double[] layer : measure(model, kernels, event, part, step, lane)) {
              sums[0] += layer[0];
              sums[1] += layer[1];
            }
          }
          log.info(
              "PRISM-T1 step={} lane={} pieces={} forward={} backward={} total={} ms",
              step,
              lane,
              prepared.lanes().get(lane).size(),
              String.format("%.2f", sums[0]),
              String.format("%.2f", sums[1]),
              String.format("%.2f", sums[0] + sums[1]));
          totals[lane][0] += sums[0];
          totals[lane][1] += sums[1];
          lanes[lane] = sums[0] + sums[1];
        }
        slowest += Arrays.stream(lanes).max().orElseThrow();
        feed.release(prepared);
      }
      StringBuilder line = new StringBuilder("PRISM-T1 mean of " + STEPS + " steps (ms per step):");
      for (int lane = 0; lane < LANES; lane++) {
        line.append(
            String.format(
                " lane%d forward=%.2f backward=%.2f", lane, totals[lane][0] / STEPS, totals[lane][1] / STEPS));
      }
      line.append(String.format(" slower lane forward+backward=%.2f", slowest / STEPS));
      log.info(line.toString());
    }
  }

  /**
   * 微小バッチを網に通して背骨の注意の配列を取り、層ごとに順伝播と逆伝播を測る。
   *
   * @return 層ごとの 1 回あたりの ms {順伝播, 逆伝播}
   */
  private static List<double[]> measure(
      Model model, TrainingKernels kernels, PtEvent event, Microbatch part, int step, int lane) {
    NDManager manager = model.getNDManager();
    PrismNetwork network = (PrismNetwork) model.getBlock();
    List<double[]> layers = new ArrayList<>();
    try (NDManager scope = manager.newSubManager();
        InferenceMode inference = manager.getEngine().newInferenceMode()) {
      Capture capture = new Capture(kernels);
      network.forward(
          new ParameterStore(scope, false),
          part.inputs(scope, Uploads.on(scope)),
          Pass.inference(Precision.BFLOAT16, AttentionTape.forwardOnly(capture)));
      for (int layer = 0; layer < capture.calls.size(); layer++) {
        Call call = capture.calls.get(layer);
        RelationAttention.Dimensions d = call.dimensions();
        Shape queries = call.query().getShape();
        NDArray gradOut = scope.randomNormal(queries).toType(DataType.BFLOAT16, false);
        NDArray gradQuery = scope.create(queries, DataType.BFLOAT16);
        NDArray gradKeyValue = scope.create(call.keyValue().getShape(), DataType.BFLOAT16);
        NDArray gradTable =
            scope.create(new Shape((long) d.rows() * d.heads(), d.tableRows()), DataType.FLOAT32);
        double forward =
            millis(
                event,
                () ->
                    kernels.forward(
                        d,
                        call.query(),
                        call.keyValue(),
                        call.mask(),
                        call.codes(),
                        call.table(),
                        call.out(),
                        call.outFloat(),
                        call.logSumExp()));
        double backward =
            millis(
                event,
                () ->
                    kernels.backward(
                        d,
                        call.query(),
                        call.keyValue(),
                        call.mask(),
                        call.codes(),
                        call.table(),
                        call.outFloat(),
                        call.logSumExp(),
                        gradOut,
                        gradQuery,
                        gradKeyValue,
                        gradTable));
        log.info(
            "PRISM-T1 step={} lane={} rows={} S={} layer={} queries={} forward={} backward={} ms",
            step,
            lane,
            d.rows(),
            d.keys(),
            layer,
            d.queries(),
            String.format("%.3f", forward),
            String.format("%.3f", backward));
        layers.add(new double[] {forward, backward});
      }
    }
    return layers;
  }

  /** 演算を温めてから {@value #LAUNCHES} 回積み、ストリームの終わりを待つまでの 1 回あたりの ms。 */
  private static double millis(PtEvent event, Runnable launch) {
    for (int i = 0; i < WARMUP; i++) {
      launch.run();
    }
    event.record();
    event.synchronize();
    long started = System.nanoTime();
    for (int i = 0; i < LAUNCHES; i++) {
      launch.run();
    }
    event.record();
    event.synchronize();
    return (System.nanoTime() - started) / 1e6 / LAUNCHES;
  }
}

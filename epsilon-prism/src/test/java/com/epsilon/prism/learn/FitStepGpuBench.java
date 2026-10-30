package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.model.AttentionTape;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.LossInputs;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 学習の歩の時間の測り（通常の試験には入れず、サーバの GPU で {@code -Dtest=FitStepGpuBench -Dprism.bench.rows=<rows-v2/t>} で回す）。
 * 本物の行を本番の詰め方（{@link StepFeed}、トークンの上限 {@link FitPlan#TOKEN_BUDGET}）で微小バッチにする。測る範囲は
 * {@code -Dprism.bench.mode}（{@link Mode}。既定 all）。
 *
 * <ol>
 *   <li>{@link Fit#run}（4096 行の歩、深さ 2 と 3、GPU 1 枚と全部）の行/秒と、段ごとの秒。当てはめは本番と同じ（最後の塊を分け、背骨の注意は
 *       同梱した libprism_hip.so の演算）で、系の性質 {@value FitProfile#PROPERTY} があれば段の時計の表を当てはめの終わりに出す。
 *   <li>GPU 1 枚で、歩の行の数ごとの「順伝播だけ」と「順伝播・損失・逆伝播」（djl-rocm の注意と自前の HIP の注意）の ms と、詰めた
 *       トークンと本当のトークンの比。
 *   <li>{@code -Dprism.bench.trace=<file>} があれば、4096 行の 2 歩の演算ごとの記録（PyTorch の legacy profiler、GPU の時間つき）を書く。
 * </ol>
 */
public class FitStepGpuBench {
  private static final Logger log = LoggerFactory.getLogger(FitStepGpuBench.class);

  private static final int THREADS = 16;
  private static final int STEP_ROWS = 4096;
  private static final int FIT_STEPS = 40;
  private static final int WARMUP = 2;
  private static final int MEASURED = 5;

  /** 測る範囲。 */
  private enum Mode {
    /** 全部（上の 1〜3）。 */
    ALL,
    /** 本番の形（深さ 2・GPU 全部）の {@link Fit#run} だけ（2 GPU の分け方や当てはめの計測を比べる）。 */
    FIT,
    /** GPU 1 枚で 4096 行の歩を自前の注意で温めの後 5 歩だけ（rocprofv3 でカーネルの時間を取る）。 */
    QUICK;

    static Mode configured() {
      return valueOf(System.getProperty("prism.bench.mode", "all").strip().toUpperCase(Locale.ROOT));
    }
  }

  @Test(timeOut = 3_600_000)
  public void measureTrainingSteps() throws Exception {
    String directory = System.getProperty("prism.bench.rows");
    int gpus = Engine.getInstance().getGpuCount();
    if (directory == null || gpus == 0) {
      throw new SkipException("needs -Dprism.bench.rows and a GPU");
    }
    Mode mode = Mode.configured();
    try (CohortStore store = new CohortStore()) {
      store.addCohort(Path.of(directory));
      int cohort = store.cohorts().get(0);
      int[] rows = Fit.shuffle(store.select(cohort, cohort, Selection.LABELS), 5);
      log.info("PRISM-BENCH cohort={} rows={} mode={}", cohort, rows.length, mode);
      Device[] all = new Device[gpus];
      for (int gpu = 0; gpu < gpus; gpu++) {
        all[gpu] = Device.gpu(gpu);
      }
      if (mode == Mode.FIT) {
        fit(store, rows, PrismShape.standard(), all);
        return;
      }
      for (int depth = PrismShape.MIN_DEPTH; mode == Mode.ALL && depth <= PrismShape.MAX_DEPTH; depth++) {
        fit(store, rows, PrismShape.standard(depth), new Device[] {all[0]});
        if (gpus > 1) {
          fit(store, rows, PrismShape.standard(depth), all);
        }
      }
      boolean quick = mode == Mode.QUICK;
      FitPlan plan = plan(PrismShape.standard(), 1);
      try (Model model = Fit.blank(plan.shape(), all[0], plan.seed());
          PtStream stream = ((PtEngine) model.getNDManager().getEngine()).newStream(all[0]);
          PtStreamScope ignored = stream.openScope()) {
        for (Pair<String, Parameter> pair : model.getBlock().getParameters()) {
          pair.getValue().getArray().setRequiresGradient(true);
        }
        HipAttention attention = HipAttention.on(HipLibrary.load(), stream);
        int from = 0;
        for (int stepRows : quick ? new int[0] : new int[] {1024, 2048, 4096, 8192}) {
          from = steps(store, rows, from, plan, model, stepRows, false, null, null);
          from = steps(store, rows, from, plan, model, stepRows, true, null, null);
          from = steps(store, rows, from, plan, model, stepRows, true, attention, null);
        }
        if (quick) {
          steps(store, rows, from, plan, model, STEP_ROWS, true, attention, null);
        }
        String trace = System.getProperty("prism.bench.trace");
        if (trace != null) {
          steps(store, rows, from, plan, model, STEP_ROWS, true, attention, trace);
        }
      }
    }
  }

  private static FitPlan plan(PrismShape shape, int steps) {
    return new FitPlan(
        new FitSettings(steps, STEP_ROWS, 2e-4, 1e-5, 1f, 0.01, 0.02),
        shape,
        AuxiliaryLoss.standard(),
        FitPlan.TOKEN_BUDGET,
        3);
  }

  /** Fit.run の歩を測る（短い当てはめで温めてから）。 */
  private static void fit(CohortStore store, int[] rows, PrismShape shape, Device[] devices) {
    FitPlan warm = plan(shape, 3);
    FitPlan plan = plan(shape, FIT_STEPS);
    try (Model model = Fit.blank(shape, devices[0], plan.seed())) {
      Fit.run(store, rows, model, warm, Teacher.RAW, NetworkDevices.of(devices), THREADS);
      FitResult result = Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(devices), THREADS);
      log.info(
          "PRISM-BENCH fit depth={} devices={} ms/step={} rows/s={} cohort(3.6M rows)={}s"
              + " stages{feed,transfer,compute,optimizer}={} loss {} -> {}",
          shape.depth(),
          devices.length,
          String.format("%.1f", result.seconds() * 1e3 / result.steps()),
          String.format("%.0f", result.steps() * (double) STEP_ROWS / result.seconds()),
          String.format("%.0f", 3.6e6 * result.seconds() / (result.steps() * (double) STEP_ROWS)),
          Arrays.toString(result.stageSeconds()),
          String.format("%.4f", result.firstLoss()),
          String.format("%.4f", result.lastLoss()));
    }
  }

  /**
   * GPU 1 枚で歩を回して測る（同期は歩の終わりに値を 1 つ読む）。attention があれば背骨の注意を自前の HIP の演算にする。trace があれば
   * 温めの後の 2 歩を記録する。
   *
   * @return 次に使う行の位置
   */
  private static int steps(
      CohortStore store,
      int[] rows,
      int from,
      FitPlan plan,
      Model model,
      int stepRows,
      boolean backward,
      HipAttention attention,
      String trace) {
    PrismNetwork network = (PrismNetwork) model.getBlock();
    NDManager manager = model.getNDManager();
    Parameter probe = network.getParameters().valueAt(0);
    PrismLoss loss = new PrismLoss(plan.auxiliary());
    int measured = trace == null ? MEASURED : 2;
    long nanos = 0;
    long real = 0;
    long padded = 0;
    long parts = 0;
    try (StepFeed feed = new StepFeed(store, plan, Teacher.RAW, 1, THREADS)) {
      for (int step = 0; step < WARMUP + measured; step++) {
        if (from + stepRows > rows.length) {
          from = 0;
        }
        StepFeed.Step prepared = feed.prepare(Arrays.copyOfRange(rows, from, from + stepRows)).join();
        from += stepRows;
        boolean counted = step >= WARMUP;
        if (counted && step == WARMUP && trace != null) {
          JniUtils.startProfile(true, true, false);
        }
        long started = System.nanoTime();
        List<Microbatch> lane = prepared.lanes().get(0);
        for (Microbatch part : lane) {
          try (NDManager scope = manager.newSubManager()) {
            NetworkInputs inputs = part.inputs(scope);
            if (backward) {
              try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
                LossInputs targets = part.lossInputs(inputs, part.labels());
                AttentionTape tape = attention == null ? null : new AttentionTape(attention);
                Outputs outputs =
                    network.forward(
                        new ParameterStore(scope, false),
                        inputs,
                        Pass.training(Precision.BFLOAT16, part.measured(), part.auxiliary(), tape));
                NDArray value =
                    loss.total(outputs, targets, 1f, prepared.denominator(), part.share());
                if (tape == null) {
                  collector.backward(value);
                } else {
                  tape.backward(collector, value);
                }
              }
              try (NDArray gradient = probe.getArray().getGradient();
                  NDArray sum = gradient.sum()) {
                sum.getFloat();
              }
            } else {
              network
                  .forward(new ParameterStore(scope, false), inputs, Pass.inference(Precision.BFLOAT16))
                  .scores()
                  .toFloatArray();
            }
          }
          if (counted) {
            real += part.batch().plan().count(Axis.TOKEN);
            padded += part.tokens();
            parts++;
          }
        }
        if (counted) {
          nanos += System.nanoTime() - started;
        }
        feed.release(prepared);
      }
      if (trace != null) {
        JniUtils.stopProfile(trace);
      }
      double ms = nanos / 1e6 / measured;
      log.info(
          "PRISM-BENCH step rows={} {} ms/step={} rows/s={} microbatches/step={}"
              + " tokens/row real={} padded={} (padding x{}) feed={}s{}",
          stepRows,
          backward ? (attention == null ? "train-djl" : "train-hip") : "forward",
          String.format("%.1f", ms),
          String.format("%.0f", stepRows / (ms / 1e3)),
          String.format("%.1f", parts / (double) measured),
          String.format("%.1f", real / (double) measured / stepRows),
          String.format("%.1f", padded / (double) measured / stepRows),
          String.format("%.2f", padded / (double) real),
          String.format("%.2f", feed.seconds()),
          trace == null ? "" : " trace=" + trace);
    }
    return from;
  }
}

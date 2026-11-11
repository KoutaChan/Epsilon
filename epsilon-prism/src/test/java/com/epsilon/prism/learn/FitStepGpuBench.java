package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtMemoryStats;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.AttentionTape;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.model.StagedBatch;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 学習の 1 ステップの時間を測るベンチ（行のコホートは {@code -Dprism.bench.rows=<rows/t>} で渡す）。本物の行を、本番と同じ詰め方
 * （{@link StepFeed}、トークンの上限 {@link FitPlan#TOKEN_BUDGET}）でマイクロバッチにする。測る範囲は {@code -Dprism.bench.mode} で選ぶ
 * （{@link Mode}。既定は all）。
 *
 * <ol>
 *   <li>{@link Fit#run}（1 ステップ 4096 行、深さ 2 と 3、GPU 1 枚と全部）の行/秒と、段ごとの秒。学習の仕方は本番と同じ（最後のバッチを分け、
 *       バックボーンの Attention は同梱した libprism_hip.so の演算を使う）。システムプロパティ {@value FitProfile#PROPERTY} があれば、学習の
 *       終わりに段ごとのタイマーの表を出す。
 *   <li>GPU 1 枚で、1 ステップの行の数ごとに「順伝播だけ」と「順伝播・損失・逆伝播」（djl-rocm の Attention と自前の HIP の Attention）の ms
 *       と、詰めたトークンと本当のトークンの比を出す。
 *   <li>{@code -Dprism.bench.trace=<file>} があれば、4096 行の 2 ステップについて演算ごとの記録（PyTorch の legacy profiler、GPU の時間つき）を書く。
 * </ol>
 *
 * <p>VRAM のピークは、PyTorch のキャッシュアロケーターの記録から読む（測るステップの前にピークを今の値に戻し、測った後に読む）。確保した量と、
 * キャッシュが取った量を GB で出す。
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
    /** 本番の形（深さ 2・GPU 全部）の {@link Fit#run} だけ。 */
    FIT,
    /** GPU 1 枚・1 ステップ 4096 行・自前の Attention で、ウォームアップの後 5 ステップだけ回す。 */
    QUICK,
    /**
     * 形のアーム（{@link ShapeArm}。v2 は深さ 2）ごとに、GPU 1 枚・1 ステップ 4096 行で「順伝播・損失・逆伝播」（djl-rocm の Attention と
     * 自前の HIP の Attention）の ms と VRAM のピークを測る。続けて、GPU 1 枚の {@link Fit#run} の ms/ステップと VRAM のピークを測る。
     */
    ARMS;

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
      InputFormat format = InputFormat.of(store.variant());
      int[] rows = Fit.shuffle(store.select(cohort, cohort, Selection.LABELS), 5);
      log.info("PRISM-BENCH cohort={} variant={} rows={} mode={}", cohort, store.variant(), rows.length, mode);
      Device[] all = new Device[gpus];
      for (int gpu = 0; gpu < gpus; gpu++) {
        all[gpu] = Device.gpu(gpu);
      }
      if (mode == Mode.FIT) {
        fit(store, rows, format, PrismShape.standard(), all, "v2");
        return;
      }
      if (mode == Mode.ARMS) {
        for (ShapeArm arm : ShapeArm.values()) {
          arm(store, rows, format, arm, all[0]);
        }
        return;
      }
      for (int depth = PrismShape.MIN_DEPTH; mode == Mode.ALL && depth <= PrismShape.MAX_DEPTH; depth++) {
        fit(store, rows, format, PrismShape.standard(depth), new Device[] {all[0]}, "v2");
        if (gpus > 1) {
          fit(store, rows, format, PrismShape.standard(depth), all, "v2");
        }
      }
      boolean quick = mode == Mode.QUICK;
      FitPlan plan = plan(PrismShape.standard(), 1);
      try (Model model = Fit.blank(format, plan.shape(), all[0], plan.seed());
          PtStream stream = ((PtEngine) model.getNDManager().getEngine()).newStream(all[0]);
          PtStream copies = ((PtEngine) model.getNDManager().getEngine()).newStream(all[0]);
          PtStreamScope ignored = stream.openScope()) {
        for (Pair<String, Parameter> pair : model.getBlock().getParameters()) {
          pair.getValue().getArray().setRequiresGradient(true);
        }
        HipAttention attention = HipAttention.on(HipLibrary.load(), stream);
        int from = 0;
        for (int stepRows : quick ? new int[0] : new int[] {1024, 2048, 4096, 8192}) {
          from = steps(store, rows, from, plan, model, copies, stepRows, false, null, null, "v2");
          from = steps(store, rows, from, plan, model, copies, stepRows, true, null, null, "v2");
          from = steps(store, rows, from, plan, model, copies, stepRows, true, attention, null, "v2");
        }
        if (quick) {
          steps(store, rows, from, plan, model, copies, STEP_ROWS, true, attention, null, "v2");
        }
        String trace = System.getProperty("prism.bench.trace");
        if (trace != null) {
          steps(store, rows, from, plan, model, copies, STEP_ROWS, true, attention, trace, "v2");
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

  /** アームの形（v2 は深さ 2、v3 は深さ 3）を GPU 1 枚で測る。djl-rocm と自前の Attention で 4096 行のステップを回し、続けて Fit.run を回す。 */
  private static void arm(CohortStore store, int[] rows, InputFormat format, ShapeArm arm, Device device) {
    PrismShape shape = arm.shape(arm == ShapeArm.V2 ? PrismShape.DEFAULT_DEPTH : ShapeArm.V3_DEPTH);
    String name = arm.name().toLowerCase(Locale.ROOT);
    FitPlan plan = plan(shape, 1);
    try (Model model = Fit.blank(format, shape, device, plan.seed());
        PtStream stream = ((PtEngine) model.getNDManager().getEngine()).newStream(device);
        PtStream copies = ((PtEngine) model.getNDManager().getEngine()).newStream(device);
        PtStreamScope ignored = stream.openScope()) {
      for (Pair<String, Parameter> pair : model.getBlock().getParameters()) {
        pair.getValue().getArray().setRequiresGradient(true);
      }
      HipAttention attention = HipAttention.on(HipLibrary.load(), stream);
      int from = steps(store, rows, 0, plan, model, copies, STEP_ROWS, true, null, null, name);
      steps(store, rows, from, plan, model, copies, STEP_ROWS, true, attention, null, name);
    }
    fit(store, rows, format, shape, new Device[] {device}, name);
  }

  /** Fit.run のステップを測る（短い学習でウォームアップしてから）。VRAM のピークは、測る学習の間に先頭のデバイスで記録した値。 */
  private static void fit(
      CohortStore store, int[] rows, InputFormat format, PrismShape shape, Device[] devices, String name) {
    FitPlan warm = plan(shape, 3);
    FitPlan plan = plan(shape, FIT_STEPS);
    try (Model model = Fit.blank(format, shape, devices[0], plan.seed())) {
      Fit.run(store, rows, model, warm, Teacher.RAW, NetworkDevices.of(devices), THREADS);
      PtEngine engine = (PtEngine) model.getNDManager().getEngine();
      engine.resetPeakMemoryStats(devices[0]);
      FitResult result = Fit.run(store, rows, model, plan, Teacher.RAW, NetworkDevices.of(devices), THREADS);
      log.info(
          "PRISM-BENCH fit depth={} devices={} ms/step={} rows/s={} 1M rows={}s"
              + " stages{feed,transfer,compute,optimizer}={} loss {} -> {} {} shape={}",
          shape.depth(),
          devices.length,
          String.format("%.1f", result.seconds() * 1e3 / result.steps()),
          String.format("%.0f", result.steps() * (double) STEP_ROWS / result.seconds()),
          String.format("%.0f", 1e6 * result.seconds() / (result.steps() * (double) STEP_ROWS)),
          Arrays.toString(result.stageSeconds()),
          String.format("%.4f", result.firstLoss()),
          String.format("%.4f", result.lastLoss()),
          vram(engine.getMemoryStats(devices[0])),
          name);
    }
  }

  /**
   * GPU 1 枚でステップを回して測る（同期は、ステップの終わりに値を 1 つ読んで取る）。逆伝播のステップでは、前もって並べた入力をコピー用の
   * ストリームで非同期に送り、計算のストリームは順伝播の前にコピーの終わりを待つ（学習器と違い、次のマイクロバッチのコピーを先に積まない）。
   * {@code attention} があれば、バックボーンの Attention を自前の HIP の演算にする。{@code trace} があれば、ウォームアップの後の 2 ステップを
   * 記録する。VRAM のピークは、ウォームアップの後のステップのピーク。
   *
   * @return 次に使う行の位置
   */
  private static int steps(
      CohortStore store,
      int[] rows,
      int from,
      FitPlan plan,
      Model model,
      PtStream copies,
      int stepRows,
      boolean backward,
      HipAttention attention,
      String trace,
      String name) {
    PrismNetwork network = (PrismNetwork) model.getBlock();
    NDManager manager = model.getNDManager();
    PtEngine engine = (PtEngine) manager.getEngine();
    Device device = manager.getDevice();
    Parameter probe = network.getParameters().valueAt(0);
    PrismLoss loss = new PrismLoss(plan.auxiliary());
    int measured = trace == null ? MEASURED : 2;
    long nanos = 0;
    long real = 0;
    long padded = 0;
    long parts = 0;
    try (StepFeed feed = new StepFeed(store, plan, 1, THREADS)) {
      for (int step = 0; step < WARMUP + measured; step++) {
        if (from + stepRows > rows.length) {
          from = 0;
        }
        StepFeed.Step prepared = feed.prepare(Arrays.copyOfRange(rows, from, from + stepRows), Teacher.RAW).join();
        from += stepRows;
        boolean counted = step >= WARMUP;
        if (counted && step == WARMUP) {
          engine.resetPeakMemoryStats(device);
          if (trace != null) {
            JniUtils.startProfile(true, true, false);
          }
        }
        long started = System.nanoTime();
        List<Microbatch> lane = prepared.lanes().get(0);
        for (Microbatch part : lane) {
          try (NDManager scope = manager.newSubManager()) {
            if (backward) {
              try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
                StagedBatch.Sent sent = part.send(scope, copies).receive();
                AttentionTape tape = attention == null ? null : new AttentionTape(attention);
                Outputs outputs =
                    network.forward(
                        new ParameterStore(scope, false),
                        sent.inputs(),
                        Pass.training(Precision.BFLOAT16, part.measured(), part.auxiliary(), tape));
                NDArray value =
                    loss.total(outputs, sent.loss(), 1f, prepared.denominator(), part.share());
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
                  .forward(
                      new ParameterStore(scope, false),
                      part.inputs(scope, PinnedUploads.on(scope)),
                      Pass.inference(Precision.BFLOAT16))
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
              + " tokens/row real={} padded={} (padding x{}) feed={}s {} shape={}{}",
          stepRows,
          backward ? (attention == null ? "train-djl" : "train-hip") : "forward",
          String.format("%.1f", ms),
          String.format("%.0f", stepRows / (ms / 1e3)),
          String.format("%.1f", parts / (double) measured),
          String.format("%.1f", real / (double) measured / stepRows),
          String.format("%.1f", padded / (double) measured / stepRows),
          String.format("%.2f", padded / (double) real),
          String.format("%.2f", feed.seconds()),
          vram(engine.getMemoryStats(device)),
          name,
          trace == null ? "" : " trace=" + trace);
    }
    return from;
  }

  /** VRAM のピーク（確保した量と、キャッシュが取った量。GB）。 */
  private static String vram(PtMemoryStats stats) {
    return String.format(
        "vram peak allocated=%.2fGB reserved=%.2fGB",
        stats.getPeakAllocatedBytes() / 1e9, stats.getPeakReservedBytes() / 1e9);
  }
}

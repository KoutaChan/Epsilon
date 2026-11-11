package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.PrismNetwork;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 冷えた GPU で、三人打ちの HIP の推論の速さを測る（テストのグループは rocm、GPU 1 枚）。三人打ちの半荘を打って集めた判断 {@value #ROWS} 行のバッチを、
 * v2 の形と V3 の形で {@value #WARMUP} + {@value #ITERATIONS} 回実行し、カーネルごとの時間の中央値と経過時間の中央値を 1 判断あたりの値で
 * ログに書く。カーネルの合計は 1 判断あたり {@value #LINE_MICROS} µs 以下であること。名前が *Test でないので既定のテストには入らず、名前を
 * 指定したときだけ実行される。
 */
public class ThreePlayerHipBench {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerHipBench.class);

  private static final int ROWS = 2048;
  private static final int WARMUP = 5;
  private static final int ITERATIONS = 20;
  private static final long VERSION = 5;

  /** 1 判断あたりのカーネルの時間のしきい値（µs）。 */
  private static final double LINE_MICROS = 3.8;

  @Test(groups = "rocm")
  public void coldSpeed() {
    HostBatch batch = ThreePlayerRows.encode(ThreePlayerRows.simulatedRows(ROWS, 800));
    List<String> failures = new ArrayList<>();
    for (ThreePlayerRows.Arm arm : ThreePlayerRows.Arm.values()) {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = GpuCase.newStream()) {
        PrismNetwork network = ThreePlayerRows.network(manager, arm.shape(), 70 + arm.ordinal());
        WeightLayout layout = WeightLayout.of(network);
        WeightPack pack = WeightPack.build(layout, network, VERSION);
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(ROWS))) {
          backend.publish(pack);
          double micros = measure(backend, batch, arm.label());
          if (!(micros <= LINE_MICROS)) {
            failures.add(arm.label() + " kernels take " + micros + " us per decision");
          }
        }
      }
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** バッチを {@value #WARMUP} + {@value #ITERATIONS} 回実行してログに 1 行書き、カーネルの中央値の合計を 1 判断あたり（µs）で返す。 */
  private static double measure(HipBackend backend, HostBatch input, String arm) {
    HipBatch batch = HipBackend.batchOf(input);
    int decisions = batch.rows();
    ByteBuffer scores = ByteBuffer.allocateDirect(batch.actions() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    backend.plan().profile(true);
    float[][] millis = new float[ITERATIONS][];
    float[] totals = new float[ITERATIONS];
    for (int i = 0; i < WARMUP + ITERATIONS; i++) {
      long start = System.nanoTime();
      backend.plan().forward(input.blob().duplicate().position(0), batch, scores, VERSION);
      backend.synchronize();
      long wall = System.nanoTime() - start;
      float[] kernels = backend.plan().kernelMillis();
      if (i >= WARMUP) {
        millis[i - WARMUP] = kernels;
        totals[i - WARMUP] = wall / 1e6f;
      }
    }
    StringBuilder line = new StringBuilder();
    double sum = 0;
    for (HipKernel kernel : HipKernel.values()) {
      float[] samples = new float[ITERATIONS];
      for (int i = 0; i < ITERATIONS; i++) {
        samples[i] = millis[i][kernel.ordinal()];
      }
      Arrays.sort(samples);
      double median = samples[ITERATIONS / 2];
      if (Double.isNaN(median)) {
        continue;  // この形では使わないカーネル
      }
      sum += median;
      line.append(String.format(Locale.ROOT, " %s=%.3fms(%.3fus)", kernel, median, median * 1e3 / decisions));
    }
    Arrays.sort(totals);
    int fixed = backend.layout().architecture().counts().fixed();
    double micros = sum * 1e3 / decisions;
    log.info(
        "PRISM-3P-HIP-SPEED arm {} rows {} tokens/row {} transitions/row {} actions/row {}:{} kernels={}ms"
            + " ({} us/decision) wall={}ms ({} us/decision)",
        arm,
        decisions,
        String.format(Locale.ROOT, "%.1f", ((double) fixed * decisions + batch.events()) / decisions),
        String.format(Locale.ROOT, "%.1f", (double) batch.transitions() / decisions),
        String.format(Locale.ROOT, "%.1f", (double) batch.actions() / decisions),
        line,
        String.format(Locale.ROOT, "%.3f", sum),
        String.format(Locale.ROOT, "%.3f", micros),
        String.format(Locale.ROOT, "%.3f", totals[ITERATIONS / 2]),
        String.format(Locale.ROOT, "%.3f", totals[ITERATIONS / 2] * 1e3 / decisions));
    return micros;
  }
}

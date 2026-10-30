package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 通しの順伝播（rocm の組。GPU 1 枚）:
 *
 * <ul>
 *   <li>HIP の s と eager の s（CPU の BF16 と FP32）の差の中央値・p99・最大・役割ごとの符号つきの平均、argmax の一致率
 *   <li>公開の出来事が 90 を超える行（T > 128）を含む塊でも同じ（記録の行の頭は「PRISM-HIP-T5 long」）
 *   <li>塊を組み直しても（行を混ぜ、前に別の行を足しても）同じ行の s がビットで同じ
 *   <li>白紙の網で s が全て +0.0f、版が違えば例外
 *   <li>1 判断あたりの GPU の時間（カーネルごと、本番の形の 2048 行の塊）
 * </ul>
 */
@Test(groups = "rocm")
public class HipForwardTest {
  private static final Logger log = LoggerFactory.getLogger(HipForwardTest.class);

  private static final int BATCHES = 8;
  private static final int SPEED_ROWS = 2048;
  private static final int WARMUP = 5;
  private static final int ITERATIONS = 20;

  /** s と argmax の一致（深さ 2 と 3、24 行 × 8 塊）。eager は CPU。 */
  public void scoresMatchEager() {
    for (int depth = 2; depth <= 3; depth++) {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = newStream()) {
        PrismNetwork network = HipFixtures.network(manager, depth, 50 + depth);
        ParameterStore store = new ParameterStore(manager, false);
        WeightLayout layout = WeightLayout.of(network);
        WeightPack pack = WeightPack.build(layout, network, 3);
        List<Float> hip = new ArrayList<>();
        List<Float> bf16 = new ArrayList<>();
        List<Float> fp32 = new ArrayList<>();
        List<Float> reference = new ArrayList<>();
        List<HipParity.Actions> actions = new ArrayList<>();
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(GpuCase.ROWS))) {
          backend.publish(pack);
          for (int b = 0; b < BATCHES; b++) {
            HostBatch batch = HipFixtures.encode(HipFixtures.mixedRows(GpuCase.ROWS, 600 + 100L * depth + 13L * b));
            float[] s = backend.forward(batch, 3);
            EagerStages.Stages half;
            EagerStages.Stages full;
            try (NDManager scope = manager.newSubManager()) {
              half = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
            }
            try (NDManager scope = manager.newSubManager()) {
              full = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.FLOAT32);
            }
            float[] cpu = new HipReference(pack).forward(batch);
            add(hip, s);
            add(bf16, half.scores());
            add(fp32, full.scores());
            add(reference, cpu);
            actions.add(HipParity.Actions.of(batch));
          }
        }
        HipParity.Actions joined = join(actions);
        HipParity.Report half = HipParity.compare(floats(hip), floats(bf16), joined);
        HipParity.Report full = HipParity.compare(floats(hip), floats(fp32), joined);
        HipParity.Report cpu = HipParity.compare(floats(hip), floats(reference), joined);
        log.info("PRISM-HIP-T5 depth {} HIP vs eager BF16 (CPU): {} passes={}", depth, half.summary(), half.passes());
        log.info("PRISM-HIP-T5 depth {} HIP vs eager FP32 (CPU): {} passes={}", depth, full.summary(), full.passes());
        log.info("PRISM-HIP-T5 depth {} HIP vs CPU reference: {}", depth, cpu.summary());
        Diff diff = Diff.fp32(floats(bf16), floats(hip));
        log.info("PRISM-HIP-T5 depth {} {}", depth, diff.line("s vs eager BF16"));
        Assert.assertTrue(diff.relativeRms() <= 2e-2, "s against eager BF16");
      }
    }
  }

  /**
   * 普通の行 2 つと公開の出来事が 90 を超える行 2 つの塊の s と eager の BF16 の s（深さ 2 と 3）。行のトークンが 128 を超えるので、K2 は
   * 波の 2 つ目の問い合わせの塊の q を LDS に置いて読む道を通る。
   */
  public void longRowsMatchEager() {
    List<long[]> rows = HipFixtures.longMixedRows();
    for (int depth = 2; depth <= 3; depth++) {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = newStream()) {
        PrismNetwork network = HipFixtures.network(manager, depth, 90 + depth);
        ParameterStore store = new ParameterStore(manager, false);
        WeightLayout layout = WeightLayout.of(network);
        HostBatch batch = HipFixtures.encode(rows);
        float[] s;
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(rows.size()))) {
          backend.publish(WeightPack.build(layout, network, 6));
          s = backend.forward(batch, 6);
        }
        EagerStages.Stages half;
        try (NDManager scope = manager.newSubManager()) {
          half = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
        }
        HipParity.Report report = HipParity.compare(s, half.scores(), HipParity.Actions.of(batch));
        Diff diff = Diff.fp32(half.scores(), s);
        log.info(
            "PRISM-HIP-T5 long depth {} HIP vs eager BF16 (CPU): {} passes={}", depth, report.summary(), report.passes());
        log.info("PRISM-HIP-T5 long depth {} {}", depth, diff.line("s vs eager BF16"));
        Assert.assertTrue(diff.relativeRms() <= 2e-2, "long rows: s against eager BF16");
      }
    }
  }

  /** 行を混ぜ、前に別の行を足して塊を組み直しても、同じ行の s がビットで同じ。 */
  public void rebatchingKeepsRowScoresBitwise() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream = newStream()) {
      PrismNetwork network = HipFixtures.network(manager, 2, 61);
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, 4);
      List<long[]> rows = HipReferenceTest.mixedRows(GpuCase.ROWS);
      List<long[]> other = rows(GpuCase.ROWS, 700);
      List<long[]> shuffled = new ArrayList<>(rows);
      Collections.shuffle(shuffled, new Random(9));
      List<long[]> prefixed = new ArrayList<>(other.subList(0, 7));
      prefixed.addAll(shuffled);
      try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(4 * GpuCase.ROWS))) {
        backend.publish(pack);
        HostBatch first = HipFixtures.encode(rows);
        HostBatch second = HipFixtures.encode(prefixed);
        float[] a = backend.forward(first, 4);
        float[] b = backend.forward(second, 4);
        int compared = 0;
        for (int r = 0; r < rows.size(); r++) {
          int s = prefixed.indexOf(rows.get(r));
          for (int k = 0; k < HipFixtures.actions(first, r); k++) {
            Assert.assertEquals(
                Float.floatToRawIntBits(b[HipFixtures.firstAction(second, s) + k]),
                Float.floatToRawIntBits(a[HipFixtures.firstAction(first, r) + k]),
                "row " + r + " action " + k);
            compared++;
          }
        }
        log.info("PRISM-HIP-T5 rebatch: {} scores bitwise equal", compared);
      }
    }
  }

  /** 白紙の網で s が全て +0.0f。古い版の forward は例外。 */
  public void blankScoresZeroAndStaleVersionsFail() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream = newStream()) {
      manager.getEngine().setRandomSeed(7);
      PrismNetwork network = new PrismNetwork(PrismShape.standard());
      network.initialize(manager);
      WeightLayout layout = WeightLayout.of(network);
      try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(GpuCase.ROWS))) {
        backend.publish(WeightPack.build(layout, network, 0));
        HostBatch batch = HipFixtures.encode(HipReferenceTest.mixedRows(GpuCase.ROWS));
        float[] s = backend.forward(batch, 0);
        for (int a = 0; a < s.length; a++) {
          Assert.assertEquals(Float.floatToRawIntBits(s[a]), 0, "blank score " + a + " = " + s[a]);
        }
        Assert.expectThrows(IllegalStateException.class, () -> backend.forward(batch, 1));
        log.info("PRISM-HIP-T5 blank: {} scores are +0.0f; stale version rejected", s.length);
      }
    }
  }

  /** 1 判断あたりの GPU の時間（本番の形 H256・F512・G256、2048 行の塊、深さ 2 と 3）。 */
  public void measureSpeed() {
    List<long[]> rows = rows(SPEED_ROWS, 800);
    HostBatch batch = HipFixtures.encode(rows);
    int decisions = batch.plan().rows();
    for (int depth = 2; depth <= 3; depth++) {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = newStream()) {
        PrismNetwork network = HipFixtures.network(manager, depth, 70 + depth);
        WeightLayout layout = WeightLayout.of(network);
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(SPEED_ROWS))) {
          backend.publish(WeightPack.build(layout, network, 5));
          backend.plan().profile(true);
          float[][] millis = new float[ITERATIONS][];
          float[] totals = new float[ITERATIONS];
          for (int i = 0; i < WARMUP + ITERATIONS; i++) {
            long start = System.nanoTime();
            backend.forward(batch, 5);
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
            sum += median;
            line.append(String.format(Locale.ROOT, " %s=%.3fms(%.3fus)", kernel, median, median * 1e3 / decisions));
          }
          Arrays.sort(totals);
          log.info(
              "PRISM-HIP-SPEED depth {} rows {} tokens/row {} transitions/row {} actions/row {}:{} kernels={}ms"
                  + " ({} us/decision) wall={}ms ({} us/decision)",
              depth,
              decisions,
              String.format(Locale.ROOT, "%.1f", (39.0 * decisions + batch.plan().count(Axis.EVENT)) / decisions),
              String.format(Locale.ROOT, "%.1f", (double) batch.plan().count(Axis.TRANSITION) / decisions),
              String.format(Locale.ROOT, "%.1f", (double) batch.plan().count(Axis.ACTION) / decisions),
              line,
              String.format(Locale.ROOT, "%.3f", sum),
              String.format(Locale.ROOT, "%.3f", sum * 1e3 / decisions),
              String.format(Locale.ROOT, "%.3f", totals[ITERATIONS / 2]),
              String.format(Locale.ROOT, "%.3f", totals[ITERATIONS / 2] * 1e3 / decisions));
        }
      }
    }
  }

  /** GREEDY と CALLS の半荘から交互に集めた行。 */
  private static List<long[]> rows(int count, long seed) {
    List<long[]> greedy = HipFixtures.observations(seed, 1 + count / 600, DecisionWalk.Policy.GREEDY, count);
    List<long[]> calls = HipFixtures.observations(seed + 1, 1 + count / 600, DecisionWalk.Policy.CALLS, count);
    List<long[]> rows = new ArrayList<>();
    for (int i = 0; rows.size() < count && (i < greedy.size() || i < calls.size()); i++) {
      if (i < greedy.size()) {
        rows.add(greedy.get(i));
      }
      if (i < calls.size() && rows.size() < count) {
        rows.add(calls.get(i));
      }
    }
    return rows;
  }

  private static PtStream newStream() {
    return ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
  }

  private static void add(List<Float> target, float[] values) {
    for (float value : values) {
      target.add(value);
    }
  }

  private static float[] floats(List<Float> values) {
    float[] result = new float[values.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  /** 塊ごとの手の形を、s を続けた並びの 1 つの手の形にする。 */
  private static HipParity.Actions join(List<HipParity.Actions> parts) {
    List<Integer> first = new ArrayList<>();
    List<Integer> count = new ArrayList<>();
    List<ActionRole> roles = new ArrayList<>();
    int offset = 0;
    for (HipParity.Actions part : parts) {
      for (int r = 0; r < part.rowFirst().length; r++) {
        first.add(offset + part.rowFirst()[r]);
        count.add(part.rowCount()[r]);
      }
      roles.addAll(Arrays.asList(part.roles()));
      offset += part.roles().length;
    }
    return new HipParity.Actions(
        first.stream().mapToInt(Integer::intValue).toArray(),
        count.stream().mapToInt(Integer::intValue).toArray(),
        roles.toArray(new ActionRole[0]));
  }
}

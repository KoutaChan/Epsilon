package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipPlan;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputFormat.Role;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism4p.input.InputSchema;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
 * 通しの順伝播のテスト（rocm のグループ。GPU 1 枚）。形は全てのアームで試す（{@link HipFixtures.Arm}、システムプロパティで絞れる）。
 *
 * <ul>
 *   <li>HIP の s を、eager の s（CPU の BF16 と FP32）と CPU の参照の s と比べる。差の中央値・p99・最大、手の種類ごとの符号つきの平均、
 *       argmax の一致率を出す。しきい値は相対 RMS で、eager の BF16 とは {@value #EAGER_LIMIT}、eager の FP32 とは {@value #FP32_LIMIT}、
 *       参照とは {@value #REFERENCE_LIMIT}。eager 自身の BF16 の丸めでは順位が変わらない行では、argmax の一致が
 *       {@value HipParity#ARGMAX_LIMIT} 以上（{@link #argmax}）
 *   <li>公開イベントが 90 を超える行（T > 128）のバッチでも同じことを確かめる（ログの行の先頭は「PRISM-HIP-T5 long」）
 *   <li>同じバッチを 2 回流すと s がビット単位で同じ。行を混ぜ、前に別の行を足してバッチを組み直しても、同じ行の s がビット単位で同じ
 *   <li>初期化したままのネットワークでは s が全て +0.0f。バージョンが違えば例外
 *   <li>1 判断あたりの GPU の時間（カーネルごと、本番の形の 2048 行のバッチ）
 * </ul>
 */
@Test(groups = "rocm")
public class HipForwardTest {
  private static final Logger log = LoggerFactory.getLogger(HipForwardTest.class);

  private static final int BATCHES = 8;
  private static final int SPEED_ROWS = 2048;
  private static final int WARMUP = 5;
  private static final int ITERATIONS = 20;
  private static final double EAGER_LIMIT = 4e-2;
  private static final double FP32_LIMIT = 1e-1;
  private static final double REFERENCE_LIMIT = 2e-2;

  /** 長い行のバッチに入れる長い行の数。 */
  private static final int LONG_ROWS = 48;

  /** s と argmax の一致を確かめる（全てのアーム、24 行 × 8 バッチ）。eager は CPU で動かす。 */
  public void scoresMatchEager() {
    List<String> failures = new ArrayList<>();
    eachArm(failures, HipFixtures.Arm.selected(), arm -> {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = GpuCase.newStream()) {
        PrismNetwork network = HipFixtures.network(manager, arm.shape(), 50 + arm.ordinal());
        ParameterStore store = new ParameterStore(manager, false);
        WeightLayout layout = WeightLayout.of(network);
        WeightPack pack = WeightPack.build(layout, network, 3);
        Scores scores = new Scores();
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(GpuCase.ROWS))) {
          backend.publish(pack);
          for (int b = 0; b < BATCHES; b++) {
            HostBatch batch =
                HipFixtures.encode(HipFixtures.mixedRows(GpuCase.ROWS, 600 + 100L * arm.ordinal() + 13L * b));
            float[] s = backend.forward(batch, 3);
            EagerStages.Stages eager = scores.add(s, network, store, manager, pack, batch);
            if (b == 0) {
              trace(backend.plan(), eager, arm.label() + " vs eager BF16", layout.architecture().counts().fixed()
                  * batch.plan().rows(), batch.plan().count(Axis.TRANSITION));
            }
          }
          log.info("PRISM-HIP-T5 arm {} largest score differences vs eager BF16:{}", arm.label(),
              largest(scores.hip(), scores.bf16(), join(scores.actions), 8));
        }
        scores.check(failures, arm.label(), "");
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /**
   * 普通の行 2 つと、公開イベントが 90 を超える行 {@value #LONG_ROWS} 個からなるバッチの s を確かめる（全てのアーム）。行のトークンが 128 を
   * 超えるので、K2・K4 は 10 ウェーブ構成のカーネルを通る（普通の行はほかの構成のカーネル）。argmax は長い行だけでも数える。
   */
  public void longRowsMatchEager() {
    List<long[]> rows = new ArrayList<>(HipFixtures.mixedRows(2, 900));
    rows.addAll(HipFixtures.longRows(LONG_ROWS));
    List<String> failures = new ArrayList<>();
    eachArm(failures, HipFixtures.Arm.selected(), arm -> {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = GpuCase.newStream()) {
        PrismNetwork network = HipFixtures.network(manager, arm.shape(), 90 + arm.ordinal());
        ParameterStore store = new ParameterStore(manager, false);
        WeightLayout layout = WeightLayout.of(network);
        WeightPack pack = WeightPack.build(layout, network, 6);
        HostBatch batch = HipFixtures.encode(rows);
        Scores scores = new Scores();
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(rows.size()))) {
          backend.publish(pack);
          scores.add(backend.forward(batch, 6), network, store, manager, pack, batch);
        }
        scores.check(failures, arm.label(), "long ");
        HipParity.Actions longOnly = subset(HipParity.Actions.of(batch), 2, rows.size());
        HipParity.Report report = HipParity.compare(scores.hip(), scores.bf16(), longOnly);
        log.info("PRISM-HIP-T5 long-only arm {} HIP vs eager BF16 (CPU): {} flips: {}", arm.label(), report.summary(),
            flips(scores.hip(), scores.bf16(), longOnly));
        argmax(failures, "long-only " + arm.label(), scores.hip(), scores.bf16(), scores.fp32(), longOnly);
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 同じバッチを 2 回流すと s がビット単位で同じ。行を混ぜ、前に別の行を足してバッチを組み直しても、同じ行の s がビット単位で同じ（全てのアーム）。 */
  public void rebatchingKeepsRowScoresBitwise() {
    List<long[]> rows = HipReferenceTest.mixedRows(GpuCase.ROWS);
    List<long[]> other = HipFixtures.simulatedRows(GpuCase.ROWS, 700);
    List<long[]> shuffled = new ArrayList<>(rows);
    Collections.shuffle(shuffled, new Random(9));
    List<long[]> prefixed = new ArrayList<>(other.subList(0, 7));
    prefixed.addAll(shuffled);
    List<String> failures = new ArrayList<>();
    eachArm(failures, HipFixtures.Arm.selected(), arm -> {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = GpuCase.newStream()) {
        PrismNetwork network = HipFixtures.network(manager, arm.shape(), 61 + arm.ordinal());
        WeightLayout layout = WeightLayout.of(network);
        WeightPack pack = WeightPack.build(layout, network, 4);
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(4 * GpuCase.ROWS))) {
          backend.publish(pack);
          HostBatch first = HipFixtures.encode(rows);
          HostBatch second = HipFixtures.encode(prefixed);
          float[] a = backend.forward(first, 4);
          float[] b = backend.forward(second, 4);
          float[] again = backend.forward(first, 4);
          Assert.assertTrue(HipParity.bitwiseEqual(a, again), arm.label() + ": two runs of one batch differ");
          int compared = 0;
          for (int r = 0; r < rows.size(); r++) {
            int s = prefixed.indexOf(rows.get(r));
            for (int k = 0; k < HipFixtures.actions(first, r); k++) {
              Assert.assertEquals(
                  Float.floatToRawIntBits(b[HipFixtures.firstAction(second, s) + k]),
                  Float.floatToRawIntBits(a[HipFixtures.firstAction(first, r) + k]),
                  arm.label() + " row " + r + " action " + k);
              compared++;
            }
          }
          log.info("PRISM-HIP-T5 rebatch arm {}: {} scores bitwise equal, two runs bitwise equal", arm.label(), compared);
        }
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 初期化したままのネットワークでは s が全て +0.0f。古いバージョンで forward すると例外になる（全てのアーム）。 */
  public void blankScoresZeroAndStaleVersionsFail() {
    List<String> failures = new ArrayList<>();
    eachArm(failures, HipFixtures.Arm.selected(), arm -> {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = GpuCase.newStream()) {
        manager.getEngine().setRandomSeed(7);
        PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, arm.shape());
        network.initialize(manager);
        WeightLayout layout = WeightLayout.of(network);
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(GpuCase.ROWS))) {
          backend.publish(WeightPack.build(layout, network, 0));
          HostBatch batch = HipFixtures.encode(HipReferenceTest.mixedRows(GpuCase.ROWS));
          float[] s = backend.forward(batch, 0);
          for (int a = 0; a < s.length; a++) {
            Assert.assertEquals(Float.floatToRawIntBits(s[a]), 0, arm.label() + " blank score " + a + " = " + s[a]);
          }
          Assert.expectThrows(IllegalStateException.class, () -> backend.forward(batch, 1));
          log.info("PRISM-HIP-T5 blank arm {}: {} scores are +0.0f; stale version rejected", arm.label(), s.length);
        }
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 1 判断あたりの GPU の時間を測る（全てのアーム）。バッチは模擬の半荘の 2048 行で、アームごとに 1 行をログに書く。 */
  public void measureSpeed() {
    List<String> failures = new ArrayList<>();
    HostBatch batch = HipFixtures.encode(HipFixtures.simulatedRows(SPEED_ROWS, 800));
    eachArm(failures, HipFixtures.Arm.selected(), arm -> {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu());
          PtStream stream = GpuCase.newStream()) {
        PrismNetwork network = HipFixtures.network(manager, arm.shape(), 70 + arm.ordinal());
        WeightLayout layout = WeightLayout.of(network);
        WeightPack pack = WeightPack.build(layout, network, 5);
        int rows = batch.plan().rows();
        try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(rows))) {
          backend.publish(pack);
          measure(backend, batch.blob(), HipBackend.batchOf(batch), pack.version(), arm.label());
        }
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }
  /**
   * 通しの順伝播の後の作業領域（y、Cross-Attention のキーとバリュー、z、B）を各段の値と比べてログに書く。通しの s の差が、どの段で
   * 大きくなるかを見るため。
   *
   * @param plan 順伝播を実行した計画
   * @param expected 同じバッチの各段の値
   * @param label ログの行の名前
   * @param fixedTokens 局・席・牌のトークンの数 F·R
   * @param transitions 遷移の数
   */
  private static void trace(HipPlan plan, EagerStages.Stages expected, String label, int fixedTokens, int transitions) {
    int hidden = expected.states().length / fixedTokens;
    log.info("PRISM-HIP-T5 trace {}: {}; {}; {}; {}", label,
        Diff.bf16(expected.states(), read(plan, HipRegion.STATE, expected.states().length, true)).line("y"),
        Diff.bf16(expected.memory(), read(plan, HipRegion.MEMORY, expected.memory().length, true)).line("KV"),
        Diff.bf16(expected.transitions(), read(plan, HipRegion.TRANSITION, transitions * hidden, true)).line("z"),
        Diff.fp32(expected.values(), read(plan, HipRegion.VALUE, transitions, false)).line("B"));
  }

  private static float[] read(HipPlan plan, HipRegion region, int count, boolean bf16) {
    ByteBuffer buffer =
        ByteBuffer.allocateDirect(count * (bf16 ? Short.BYTES : Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
    plan.copyOut(region, buffer, buffer.capacity());
    plan.synchronize();
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] =
          bf16 ? Float.intBitsToFloat((buffer.getShort(i * Short.BYTES) & 0xFFFF) << 16) : buffer.getFloat(i * Float.BYTES);
    }
    return values;
  }

  /** 差の大きい手を k 個選び、行・手の種類・参照の s・HIP の s を並べて返す。 */
  private static String largest(float[] hip, float[] reference, HipParity.Actions actions, int k) {
    Integer[] order = new Integer[actions.roles().length];
    for (int a = 0; a < order.length; a++) {
      order[a] = a;
    }
    Arrays.sort(order, (x, y) -> Double.compare(Math.abs((double) hip[y] - reference[y]), Math.abs((double) hip[x] - reference[x])));
    int[] rowOf = new int[order.length];
    for (int r = 0; r < actions.rowFirst().length; r++) {
      for (int a = actions.rowFirst()[r]; a < actions.rowFirst()[r] + actions.rowCount()[r]; a++) {
        rowOf[a] = r;
      }
    }
    StringBuilder out = new StringBuilder();
    for (int i = 0; i < Math.min(k, order.length); i++) {
      int a = order[i];
      out.append(String.format(Locale.ROOT, " [row %d %s ref %.4f hip %.4f]", rowOf[a], actions.roles()[a].name(),
          reference[a], hip[a]));
    }
    return out.toString();
  }

  /** アームごとの確かめ（投げた例外とアサーションの失敗は、そのアームの失敗として集める）。 */
  @FunctionalInterface
  private interface ArmCheck {
    void run(HipFixtures.Arm arm) throws Exception;
  }

  /** アームごとに確かめる（あるアームが失敗しても、ほかのアームの確かめは止めない）。 */
  private static void eachArm(List<String> failures, List<HipFixtures.Arm> arms, ArmCheck check) {
    for (HipFixtures.Arm arm : arms) {
      try {
        check.run(arm);
      } catch (Exception | AssertionError failure) {
        log.info("PRISM-HIP-T5 arm {} stopped: {}", arm.label(), failure.toString());
        failures.add(arm.label() + " stopped: " + failure);
      }
    }
  }

  /** 冷えた GPU でバッチを {@value #WARMUP} + {@value #ITERATIONS} 回流し、カーネルごとの時間の中央値と経過時間の中央値を 1 行に書く。 */
  private static void measure(HipBackend backend, ByteBuffer blob, HipBatch batch, long version, String arm) {
    int decisions = batch.rows();
    ByteBuffer scores = ByteBuffer.allocateDirect(batch.actions() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    backend.plan().profile(true);
    float[][] millis = new float[ITERATIONS][];
    float[] totals = new float[ITERATIONS];
    for (int i = 0; i < WARMUP + ITERATIONS; i++) {
      long start = System.nanoTime();
      backend.plan().forward(blob.duplicate().position(0), batch, scores, version);
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
    log.info(
        "PRISM-HIP-SPEED arm {} rows {} tokens/row {} transitions/row {} actions/row {}:{} kernels={}ms"
            + " ({} us/decision) wall={}ms ({} us/decision)",
        arm,
        decisions,
        String.format(Locale.ROOT, "%.1f", ((double) fixed * decisions + batch.events()) / decisions),
        String.format(Locale.ROOT, "%.1f", (double) batch.transitions() / decisions),
        String.format(Locale.ROOT, "%.1f", (double) batch.actions() / decisions),
        line,
        String.format(Locale.ROOT, "%.3f", sum),
        String.format(Locale.ROOT, "%.3f", sum * 1e3 / decisions),
        String.format(Locale.ROOT, "%.3f", totals[ITERATIONS / 2]),
        String.format(Locale.ROOT, "%.3f", totals[ITERATIONS / 2] * 1e3 / decisions));
  }

  /** HIP、eager の BF16 と FP32、CPU の参照の s を、バッチをまたいでつなげた並びと、その手の形。 */
  private static final class Scores {
    private final List<Float> hip = new ArrayList<>();
    private final List<Float> bf16 = new ArrayList<>();
    private final List<Float> fp32 = new ArrayList<>();
    private final List<Float> reference = new ArrayList<>();
    private final List<HipParity.Actions> actions = new ArrayList<>();

    /** 1 バッチぶんの HIP の s を足し、同じバッチの eager の BF16 と FP32、参照の s を計算して足す。eager の BF16 の各段の値を返す。 */
    EagerStages.Stages add(
        float[] s, PrismNetwork network, ParameterStore store, NDManager manager, WeightPack pack, HostBatch batch) {
      EagerStages.Stages half;
      EagerStages.Stages full;
      try (NDManager scope = manager.newSubManager()) {
        half = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
      }
      try (NDManager scope = manager.newSubManager()) {
        full = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.FLOAT32);
      }
      append(hip, s);
      append(bf16, half.scores());
      append(fp32, full.scores());
      append(reference, new HipReference(pack).forward(batch));
      actions.add(HipParity.Actions.of(batch));
      return half;
    }

    float[] hip() {
      return floats(hip);
    }

    float[] bf16() {
      return floats(bf16);
    }

    float[] fp32() {
      return floats(fp32);
    }

    /** ログの行を書き、しきい値を外れたものを {@code failures} に足す。 */
    void check(List<String> failures, String arm, String prefix) {
      HipParity.Actions joined = join(actions);
      HipParity.Report half = HipParity.compare(floats(hip), floats(bf16), joined);
      HipParity.Report full = HipParity.compare(floats(hip), floats(fp32), joined);
      HipParity.Report cpu = HipParity.compare(floats(hip), floats(reference), joined);
      log.info("PRISM-HIP-T5 {}arm {} HIP vs eager BF16 (CPU): {} passes={}", prefix, arm, half.summary(), half.passes());
      log.info("PRISM-HIP-T5 {}arm {} HIP vs eager FP32 (CPU): {} passes={}", prefix, arm, full.summary(), full.passes());
      log.info("PRISM-HIP-T5 {}arm {} HIP vs CPU reference: {}", prefix, arm, cpu.summary());
      Diff half16 = Diff.fp32(floats(bf16), floats(hip));
      Diff full32 = Diff.fp32(floats(fp32), floats(hip));
      Diff mirror = Diff.fp32(floats(reference), floats(hip));
      log.info("PRISM-HIP-T5 {}arm {} {}", prefix, arm, half16.line("s vs eager BF16"));
      log.info("PRISM-HIP-T5 {}arm {} {}", prefix, arm, full32.line("s vs eager FP32"));
      log.info("PRISM-HIP-T5 {}arm {} {}", prefix, arm, mirror.line("s vs CPU reference"));
      String label = prefix + arm;
      log.info("PRISM-HIP-T5 {}arm {} argmax flips vs eager BF16: {}", prefix, arm, flips(floats(hip), floats(bf16), joined));
      argmax(failures, label, floats(hip), floats(bf16), floats(fp32), joined);
      if (!(half16.relativeRms() <= EAGER_LIMIT)) {
        failures.add(label + " s vs eager BF16 " + half16.relativeRms());
      }
      if (!(full32.relativeRms() <= FP32_LIMIT)) {
        failures.add(label + " s vs eager FP32 " + full32.relativeRms());
      }
      if (!(mirror.relativeRms() <= REFERENCE_LIMIT)) {
        failures.add(label + " s vs CPU reference " + mirror.relativeRms());
      }
    }

    private static void append(List<Float> target, float[] values) {
      for (float value : values) {
        target.add(value);
      }
    }
  }

  private static float[] floats(List<Float> values) {
    float[] result = new float[values.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  /** バッチごとの手の形を、s をつなげた並びに合わせた 1 つの手の形にまとめる。 */
  private static HipParity.Actions join(List<HipParity.Actions> parts) {
    List<Integer> first = new ArrayList<>();
    List<Integer> count = new ArrayList<>();
    List<Role> roles = new ArrayList<>();
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
        roles.toArray(new Role[0]));
  }

  /**
   * argmax の一致をログに書き、BF16 の丸めで順位の変わらない行での一致を確かめる。パラメーターを乱数でずらしたネットワークには、上位 2 手の差が
   * 丸めの差より小さい行が多い。そうした行では、丸める位置が同じ CPU の参照とでも argmax が入れ替わるので、全ての行での一致（しきい値
   * {@value HipParity#ARGMAX_LIMIT}）はログに書くだけにする。確かめるのは eager 自身の BF16 の丸めでは順位が変わらない行、つまり eager の
   * FP32 の上位 2 手の差が、その行の eager の BF16 と FP32 の s の差の最大の 2 倍を超える行で、HIP の argmax が eager の FP32 と一致する割合
   * （{@value HipParity#ARGMAX_LIMIT} 以上）。
   */
  private static void argmax(
      List<String> failures, String label, float[] hip, float[] bf16, float[] fp32, HipParity.Actions actions) {
    HipParity.Report half = HipParity.compare(hip, bf16, actions);
    int decided = 0;
    int agree = 0;
    for (int r = 0; r < actions.rowFirst().length; r++) {
      int first = actions.rowFirst()[r];
      int n = actions.rowCount()[r];
      if (n < 2) {
        continue;
      }
      double noise = 0;
      for (int k = first; k < first + n; k++) {
        noise = Math.max(noise, Math.abs((double) bf16[k] - fp32[k]));
      }
      if (gap(fp32, first, n) > 2 * noise) {
        decided++;
        agree += argmax(hip, first, n) == argmax(fp32, first, n) ? 1 : 0;
      }
    }
    double rate = decided == 0 ? 1 : (double) agree / decided;
    log.info(
        "PRISM-HIP-T5 {} argmax vs eager BF16 {} on all {} rows (line {} {}); vs eager FP32 {} on {} rows that eager's own BF16"
            + " rounding does not reorder",
        label,
        String.format(Locale.ROOT, "%.5f", half.argmaxAgreement()),
        half.rows(),
        HipParity.ARGMAX_LIMIT,
        half.argmaxAgreement() >= HipParity.ARGMAX_LIMIT ? "met" : "missed",
        String.format(Locale.ROOT, "%.5f", rate),
        decided);
    if (rate < HipParity.ARGMAX_LIMIT) {
      failures.add(label + " argmax on rows that BF16 rounding does not reorder " + rate + " (" + decided + " rows)");
    }
  }

  /** 行の上位 2 手の差。 */
  private static double gap(float[] scores, int first, int count) {
    double top = Double.NEGATIVE_INFINITY;
    double second = Double.NEGATIVE_INFINITY;
    for (int k = first; k < first + count; k++) {
      if (scores[k] > top) {
        second = top;
        top = scores[k];
      } else if (scores[k] > second) {
        second = scores[k];
      }
    }
    return top - second;
  }

  /**
   * argmax が食い違う行ごとに、参照の上位 2 手の差と、その行の s の差の最大を並べる。上位 2 手の差が s の差より小さければ、丸めの差で
   * 入れ替わる僅差の行。
   */
  static String flips(float[] hip, float[] reference, HipParity.Actions actions) {
    StringBuilder out = new StringBuilder();
    int count = 0;
    for (int r = 0; r < actions.rowFirst().length; r++) {
      int first = actions.rowFirst()[r];
      int n = actions.rowCount()[r];
      int a = argmax(hip, first, n);
      int b = argmax(reference, first, n);
      if (a == b) {
        continue;
      }
      double delta = 0;
      for (int k = first; k < first + n; k++) {
        delta = Math.max(delta, Math.abs((double) hip[k] - reference[k]));
      }
      count++;
      out.append(String.format(Locale.ROOT, " [row %d gap %.3e max|ds| %.3e]", r,
          (double) reference[b] - reference[a], delta));
    }
    return count + out.toString();
  }

  private static int argmax(float[] scores, int first, int count) {
    int best = first;
    for (int k = first + 1; k < first + count; k++) {
      if (scores[k] > scores[best]) {
        best = k;
      }
    }
    return best;
  }

  /** 行 [from, to) だけの手の形を返す（手の並びはバッチのまま）。 */
  private static HipParity.Actions subset(HipParity.Actions actions, int from, int to) {
    return new HipParity.Actions(
        Arrays.copyOfRange(actions.rowFirst(), from, to), Arrays.copyOfRange(actions.rowCount(), from, to),
        actions.roles());
  }
}


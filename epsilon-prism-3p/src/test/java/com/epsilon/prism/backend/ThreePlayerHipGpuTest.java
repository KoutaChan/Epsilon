package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.hip.LayerRole;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputFormat.Role;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
 * 三人打ちの PRISM で、HIP の推論が eager と一致するかを確かめる（rocm のグループ。GPU 1 枚）。ネットワークは三人打ちの入力形式で乱数で
 * 初期化した v2 の形と V3 で、行はエンジンで打った三人打ちの半荘の判断（全ての手の種類と、公開のイベントが多い行を含む）。
 *
 * <ul>
 *   <li>K0: 関係のコードが、CPU の参照とワードごとに一致する（行のトークンより後ろを指すワードも含む）
 *   <li>段ごと: カーネルに eager の各段の入力を与えて 1 つずつ実行する。出力は eager の段（CPU の BF16 と FP32）と相対 RMS
 *       {@value #EAGER_LIMIT} 以内、同じ入力からの CPU の参照と {@value #REFERENCE_LIMIT} 以内（K7 の s は {@value #SCORE_REFERENCE_LIMIT}）
 *   <li>通しの s: eager（GPU の BF16 と CPU の FP32）と相対 RMS {@value #SCORE_LIMIT} 以内、CPU の参照と {@value #SCORE_REFERENCE_LIMIT} 以内。
 *       argmax は、eager 自身の BF16 の丸めで順位が変わらない行で {@value HipParity#ARGMAX_LIMIT} 以上一致する。全ての行と自信のある行の
 *       一致はログに出すだけ
 *   <li>同じバッチを 2 回実行しても s がビット単位で同じ。行を並べ替え、前に別の行を足してバッチを組み直しても、同じ行の s はビット単位で同じ
 * </ul>
 */
@Test(groups = "rocm")
public class ThreePlayerHipGpuTest {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerHipGpuTest.class);

  private static final double EAGER_LIMIT = 2e-2;
  private static final double SCORE_LIMIT = 4e-2;
  private static final double REFERENCE_LIMIT = 1e-2;
  private static final double SCORE_REFERENCE_LIMIT = 2e-2;

  /** 通しの s を比べる、ふつうのバッチの数。 */
  private static final int BATCHES = 4;

  /** 長い行のバッチに入れる、公開のイベントが多い行の数。 */
  private static final int LONG_ROWS = 12;

  /** 推論に渡す重みのバージョン。 */
  private static final long VERSION = 1;

  /**
   * K0 の関係のコードを、CPU の参照（{@link HipReference#relationCodes}）とワードごとに比べる。全ての行で、行のトークンを覆うクエリと
   * キーのタイルの全ての組を比べる。ふつうのバッチと長い行のバッチで確かめる。
   */
  public void relationCodesMatchTheReference() {
    List<long[]> longRows = new ArrayList<>(ThreePlayerRows.mixedRows(2, 900));
    longRows.addAll(ThreePlayerRows.longRows(LONG_ROWS));
    ThreePlayerRows.Arm arm = ThreePlayerRows.Arm.V2;
    for (boolean isLong : new boolean[] {false, true}) {
      List<long[]> rows = isLong ? longRows : ThreePlayerRows.mixedRows(GpuCase.ROWS, 900);
      try (GpuCase c =
          new GpuCase(
              (isLong ? "long " : "") + arm.label(),
              manager -> ThreePlayerRows.network(manager, arm.shape(), 41),
              ThreePlayerRows.encode(rows))) {
        c.putInput();
        run(c, HipKernel.RELATION, 0);
        int blocks = c.architecture.counts().codeBlocks();
        int[] words = c.getInts(HipRegion.RELATION_CODE, c.hipBatch.rows() * blocks * blocks * 256);
        HipReference reference = c.reference();
        HipReference.Input in = c.input(reference);
        long checked = 0;
        long wrong = 0;
        for (int r = 0; r < in.rows; r++) {
          int[][] expected = reference.relationCodes(in, r);
          for (int queryBlock = 0; queryBlock < expected.length; queryBlock++) {
            for (int k = 0; k < expected[queryBlock].length; k++) {
              int at = ((r * blocks + queryBlock) * blocks + k / 256) * 256 + k % 256;
              checked++;
              wrong += words[at] == expected[queryBlock][k] ? 0 : 1;
            }
          }
        }
        log.info("PRISM-3P-HIP-K0 {}: {} words checked, {} differ from the CPU reference", c.name, checked, wrong);
        Assert.assertEquals(wrong, 0L, c.name + ": K0 words that differ from the CPU reference");
      }
    }
  }

  /** カーネルに eager の各段の入力を与え、出力を eager の段（BF16 と FP32）と CPU の参照と比べる。ふつうのバッチと長い行のバッチの両方で確かめる。 */
  public void kernelsMatchEagerAndTheReference() {
    List<String> failures = new ArrayList<>();
    List<long[]> longRows = new ArrayList<>(ThreePlayerRows.mixedRows(2, 900));
    longRows.addAll(ThreePlayerRows.longRows(LONG_ROWS));
    for (ThreePlayerRows.Arm arm : ThreePlayerRows.Arm.values()) {
      for (boolean isLong : new boolean[] {false, true}) {
        List<long[]> rows = isLong ? longRows : ThreePlayerRows.mixedRows(GpuCase.ROWS, 900);
        String name = (isLong ? "long " : "") + arm.label();
        try (GpuCase c =
            new GpuCase(
                name,
                manager -> ThreePlayerRows.network(manager, arm.shape(), 41 + 2L * arm.ordinal()),
                ThreePlayerRows.encode(rows))) {
          EagerStages.Stages full;
          try (NDManager scope = c.manager.newSubManager()) {
            full =
                EagerStages.run(
                    c.network, new ParameterStore(c.manager, false), EagerStages.inputs(scope, c.batch),
                    Precision.FLOAT32);
          }
          kernels(failures, c, c.stages, "eager BF16", true);
          kernels(failures, c, full, "eager FP32", false);
        } catch (RuntimeException failure) {
          failures.add(name + " stopped: " + failure);
        }
      }
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /**
   * カーネル K1〜K9 を 1 つずつ、段の値を入力にして実行し、出力を比べる。Attention に入れる正規化済みの入力は、層 0 では K1 が作り、
   * それより後の層では参照実装が段の値から作る（参照の正規化はカーネルと同じ丸め方）。
   *
   * @param e 入力にし、出力と比べる段の値
   * @param against ログの行に書く、比べる相手の名前
   * @param withReference 同じ入力の CPU の参照とも比べるなら true
   */
  private static void kernels(
      List<String> failures, GpuCase c, EagerStages.Stages e, String against, boolean withReference) {
    Checks checks = new Checks(failures, c.name + " vs " + against + " ", withReference);
    HipArchitecture a = c.architecture;
    int hidden = a.hidden();
    int attention = a.attention();
    int fixedValues = c.fixedTokens() * hidden;

    c.putInput();
    run(c, HipKernel.EMBED, 0);
    HipReference reference = c.reference();
    HipReference.Input in = c.input(reference);
    reference.embed(in);
    float[] x = c.getBf16(HipRegion.RESIDUAL, c.tokens() * hidden);
    float[] statics = c.getBf16(HipRegion.STATIC, e.statics().length);
    checks.eager("K1 x", Diff.bf16(e.embedded(), x));
    checks.eager("K1 t_s", Diff.bf16(e.statics(), statics));
    checks.reference("K1 x", Diff.bf16(reference.residual, x));
    checks.reference("K1 t_s", Diff.bf16(reference.statics, statics));

    int allTokens = a.allTokenLayers();
    for (int layer = 0; layer < allTokens; layer++) {
      float[] input = layer == 0 ? e.embedded() : e.layers().get(layer - 1);
      reference = c.reference();
      in = c.input(reference);
      c.putInput();
      run(c, HipKernel.RELATION, 0);
      if (layer == 0) {
        run(c, HipKernel.EMBED, 0);
        reference.embed(in);
      } else {
        c.putBf16(HipRegion.NORMALIZED, reference.attentionInput(input, layer));
      }
      run(c, HipKernel.ATTENTION_ALL, layer);
      float[] o = c.getBf16(HipRegion.ATTENTION, c.tokens() * attention);
      checks.eager("K2 o layer " + layer, Diff.bf16(e.attention().get(layer), o));
      reference.residual = input.clone();
      reference.attention(in, layer);
      checks.reference("K2 o layer " + layer, Diff.bf16(Arrays.copyOf(reference.attentionOut, o.length), o));

      boolean last = a.layer(layer + 1).role() == LayerRole.FIXED_TOKENS;
      int written = last ? fixedValues : input.length;
      c.putBf16(HipRegion.RESIDUAL, input);
      c.putBf16(HipRegion.ATTENTION, e.attention().get(layer));
      run(c, HipKernel.MLP_ALL, layer);
      x = c.getBf16(HipRegion.RESIDUAL, written);
      reference = c.reference();
      in = c.input(reference);
      reference.residual = input.clone();
      reference.attentionOut = e.attention().get(layer).clone();
      reference.mlp(in, layer);
      checks.eager("K3 x layer " + layer, Diff.bf16(Arrays.copyOf(e.layers().get(layer), written), x));
      checks.reference("K3 x layer " + layer, Diff.bf16(Arrays.copyOf(reference.residual, written), x));
      if (last) {
        int events = c.hipBatch.events();
        int width = 2 * attention;
        float[] keyValues = c.getBf16(HipRegion.EVENT_KEY_VALUE, a.copiedEventLayers() * events * width);
        reference.eventKeyValue(in);
        for (int f = 1; f < a.fixedLayers(); f++) {
          float[] actual = Arrays.copyOfRange(keyValues, (f - 1) * events * width, f * events * width);
          checks.eager("K3 event kv layer " + (allTokens + f), Diff.bf16(e.eventKeyValues().get(f), actual));
          checks.reference("K3 event kv layer " + (allTokens + f), Diff.bf16(reference.eventKeyValues[f], actual));
        }
        if (a.reread()) {
          int from = c.fixedTokens() * width;
          float[] memory = c.getBf16(HipRegion.REREAD_MEMORY, from + events * width);
          float[] actual = Arrays.copyOfRange(memory, from, memory.length);
          checks.eager("K3 reread events", Diff.bf16(Arrays.copyOfRange(e.rereadMemory(), from, memory.length), actual));
          checks.reference(
              "K3 reread events", Diff.bf16(Arrays.copyOfRange(reference.rereadMemory, from, memory.length), actual));
        }
      }
    }

    float[] backbone = e.layers().get(allTokens - 1);
    for (int layer = allTokens; layer < a.depth(); layer++) {
      float[] input = Arrays.copyOf(e.layers().get(layer - 1), fixedValues);
      reference = c.reference();
      in = c.input(reference);
      c.putInput();
      run(c, HipKernel.RELATION, 0);
      reference.residual = backbone.clone();
      reference.eventKeyValue(in);
      System.arraycopy(input, 0, reference.residual, 0, fixedValues);
      boolean first = a.fixedIndex(layer) == 0;
      c.putBf16(HipRegion.NORMALIZED, reference.attentionInput(first ? reference.residual : input, layer));
      if (a.copiedEventLayers() > 0) {
        c.putBf16(
            HipRegion.EVENT_KEY_VALUE,
            concatenate(Arrays.asList(reference.eventKeyValues).subList(1, reference.eventKeyValues.length)));
      }
      run(c, HipKernel.ATTENTION_FIXED, layer);
      float[] o = c.getBf16(HipRegion.ATTENTION, c.fixedTokens() * attention);
      reference.attention(in, layer);
      checks.eager("K4 o layer " + layer, Diff.bf16(e.attention().get(layer), o));
      checks.reference("K4 o layer " + layer, Diff.bf16(Arrays.copyOf(reference.attentionOut, o.length), o));

      float[] attended = e.attention().get(layer);
      c.putBf16(HipRegion.RESIDUAL, input);
      c.putBf16(HipRegion.ATTENTION, attended);
      reference = c.reference();
      in = c.input(reference);
      System.arraycopy(input, 0, reference.residual, 0, fixedValues);
      System.arraycopy(attended, 0, reference.attentionOut, 0, attended.length);
      if (layer + 1 < a.depth()) {
        run(c, HipKernel.MLP_FIXED, layer);
        x = c.getBf16(HipRegion.RESIDUAL, fixedValues);
        reference.mlp(in, layer);
        checks.eager("K3 fixed x layer " + layer, Diff.bf16(e.layers().get(layer), x));
        checks.reference("K3 fixed x layer " + layer, Diff.bf16(Arrays.copyOf(reference.residual, fixedValues), x));
      } else {
        run(c, HipKernel.MLP_LAST, layer);
        float[] state = c.getBf16(HipRegion.STATE, e.states().length);
        float[] memory = c.getBf16(HipRegion.MEMORY, e.memory().length);
        reference.mlpLast(in, layer);
        checks.eager("K5 y", Diff.bf16(e.states(), state));
        checks.eager("K5 KV", Diff.bf16(e.memory(), memory));
        checks.reference("K5 y", Diff.bf16(reference.state, state));
        checks.reference("K5 KV", Diff.bf16(reference.memory, memory));
      }
    }

    if (a.reread()) {
      int count = c.fixedTokens() * 2 * attention;
      c.putBf16(HipRegion.STATE, e.states());
      run(c, HipKernel.REREAD_KEY_VALUE, 0);
      float[] memory = c.getBf16(HipRegion.REREAD_MEMORY, count);
      reference = c.reference();
      reference.state = e.states().clone();
      reference.rereadKeyValue(c.input(reference));
      checks.eager("K8 reread fixed", Diff.bf16(Arrays.copyOf(e.rereadMemory(), count), memory));
      checks.reference("K8 reread fixed", Diff.bf16(Arrays.copyOf(reference.rereadMemory, count), memory));
    }

    c.putInput();
    c.putBf16(HipRegion.STATIC, e.statics());
    c.putBf16(HipRegion.STATE, e.states());
    c.putBf16(HipRegion.MEMORY, e.memory());
    run(c, HipKernel.CANDIDATE, 0);
    float[] z = c.getBf16(HipRegion.TRANSITION, e.candidates().length);
    reference = c.reference();
    reference.statics = e.statics().clone();
    reference.state = e.states().clone();
    reference.memory = e.memory().clone();
    reference.candidate(c.input(reference));
    String label = a.reread() ? "K6 z1" : "K6 z";
    checks.eager(label, Diff.bf16(e.candidates(), z));
    checks.reference(label, Diff.bf16(reference.transitions, z));
    if (!a.reread()) {
      float[] values = c.getFloats(HipRegion.VALUE, e.values().length);
      checks.eager("K6 B", Diff.fp32(e.values(), values));
      checks.reference("K6 B", Diff.fp32(reference.values, values));
    } else {
      c.putInput();
      c.putBf16(HipRegion.TRANSITION, e.candidates());
      c.putBf16(HipRegion.REREAD_MEMORY, e.rereadMemory());
      run(c, HipKernel.REREAD, 0);
      z = c.getBf16(HipRegion.TRANSITION, e.transitions().length);
      float[] values = c.getFloats(HipRegion.VALUE, e.values().length);
      reference = c.reference();
      reference.transitions = e.candidates().clone();
      reference.rereadMemory = e.rereadMemory().clone();
      reference.reread(c.input(reference));
      checks.eager("K9 z", Diff.bf16(e.transitions(), z));
      checks.eager("K9 B", Diff.fp32(e.values(), values));
      checks.reference("K9 z", Diff.bf16(reference.transitions, z));
      checks.reference("K9 B", Diff.fp32(reference.values, values));
    }

    c.putInput();
    c.putBf16(HipRegion.TRANSITION, e.transitions());
    c.putFloats(HipRegion.VALUE, e.values());
    c.putBf16(HipRegion.STATE, e.states());
    run(c, HipKernel.VALUE_GRAPH, 0);
    float[] scores = c.getFloats(HipRegion.SCORES, e.scores().length);
    reference = c.reference();
    reference.transitions = e.transitions().clone();
    reference.values = e.values().clone();
    reference.state = e.states().clone();
    reference.valueGraph(c.input(reference));
    checks.eager("K7 s", Diff.fp32(e.scores(), scores));
    checks.reference("K7 s", Diff.fp32(reference.scores, scores), SCORE_REFERENCE_LIMIT);
  }

  /** 段ごとの比べをログに出し、しきい値を外れたものを集める。参照と比べない回は、参照の行を出さない。 */
  private record Checks(List<String> failures, String prefix, boolean withReference) {

    void eager(String label, Diff diff) {
      check(prefix + label, diff, EAGER_LIMIT);
    }

    void reference(String label, Diff diff) {
      reference(label, diff, REFERENCE_LIMIT);
    }

    void reference(String label, Diff diff, double limit) {
      if (withReference) {
        check(prefix + label + " (CPU reference)", diff, limit);
      }
    }

    private void check(String label, Diff diff, double limit) {
      log.info("PRISM-3P-HIP-K {}", diff.line(label));
      if (!(diff.relativeRms() <= limit)) {
        failures.add(label + " " + diff.relativeRms());
      }
    }
  }

  /**
   * 通しの s を比べる。{@value #BATCHES} 個のふつうのバッチと長い行のバッチで、HIP の s を eager（GPU の BF16、CPU の FP32）と CPU の
   * 参照と比べる。argmax は、全ての行と自信のある行の一致をログに出し、eager 自身の BF16 の丸めで順位が変わらない行の一致を確かめる。
   */
  public void scoresMatchEager() throws IOException {
    List<HostBatch> batches = new ArrayList<>();
    for (int b = 0; b < BATCHES; b++) {
      batches.add(ThreePlayerRows.encode(ThreePlayerRows.mixedRows(GpuCase.ROWS, 600 + 13L * b)));
    }
    List<long[]> longRows = new ArrayList<>(ThreePlayerRows.mixedRows(2, 900));
    longRows.addAll(ThreePlayerRows.longRows(LONG_ROWS));
    HostBatch longBatch = ThreePlayerRows.encode(longRows);
    List<String> failures = new ArrayList<>();
    for (ThreePlayerRows.Arm arm : ThreePlayerRows.Arm.values()) {
      Path directory = Files.createTempDirectory("prism-3p-hip");
      try (GpuCase c =
              new GpuCase(
                  arm.label(),
                  manager -> ThreePlayerRows.network(manager, arm.shape(), 50 + arm.ordinal()),
                  batches.getFirst());
          NDManager gpu = NDManager.newBaseManager(Device.gpu(0))) {
        Path model = directory.resolve("model");
        c.network.save(model);
        PrismNetwork onGpu = PrismNetwork.load(model, gpu);
        Scores scores = new Scores();
        Scores longScores = new Scores();
        for (HostBatch batch : batches) {
          scores.add(c, onGpu, gpu, batch);
        }
        longScores.add(c, onGpu, gpu, longBatch);
        scores.check(failures, arm.label());
        longScores.check(failures, "long " + arm.label());
      } catch (RuntimeException failure) {
        failures.add(arm.label() + " stopped: " + failure);
      } finally {
        Snapshot.deleteTree(directory);
      }
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 同じバッチを 2 回実行しても s がビット単位で同じ。行を並べ替え、前に別の行を足して組み直しても、同じ行の s はビット単位で同じ（イベントの多い行も含む）。 */
  public void rebatchingKeepsRowScoresBitwise() {
    List<long[]> rows = new ArrayList<>(ThreePlayerRows.mixedRows(GpuCase.ROWS - 4, 900));
    rows.addAll(ThreePlayerRows.longRows(4));
    List<long[]> shuffled = new ArrayList<>(rows);
    Collections.shuffle(shuffled, new Random(9));
    List<long[]> prefixed = new ArrayList<>(ThreePlayerRows.simulatedRows(7, 700));
    prefixed.addAll(shuffled);
    HostBatch first = ThreePlayerRows.encode(rows);
    HostBatch second = ThreePlayerRows.encode(prefixed);
    for (ThreePlayerRows.Arm arm : ThreePlayerRows.Arm.values()) {
      try (GpuCase c =
          new GpuCase(
              arm.label(), manager -> ThreePlayerRows.network(manager, arm.shape(), 61 + arm.ordinal()), first)) {
        float[] a = c.backend.forward(first, VERSION);
        float[] b = c.backend.forward(second, VERSION);
        float[] again = c.backend.forward(first, VERSION);
        Assert.assertTrue(HipParity.bitwiseEqual(a, again), arm.label() + ": two runs of one batch differ");
        int compared = 0;
        for (int r = 0; r < rows.size(); r++) {
          int s = prefixed.indexOf(rows.get(r));
          Assert.assertEquals(second.plan().actions(s), first.plan().actions(r));
          for (int k = 0; k < first.plan().actions(r); k++) {
            Assert.assertEquals(
                Float.floatToRawIntBits(b[second.plan().firstAction(s) + k]),
                Float.floatToRawIntBits(a[first.plan().firstAction(r) + k]),
                arm.label() + " row " + r + " action " + k);
            compared++;
          }
        }
        log.info("PRISM-3P-HIP-T5 rebatch arm {}: {} scores bitwise equal, two runs bitwise equal", arm.label(), compared);
      }
    }
  }

  /** HIP・eager（GPU の BF16、CPU の FP32）・CPU の参照の s をバッチの順に集める。行ごとの手の位置と数、手の扱いも一緒に持つ。 */
  private static final class Scores {
    private final List<float[]> hip = new ArrayList<>();
    private final List<float[]> bf16 = new ArrayList<>();
    private final List<float[]> fp32 = new ArrayList<>();
    private final List<float[]> reference = new ArrayList<>();
    private final List<HipParity.Actions> actions = new ArrayList<>();

    void add(GpuCase c, PrismNetwork onGpu, NDManager gpu, HostBatch batch) {
      hip.add(c.backend.forward(batch, VERSION));
      try (NDManager scope = gpu.newSubManager()) {
        bf16.add(
            EagerStages.scores(
                onGpu, new ParameterStore(gpu, false), EagerStages.inputs(scope, batch), Precision.BFLOAT16));
      }
      try (NDManager scope = c.manager.newSubManager()) {
        fp32.add(
            EagerStages.scores(
                c.network, new ParameterStore(c.manager, false), EagerStages.inputs(scope, batch), Precision.FLOAT32));
      }
      reference.add(new HipReference(c.pack).forward(batch));
      actions.add(HipParity.Actions.of(batch));
    }

    /** 比べた結果をログに出し、しきい値を外れたものを失敗の一覧に足す。 */
    void check(List<String> failures, String label) {
      HipParity.Actions joined = join(actions);
      float[] h = concatenate(hip);
      float[] b = concatenate(bf16);
      float[] f = concatenate(fp32);
      float[] r = concatenate(reference);
      HipParity.Report half = HipParity.compare(h, b, joined);
      log.info("PRISM-3P-HIP-T5 {} HIP vs eager GPU BF16: {} passes={}", label, half.summary(), half.passes());
      log.info("PRISM-3P-HIP-T5 {} HIP vs eager CPU FP32: {}", label, HipParity.compare(h, f, joined).summary());
      log.info("PRISM-3P-HIP-T5 {} HIP vs CPU reference: {}", label, HipParity.compare(h, r, joined).summary());
      line(failures, label, "s vs eager GPU BF16", Diff.fp32(b, h), SCORE_LIMIT);
      line(failures, label, "s vs eager CPU FP32", Diff.fp32(f, h), SCORE_LIMIT);
      line(failures, label, "s vs CPU reference", Diff.fp32(r, h), SCORE_REFERENCE_LIMIT);
      argmax(failures, label, h, b, f, joined);
    }

    private static void line(List<String> failures, String label, String name, Diff diff, double limit) {
      log.info("PRISM-3P-HIP-T5 {} {}", label, diff.line(name));
      if (!(diff.relativeRms() <= limit)) {
        failures.add(label + " " + name + " " + diff.relativeRms());
      }
    }
  }

  /**
   * argmax の一致をログに出し、BF16 の丸めで順位が変わらない行の一致を確かめる。乱数で初期化したネットワークには、1 位と 2 位の手の差が
   * 丸めの差より小さい行が多い。そのため、全ての行の一致（しきい値 {@value HipParity#ARGMAX_LIMIT}）と自信のある行の一致はログに出すだけにする。
   * 確かめる行は、eager の FP32 での 1 位と 2 位の差が、その行の eager の BF16 と FP32 の s の差の最大値の 2 倍を超える行。その行で HIP の
   * argmax が eager の FP32 と一致する割合が {@value HipParity#ARGMAX_LIMIT} 以上であることを確かめる。そういう行が 1 つも無ければ失敗にする。
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
        "PRISM-3P-HIP-T5 {} argmax vs eager GPU BF16 {} on all {} rows (line {} {}), confident {} on {} rows; vs eager"
            + " FP32 {} on {} rows that eager's own BF16 rounding does not reorder",
        label,
        String.format(Locale.ROOT, "%.5f", half.argmaxAgreement()),
        half.rows(),
        HipParity.ARGMAX_LIMIT,
        half.argmaxAgreement() >= HipParity.ARGMAX_LIMIT ? "met" : "missed",
        String.format(Locale.ROOT, "%.5f", half.confidentAgreement()),
        half.confidentRows(),
        String.format(Locale.ROOT, "%.5f", rate),
        decided);
    if (decided == 0 || rate < HipParity.ARGMAX_LIMIT) {
      failures.add(label + " argmax on rows that BF16 rounding does not reorder " + rate + " (" + decided + " rows)");
    }
  }

  /** 行の中で、1 位と 2 位の手の s の差。 */
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

  private static int argmax(float[] scores, int first, int count) {
    int best = first;
    for (int k = first + 1; k < first + count; k++) {
      if (scores[k] > scores[best]) {
        best = k;
      }
    }
    return best;
  }

  /** バッチごとの手の位置・数・扱いを、s をつないだ並びに合わせて 1 つにまとめる。 */
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

  /** 配列を順につなぐ。 */
  private static float[] concatenate(List<float[]> parts) {
    float[] out = new float[parts.stream().mapToInt(part -> part.length).sum()];
    int at = 0;
    for (float[] part : parts) {
      System.arraycopy(part, 0, out, at, part.length);
      at += part.length;
    }
    return out;
  }

  private static void run(GpuCase c, HipKernel kernel, int layer) {
    c.plan().run(kernel, layer, c.hipBatch);
    c.plan().synchronize();
  }
}

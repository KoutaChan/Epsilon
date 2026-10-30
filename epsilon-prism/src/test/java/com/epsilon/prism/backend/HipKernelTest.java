package com.epsilon.prism.backend;

import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipRegion;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * カーネルごとの一致（rocm の組。GPU 1 枚）。各カーネルに CPU の eager（BF16）の
 * その段の入力を入れて GPU で 1 本だけ回し、eager のその段の出力と比べる（相対 RMS ≤ 2e-2、BF16 の 1 ulp 以内の割合は報告）。同じ入力で
 * CPU の参照（{@link HipReference}）の段も回して比べる（和の順の差だけのはず）。深さ 2 と 3 の両方。公開の出来事が 90 を超える行の
 * 注意は {@link #attentionLongRows} が別の塊で確かめ、記録の行の頭は「long depth L」。
 *
 * <p>検査つきの組み立てで範囲外があると GPU の trap で止まるので、カーネルを回す前に「PRISM-HIP-K start」を書いて流す（落ちた所が記録の
 * 最後の start で分かる）。
 */
@Test(groups = "rocm")
public class HipKernelTest {
  private static final Logger log = LoggerFactory.getLogger(HipKernelTest.class);

  private static final double LIMIT = 2e-2;
  private static final int PRECISION_DRAWS = 32;
  private static final long PRECISION_SEED = 4100;
  private static final String DUMP_PROPERTY = "prism.hip.k6dump";

  private final List<GpuCase> cases = new ArrayList<>();

  @BeforeClass(groups = "rocm")
  public void open() {
    cases.add(new GpuCase(2, 41));
    cases.add(new GpuCase(3, 43));
  }

  @AfterClass(groups = "rocm", alwaysRun = true)
  public void close() {
    for (GpuCase c : cases) {
      c.close();
    }
  }

  public void embed() {
    List<String> failures = new ArrayList<>();
    for (GpuCase c : cases) {
      c.putInput();
      run(c, HipKernel.EMBED, 0);
      HipReference reference = reference(c);
      reference.embed(input(c));
      int tokens = c.eager.embedded().length / hidden(c);
      float[] x = c.getBf16(HipRegion.RESIDUAL, tokens * hidden(c));
      float[] statics = c.getBf16(HipRegion.STATIC, c.eager.statics().length);
      check(failures, c, "K1 x vs eager", Diff.bf16(c.eager.embedded(), x));
      check(failures, c, "K1 t_s vs eager", Diff.bf16(c.eager.statics(), statics));
      float[] n = c.getBf16(HipRegion.NORMALIZED, tokens * hidden(c));
      check(failures, c, "K1 n vs reference",
          Diff.bf16(reference.attentionInput(Arrays.copyOf(reference.residual, n.length), 0), n));
      report(c, "K1 x vs reference", Diff.bf16(reference.residual, x));
      report(c, "K1 t_s vs reference", Diff.bf16(reference.statics, statics));
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  public void attention0() {
    List<String> failures = new ArrayList<>();
    for (GpuCase c : cases) {
      for (int layer = 0; layer + 1 < c.depth; layer++) {
        float[] input = layer == 0 ? c.eager.embedded() : c.eager.layers().get(layer - 1);
        HipReference reference = reference(c);
        c.putInput();
        run(c, HipKernel.RELATION, 0);
        c.putBf16(HipRegion.NORMALIZED, reference.attentionInput(input, layer));
        run(c, HipKernel.ATTENTION0, layer);
        float[] o = c.getBf16(HipRegion.ATTENTION, c.eager.attention().get(layer).length);
        reference.residual = input.clone();
        reference.attention(input(c), layer, false);
        check(failures, c, "K2 o layer " + layer + " vs eager", Diff.bf16(c.eager.attention().get(layer), o));
        report(c, "K2 o layer " + layer + " vs reference", Diff.bf16(reference.attentionOut, o));
      }
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /**
   * 普通の行 2 つと公開の出来事が 90 を超える行 2 つの塊（深さ 2）で K0 → K2・K4。行のトークンが 128 を超えるので、K2 は波の 2 つ目の
   * 問い合わせの塊の q を LDS に置いて読む道を通る。
   */
  public void attentionLongRows() {
    List<String> failures = new ArrayList<>();
    try (GpuCase c = new GpuCase("long depth 2", 2, 45, HipFixtures.longMixedRows())) {
      for (int layer = 0; layer < c.depth; layer++) {
        boolean last = layer == c.depth - 1;
        float[] input = layer == 0 ? c.eager.embedded() : c.eager.layers().get(layer - 1);
        HipReference reference = reference(c);
        c.putInput();
        run(c, HipKernel.RELATION, 0);
        c.putBf16(HipRegion.NORMALIZED, reference.attentionInput(input, layer));
        run(c, last ? HipKernel.ATTENTION1 : HipKernel.ATTENTION0, layer);
        float[] expected = c.eager.attention().get(layer);
        float[] o = c.getBf16(HipRegion.ATTENTION, expected.length);
        reference.residual = input.clone();
        reference.attention(input(c), layer, last);
        String label = (last ? "K4" : "K2") + " o layer " + layer;
        check(failures, c, label + " vs eager", Diff.bf16(expected, o));
        report(c, label + " vs reference", Diff.bf16(Arrays.copyOf(reference.attentionOut, expected.length), o));
      }
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  public void mlp0() {
    List<String> failures = new ArrayList<>();
    for (GpuCase c : cases) {
      for (int layer = 0; layer + 1 < c.depth; layer++) {
        float[] input = layer == 0 ? c.eager.embedded() : c.eager.layers().get(layer - 1);
        c.putBf16(HipRegion.RESIDUAL, input);
        c.putBf16(HipRegion.ATTENTION, c.eager.attention().get(layer));
        run(c, HipKernel.MLP0, layer);
        float[] x = c.getBf16(HipRegion.RESIDUAL, input.length);
        HipReference reference = reference(c);
        reference.residual = input.clone();
        reference.attentionOut = c.eager.attention().get(layer).clone();
        reference.mlp(input(c), layer);
        check(failures, c, "K3 x layer " + layer + " vs eager", Diff.bf16(c.eager.layers().get(layer), x));
        report(c, "K3 x layer " + layer + " vs reference", Diff.bf16(reference.residual, x));
        float[] n = c.getBf16(HipRegion.NORMALIZED, input.length);
        check(failures, c, "K3 n layer " + layer + " vs reference",
            Diff.bf16(reference.attentionInput(Arrays.copyOf(reference.residual, n.length), layer + 1), n));
      }
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  public void attention1() {
    List<String> failures = new ArrayList<>();
    for (GpuCase c : cases) {
      int last = c.depth - 1;
      float[] input = c.eager.layers().get(last - 1);
      HipReference reference = reference(c);
      c.putInput();
      run(c, HipKernel.RELATION, 0);
      c.putBf16(HipRegion.NORMALIZED, reference.attentionInput(input, last));
      run(c, HipKernel.ATTENTION1, last);
      float[] expected = c.eager.attention().get(last);
      float[] o = c.getBf16(HipRegion.ATTENTION, expected.length);
      reference.residual = input.clone();
      reference.attention(input(c), last, true);
      check(failures, c, "K4 o vs eager", Diff.bf16(expected, o));
      report(c, "K4 o vs reference", Diff.bf16(Arrays.copyOf(reference.attentionOut, expected.length), o));
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  public void mlpKv() {
    List<String> failures = new ArrayList<>();
    for (GpuCase c : cases) {
      int last = c.depth - 1;
      float[] input = c.eager.layers().get(last - 1);
      c.putBf16(HipRegion.RESIDUAL, input);
      c.putBf16(HipRegion.ATTENTION, c.eager.attention().get(last));
      run(c, HipKernel.MLP1_KV, last);
      float[] state = c.getBf16(HipRegion.STATE, c.eager.states().length);
      float[] memory = c.getBf16(HipRegion.MEMORY, c.eager.memory().length);
      HipReference reference = reference(c);
      reference.residual = input.clone();
      System.arraycopy(c.eager.attention().get(last), 0, reference.attentionOut, 0, c.eager.attention().get(last).length);
      reference.mlpKv(input(c), last);
      check(failures, c, "K5 y vs eager", Diff.bf16(c.eager.states(), state));
      check(failures, c, "K5 KV vs eager", Diff.bf16(c.eager.memory(), memory));
      report(c, "K5 y vs reference", Diff.bf16(reference.state, state));
      report(c, "K5 KV vs reference", Diff.bf16(reference.memory, memory));
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  public void candidate() {
    List<String> failures = new ArrayList<>();
    for (GpuCase c : cases) {
      c.putInput();
      c.putBf16(HipRegion.STATIC, c.eager.statics());
      c.putBf16(HipRegion.STATE, c.eager.states());
      c.putBf16(HipRegion.MEMORY, c.eager.memory());
      run(c, HipKernel.CANDIDATE, 0);
      float[] z = c.getBf16(HipRegion.TRANSITION, c.eager.transitions().length);
      float[] values = c.getFloats(HipRegion.VALUE, c.eager.values().length);
      HipReference reference = reference(c);
      reference.statics = c.eager.statics().clone();
      reference.state = c.eager.states().clone();
      reference.memory = c.eager.memory().clone();
      reference.candidate(input(c));
      check(failures, c, "K6 z vs eager", Diff.bf16(c.eager.transitions(), z));
      check(failures, c, "K6 B vs eager", Diff.fp32(c.eager.values(), values));
      report(c, "K6 z vs reference", Diff.bf16(reference.transitions, z));
      report(c, "K6 B vs reference", Diff.fp32(reference.values, values));
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /**
   * K6 の交差注意（段 5）の精度。深さ 2 と 3 の塊で行の鍵 MEMORY だけを {@link #PRECISION_DRAWS} 通りに揺らし（K6 で MEMORY を読むのは
   * 段 5 だけ）、K6 の z・B を交差注意だけ FP64 で計算して o を 1 回丸めた参照（{@link HipReference#exactCrossAttention}）と比べる。揺らしは
   * 種で決まるので、木どうしは同じ回の差（対の差）で比べる。eager との 1 本の差は eager の FP32 の順の丸めの模様との食い違いで、段 5 の
   * 順を替えると精度と関係なく 0.5〜2 倍ほど動く。システム属性 {@value #DUMP_PROPERTY} に dir があれば、塊ごとに回の z（BF16）と B を
   * {@code k6-<塊>.bin} に書く（木どうしで z の違う遷移を数えるため。並びは {@link #dumpDraw}）。
   */
  public void candidatePrecision() throws IOException {
    String dump = System.getProperty(DUMP_PROPERTY);
    for (GpuCase c : cases) {
      HipReference exact = reference(c);
      exact.exactCrossAttention = true;
      Path file = dump == null ? null : Path.of(dump, "k6-" + c.name.replace(' ', '-') + ".bin");
      double[][] sums = new double[2][2];
      for (int draw = 0; draw < PRECISION_DRAWS; draw++) {
        float[] memory = jitter(c.eager.memory(), new Random(PRECISION_SEED + draw));
        c.putInput();
        c.putBf16(HipRegion.STATIC, c.eager.statics());
        c.putBf16(HipRegion.STATE, c.eager.states());
        c.putBf16(HipRegion.MEMORY, memory);
        run(c, HipKernel.CANDIDATE, 0);
        float[] z = c.getBf16(HipRegion.TRANSITION, c.eager.transitions().length);
        float[] values = c.getFloats(HipRegion.VALUE, c.eager.values().length);
        exact.statics = c.eager.statics().clone();
        exact.state = c.eager.states().clone();
        exact.memory = memory;
        exact.candidate(input(c));
        Diff zExact = Diff.bf16(exact.transitions, z);
        Diff bExact = Diff.fp32(exact.values, values);
        log.info(
            "PRISM-HIP-K {} K6 precision draw {}: z vs FP64 attention {} (within 1 ulp {}), B vs FP64 attention {}",
            c.name,
            draw,
            String.format(Locale.ROOT, "%.4e", zExact.relativeRms()),
            String.format(Locale.ROOT, "%.5f", zExact.withinOneUlp()),
            String.format(Locale.ROOT, "%.4e", bExact.relativeRms()));
        if (file != null) {
          dumpDraw(file, draw, z, values);
        }
        double[] metrics = {zExact.relativeRms(), bExact.relativeRms()};
        for (int k = 0; k < metrics.length; k++) {
          sums[k][0] += metrics[k];
          sums[k][1] += metrics[k] * metrics[k];
        }
      }
      String[] labels = {"z vs FP64 attention", "B vs FP64 attention"};
      for (int k = 0; k < labels.length; k++) {
        double mean = sums[k][0] / PRECISION_DRAWS;
        double variance =
            Math.max(0, sums[k][1] / PRECISION_DRAWS - mean * mean) * PRECISION_DRAWS / (PRECISION_DRAWS - 1);
        log.info(
            "PRISM-HIP-K {} K6 precision {}: relative RMS mean {}, se {} ({} draws)",
            c.name,
            labels[k],
            String.format(Locale.ROOT, "%.4e", mean),
            String.format(Locale.ROOT, "%.2e", Math.sqrt(variance / PRECISION_DRAWS)),
            PRECISION_DRAWS);
      }
    }
  }

  /**
   * 回の z と B を file に足す（little endian）。回 0 で作り直して頭に int32 の 3 つ（回の数、z の数、B の数）を書き、回ごとに z の BF16 の
   * ビット（uint16）と B（float32）を続ける。
   */
  private static void dumpDraw(Path file, int draw, float[] z, float[] values) throws IOException {
    ByteBuffer bytes =
        ByteBuffer.allocate((draw == 0 ? 3 * Integer.BYTES : 0) + z.length * Short.BYTES + values.length * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    if (draw == 0) {
      Files.createDirectories(file.getParent());
      bytes.putInt(PRECISION_DRAWS).putInt(z.length).putInt(values.length);
    }
    for (float value : z) {
      bytes.putShort((short) (Float.floatToRawIntBits(value) >>> 16));
    }
    for (float value : values) {
      bytes.putFloat(value);
    }
    if (draw == 0) {
      Files.write(file, bytes.array());
    } else {
      Files.write(file, bytes.array(), StandardOpenOption.APPEND);
    }
  }

  /** BF16 の値の列を一様に 1 ± 5% 倍して BF16 に丸め直す。 */
  private static float[] jitter(float[] values, Random random) {
    float[] out = new float[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = HipReference.bf((float) (values[i] * (1 + 0.1 * (random.nextDouble() - 0.5))));
    }
    return out;
  }

  public void valueGraph() {
    List<String> failures = new ArrayList<>();
    for (GpuCase c : cases) {
      c.putInput();
      c.putBf16(HipRegion.TRANSITION, c.eager.transitions());
      c.putFloats(HipRegion.VALUE, c.eager.values());
      c.putBf16(HipRegion.STATE, c.eager.states());
      run(c, HipKernel.VALUE_GRAPH, 0);
      float[] scores = c.getFloats(HipRegion.SCORES, c.eager.scores().length);
      HipReference reference = reference(c);
      reference.transitions = c.eager.transitions().clone();
      reference.values = c.eager.values().clone();
      reference.state = c.eager.states().clone();
      reference.valueGraph(input(c));
      check(failures, c, "K7 s vs eager", Diff.fp32(c.eager.scores(), scores));
      report(c, "K7 s vs reference", Diff.fp32(reference.scores, scores));
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  private static void run(GpuCase c, HipKernel kernel, int layer) {
    log.info("PRISM-HIP-K start {} layer {} {}", kernel, layer, c.name);
    c.plan().run(kernel, layer, c.hipBatch);
    c.plan().synchronize();
    log.info("PRISM-HIP-K done {} layer {} {}", kernel, layer, c.name);
  }

  private static HipReference reference(GpuCase c) {
    HipReference reference = new HipReference(c.pack);
    reference.allocate(input(c));
    return reference;
  }

  private static HipReference.Input input(GpuCase c) {
    return new HipReference.Input(c.batch);
  }

  private static int hidden(GpuCase c) {
    return c.layout.shape().hidden();
  }

  private static void check(List<String> failures, GpuCase c, String label, Diff diff) {
    report(c, label, diff);
    if (!(diff.relativeRms() <= LIMIT)) {
      failures.add(c.name + " " + label + " " + diff.relativeRms());
    }
  }

  private static void report(GpuCase c, String label, Diff diff) {
    log.info("PRISM-HIP-K {} {}", c.name, diff.line(label));
  }
}

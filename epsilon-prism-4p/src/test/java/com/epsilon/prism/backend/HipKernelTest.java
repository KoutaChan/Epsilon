package com.epsilon.prism.backend;

import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipPack;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.hip.LayerRole;
import com.epsilon.prism.hip.TokenGroup;
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
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * カーネルごとの一致を確かめる（rocm のグループ。GPU 1 枚）。各カーネルに段の入力を入れて GPU で 1 本だけ動かし、同じ入力で動かした CPU の
 * 参照（{@link HipReference}）の段と比べる（相対 RMS ≤ {@value #REFERENCE_LIMIT}、s は {@value #SCORE_REFERENCE_LIMIT}）。その段の eager
 * （BF16）の出力とも比べる（≤ {@value #EAGER_LIMIT}）。形は全てのアームで試す（{@link HipFixtures.Arm}、システムプロパティで絞れる）。
 * 段の入力には eager の段の値を使う。
 *
 * <p>K1 の x と層 0 の qkv0（L0-FOLD）は、参照とビット単位で同じことを求める。K2 の層 0 は、参照の qkv0 を QKV0 に入れて動かす。最後の
 * 全トークンの層の K3 は x を局・席・牌のトークンだけに書き、2 つ目からの局・席・牌の層と 2 段目で使うイベントのキーとバリューを作る。
 * 局・席・牌の層の Attention（K4）は、最初の層なら全てのトークンの n を、後の層なら局・席・牌の n と参照のイベントの k・v を置いてから動かす。
 * 公開イベントが 90 を超える行の Attention は {@link #attentionLongRows} が別のバッチで確かめる。そのログの行では、形の名前の前に「long」が付く。
 *
 * <p>範囲チェックつきでビルドすると、範囲外の読み書きは GPU の trap で止まる。そこで、カーネルを動かす前に「PRISM-HIP-K start」をログに書いて
 * 流す（落ちた所は、ログの最後の start で分かる）。
 */
@Test(groups = "rocm")
public class HipKernelTest {
  private static final Logger log = LoggerFactory.getLogger(HipKernelTest.class);

  private static final double EAGER_LIMIT = 2e-2;
  private static final double REFERENCE_LIMIT = 1e-2;
  private static final double SCORE_REFERENCE_LIMIT = 2e-2;
  private static final int PRECISION_DRAWS = 32;
  private static final long PRECISION_SEED = 4100;
  private static final String DUMP_PROPERTY = "prism.hip.k6dump";

  private final List<GpuCase> cases = new ArrayList<>();

  @BeforeClass(groups = "rocm")
  public void open() {
    for (HipFixtures.Arm arm : HipFixtures.Arm.selected()) {
      cases.add(HipFixtures.gpuCase(arm, 41 + 2L * arm.ordinal()));
    }
  }

  @AfterClass(groups = "rocm", alwaysRun = true)
  public void close() {
    for (GpuCase c : cases) {
      c.close();
    }
  }

  public void embed() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      c.putInput();
      run(c, HipKernel.EMBED, 0);
      HipReference reference = c.reference();
      reference.embed(c.input(reference));
      float[] x = c.getBf16(HipRegion.RESIDUAL, c.tokens() * hidden(c));
      float[] statics = c.getBf16(HipRegion.STATIC, c.stages.statics().length);
      float[] qkv0 = c.getBf16(HipRegion.QKV0, c.tokens() * 3 * attention(c));
      eager(failures, c, "K1 x", c.stages.embedded(), x);
      eager(failures, c, "K1 t_s", c.stages.statics(), statics);
      bitwise(failures, c, "K1 qkv0 vs reference", reference.layer0Qkv(), qkv0);
      bitwise(failures, c, "K1 x vs reference", reference.residual, x);
      check(failures, c, "K1 t_s vs reference", Diff.bf16(reference.statics, statics), REFERENCE_LIMIT);
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 全トークンの層の Attention（K2）を確かめる。層 0 には K1 の qkv0 を、ほかの層には前の層の x の n を入れる。 */
  public void attentionAll() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      for (int layer = 0; layer < c.architecture.allTokenLayers(); layer++) {
        attentionAll(failures, c, layer);
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /**
   * 普通の行 2 つと、公開イベントが 90 を超える行 2 つからなるバッチで、K0 → K2・K4 を確かめる（全てのアーム）。行のトークンが 128 を
   * 超えるので、K2・K4 は 10 ウェーブ構成のカーネルを通る（普通の行はほかの構成のカーネル）。
   */
  public void attentionLongRows() {
    List<String> failures = new ArrayList<>();
    List<long[]> rows = HipFixtures.longMixedRows();
    for (HipFixtures.Arm arm : HipFixtures.Arm.selected()) {
      try (GpuCase c = HipFixtures.gpuCase("long " + arm.label(), arm.shape(), 45 + arm.ordinal(), rows)) {
        for (int layer = 0; layer < c.architecture.depth(); layer++) {
          if (c.architecture.layer(layer).role() == LayerRole.ALL_TOKENS) {
            attentionAll(failures, c, layer);
          } else {
            attentionFixed(failures, c, layer);
          }
        }
      } catch (RuntimeException failure) {
        failures.add("long " + arm.label() + " stopped: " + failure.getMessage());
      }
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /**
   * 全トークンの層の MLP（K3 の MLP_ALL）を確かめる。n は全てのトークンに書く。次が全トークンの層なら、x も全てのトークンに書く。最後の
   * 全トークンの層は x を局・席・牌のトークンだけに書き、2 つ目からの局・席・牌の層で使うイベントの [k | v] を作る。2 段目のある形では、
   * 2 段目で使うイベントのキーとバリューも作る。
   */
  public void mlpAll() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      HipArchitecture architecture = c.architecture;
      for (int layer = 0; layer < architecture.allTokenLayers(); layer++) {
        float[] input = layer == 0 ? c.stages.embedded() : c.stages.layers().get(layer - 1);
        boolean last = architecture.layer(layer + 1).role() == LayerRole.FIXED_TOKENS;
        int written = last ? c.fixedTokens() * hidden(c) : input.length;
        c.putBf16(HipRegion.RESIDUAL, input);
        c.putBf16(HipRegion.ATTENTION, c.stages.attention().get(layer));
        run(c, HipKernel.MLP_ALL, layer);
        float[] x = c.getBf16(HipRegion.RESIDUAL, written);
        float[] n = c.getBf16(HipRegion.NORMALIZED, input.length);
        HipReference reference = c.reference();
        reference.residual = input.clone();
        reference.attentionOut = c.stages.attention().get(layer).clone();
        reference.mlp(c.input(reference), layer);
        eager(failures, c, "K3 x layer " + layer, Arrays.copyOf(c.stages.layers().get(layer), written), x);
        check(failures, c, "K3 x layer " + layer + " vs reference",
            Diff.bf16(Arrays.copyOf(reference.residual, written), x), REFERENCE_LIMIT);
        check(failures, c, "K3 n layer " + layer + " vs reference",
            Diff.bf16(reference.attentionInput(reference.residual, layer + 1), n), REFERENCE_LIMIT);
        if (last) {
          eventKeyValues(failures, c, reference);
        }
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /**
   * 最後の全トークンの層の K3 が作ったイベントのキーとバリュー（2 つ目からの局・席・牌の層と 2 段目で使うもの）を、eager の段と比べ、
   * K3 の後の参照（参照の x から作ったキーとバリュー）とも比べる。
   *
   * @param reference 最後の全トークンの層の MLP を済ませた参照
   */
  private static void eventKeyValues(List<String> failures, GpuCase c, HipReference reference) {
    HipArchitecture architecture = c.architecture;
    int events = c.hipBatch.events();
    int width = 2 * attention(c);
    float[] keyValues = c.getBf16(HipRegion.EVENT_KEY_VALUE, architecture.copiedEventLayers() * events * width);
    reference.eventKeyValue(c.input(reference));
    for (int f = 1; f < architecture.fixedLayers(); f++) {
      int layer = architecture.allTokenLayers() + f;
      float[] actual = Arrays.copyOfRange(keyValues, (f - 1) * events * width, f * events * width);
      eager(failures, c, "K3 event kv layer " + layer, c.stages.eventKeyValues().get(f), actual);
      check(failures, c, "K3 event kv layer " + layer + " vs reference", Diff.bf16(reference.eventKeyValues[f], actual),
          REFERENCE_LIMIT);
    }
    if (c.reread()) {
      int from = c.fixedTokens() * width;
      float[] memory = c.getBf16(HipRegion.REREAD_MEMORY, from + events * width);
      float[] actual = Arrays.copyOfRange(memory, from, memory.length);
      eager(failures, c, "K3 reread events", Arrays.copyOfRange(c.stages.rereadMemory(), from, memory.length), actual);
      check(failures, c, "K3 reread events vs reference",
          Diff.bf16(Arrays.copyOfRange(reference.rereadMemory, from, memory.length), actual), REFERENCE_LIMIT);
    }
  }

  /** 局・席・牌の層の Attention（K4）を確かめる。 */
  public void attentionFixed() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      for (int layer = c.architecture.allTokenLayers(); layer < c.architecture.depth(); layer++) {
        attentionFixed(failures, c, layer);
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 最後でない局・席・牌の層の MLP（K3 の MLP_FIXED）を確かめる。型ごとの FFN なら、局・席と牌の 2 区間に分けて計算する。 */
  public void mlpFixed() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      HipArchitecture architecture = c.architecture;
      for (int layer = architecture.allTokenLayers(); layer + 1 < architecture.depth(); layer++) {
        int fixedValues = c.fixedTokens() * hidden(c);
        float[] input = Arrays.copyOf(c.stages.layers().get(layer - 1), fixedValues);
        c.putBf16(HipRegion.RESIDUAL, input);
        c.putBf16(HipRegion.ATTENTION, c.stages.attention().get(layer));
        run(c, HipKernel.MLP_FIXED, layer);
        float[] x = c.getBf16(HipRegion.RESIDUAL, fixedValues);
        float[] n = c.getBf16(HipRegion.NORMALIZED, fixedValues);
        HipReference reference = c.reference();
        System.arraycopy(input, 0, reference.residual, 0, fixedValues);
        float[] attended = c.stages.attention().get(layer);
        System.arraycopy(attended, 0, reference.attentionOut, 0, attended.length);
        reference.mlp(c.input(reference), layer);
        float[] expected = Arrays.copyOf(reference.residual, fixedValues);
        eager(failures, c, "K3 fixed x layer " + layer, c.stages.layers().get(layer), x);
        check(failures, c, "K3 fixed x layer " + layer + " vs reference", Diff.bf16(expected, x), REFERENCE_LIMIT);
        check(failures, c, "K3 fixed n layer " + layer + " vs reference",
            Diff.bf16(reference.attentionInput(expected, layer + 1), n), REFERENCE_LIMIT);
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 最後の層の MLP（K5）を確かめる。出力は最後の正規化の y（型ごとの形ならグループごとのゲイン）と、Cross-Attention のキーとバリュー。 */
  public void mlpLast() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      int last = c.architecture.depth() - 1;
      int fixedValues = c.fixedTokens() * hidden(c);
      float[] input = Arrays.copyOf(c.stages.layers().get(last - 1), fixedValues);
      c.putBf16(HipRegion.RESIDUAL, input);
      c.putBf16(HipRegion.ATTENTION, c.stages.attention().get(last));
      run(c, HipKernel.MLP_LAST, last);
      float[] state = c.getBf16(HipRegion.STATE, c.stages.states().length);
      float[] memory = c.getBf16(HipRegion.MEMORY, c.stages.memory().length);
      HipReference reference = c.reference();
      System.arraycopy(input, 0, reference.residual, 0, fixedValues);
      float[] attended = c.stages.attention().get(last);
      System.arraycopy(attended, 0, reference.attentionOut, 0, attended.length);
      reference.mlpLast(c.input(reference), last);
      eager(failures, c, "K5 y", c.stages.states(), state);
      eager(failures, c, "K5 KV", c.stages.memory(), memory);
      check(failures, c, "K5 y vs reference", Diff.bf16(reference.state, state), REFERENCE_LIMIT);
      check(failures, c, "K5 KV vs reference", Diff.bf16(reference.memory, memory), REFERENCE_LIMIT);
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** K8 の REREAD_KEY_VALUE を確かめる。y から、2 段目で使う局・席・牌のキーとバリュー [R, F, 2A] を作る。 */
  public void rereadKeyValue() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      if (!c.reread()) {
        return;
      }
      int count = c.fixedTokens() * 2 * attention(c);
      c.putBf16(HipRegion.STATE, c.stages.states());
      run(c, HipKernel.REREAD_KEY_VALUE, 0);
      float[] memory = c.getBf16(HipRegion.REREAD_MEMORY, count);
      HipReference reference = c.reference();
      reference.state = c.stages.states().clone();
      reference.rereadKeyValue(c.input(reference));
      eager(failures, c, "K8 reread fixed", Arrays.copyOf(c.stages.rereadMemory(), count), memory);
      check(failures, c, "K8 reread fixed vs reference", Diff.bf16(Arrays.copyOf(reference.rereadMemory, count), memory),
          REFERENCE_LIMIT);
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** K6 を確かめる。出力は候補層の 1 段目の z（2 段目のある形では z1）で、1 段の形では B も比べる。 */
  public void candidate() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      c.putInput();
      c.putBf16(HipRegion.STATIC, c.stages.statics());
      c.putBf16(HipRegion.STATE, c.stages.states());
      c.putBf16(HipRegion.MEMORY, c.stages.memory());
      run(c, HipKernel.CANDIDATE, 0);
      float[] z = c.getBf16(HipRegion.TRANSITION, c.stages.candidates().length);
      HipReference reference = c.reference();
      reference.statics = c.stages.statics().clone();
      reference.state = c.stages.states().clone();
      reference.memory = c.stages.memory().clone();
      reference.candidate(c.input(reference));
      String label = c.reread() ? "K6 z1" : "K6 z";
      eager(failures, c, label, c.stages.candidates(), z);
      check(failures, c, label + " vs reference", Diff.bf16(reference.transitions, z), REFERENCE_LIMIT);
      if (!c.reread()) {
        float[] values = c.getFloats(HipRegion.VALUE, c.stages.values().length);
        eagerFp32(failures, c, "K6 B", c.stages.values(), values);
        check(failures, c, "K6 B vs reference", Diff.fp32(reference.values, values), REFERENCE_LIMIT);
      }
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** K9 を確かめる。出力は候補層の 2 段目の z と B。z − z1（2 段目が足した分）の一致もログに出す。 */
  public void reread() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      if (!c.reread()) {
        return;
      }
      c.putInput();
      c.putBf16(HipRegion.TRANSITION, c.stages.candidates());
      c.putBf16(HipRegion.REREAD_MEMORY, c.stages.rereadMemory());
      run(c, HipKernel.REREAD, 0);
      float[] z = c.getBf16(HipRegion.TRANSITION, c.stages.transitions().length);
      float[] values = c.getFloats(HipRegion.VALUE, c.stages.values().length);
      HipReference reference = c.reference();
      reference.transitions = c.stages.candidates().clone();
      reference.rereadMemory = c.stages.rereadMemory().clone();
      reference.reread(c.input(reference));
      eager(failures, c, "K9 z", c.stages.transitions(), z);
      eagerFp32(failures, c, "K9 B", c.stages.values(), values);
      check(failures, c, "K9 z vs reference", Diff.bf16(reference.transitions, z), REFERENCE_LIMIT);
      check(failures, c, "K9 B vs reference", Diff.fp32(reference.values, values), REFERENCE_LIMIT);
      report(c, "K9 z - z1 vs reference",
          Diff.fp32(minus(reference.transitions, c.stages.candidates()), minus(z, c.stages.candidates())));
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /**
   * K6 の Cross-Attention（段 5）の精度を測る（1 段の四人打ちの形）。行のキー MEMORY だけを {@link #PRECISION_DRAWS} 通りに揺らし（K6 で
   * MEMORY を読むのは段 5 だけ）、K6 の z・B を参照（{@link HipReference#exactCrossAttention}）と比べる。参照は Cross-Attention だけを FP64
   * で計算し、o を 1 回だけ丸める。揺らし方は乱数の種で決まるので、コードの違うビルドどうしは同じ回の差（対の差）で比べる。システムプロパティ
   * {@value #DUMP_PROPERTY} にディレクトリがあれば、形ごとに各回の z（BF16）と B を {@code k6-<形の名前>.bin} に書く（並びは {@link #dumpDraw}）。
   */
  public void candidatePrecision() throws IOException {
    String dump = System.getProperty(DUMP_PROPERTY);
    for (GpuCase c : cases) {
      if (c.reread()) {
        continue;
      }
      HipReference exact = c.reference();
      exact.exactCrossAttention = true;
      Path file = dump == null ? null : Path.of(dump, "k6-" + c.name.replace(' ', '-') + ".bin");
      double[][] sums = new double[2][2];
      for (int draw = 0; draw < PRECISION_DRAWS; draw++) {
        float[] memory = jitter(c.stages.memory(), new Random(PRECISION_SEED + draw));
        c.putInput();
        c.putBf16(HipRegion.STATIC, c.stages.statics());
        c.putBf16(HipRegion.STATE, c.stages.states());
        c.putBf16(HipRegion.MEMORY, memory);
        run(c, HipKernel.CANDIDATE, 0);
        float[] z = c.getBf16(HipRegion.TRANSITION, c.stages.transitions().length);
        float[] values = c.getFloats(HipRegion.VALUE, c.stages.values().length);
        exact.statics = c.stages.statics().clone();
        exact.state = c.stages.states().clone();
        exact.memory = memory;
        exact.candidate(c.input(exact));
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
   * その回の z と B をファイルに追記する（little endian）。回 0 ではファイルを作り直し、ヘッダーとして int32 を 3 つ（回の数、z の数、B の数）
   * 書く。回ごとに、z の BF16 のビット（uint16）と B（float32）を続ける。
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

  /** BF16 の値の列に、一様乱数で選んだ 1 ± 5% の倍率を掛けて BF16 に丸め直す。 */
  private static float[] jitter(float[] values, Random random) {
    float[] out = new float[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = HipReference.bf((float) (values[i] * (1 + 0.1 * (random.nextDouble() - 0.5))));
    }
    return out;
  }

  public void valueGraph() {
    List<String> failures = new ArrayList<>();
    eachCase(failures, c -> {
      c.putInput();
      c.putBf16(HipRegion.TRANSITION, c.stages.transitions());
      c.putFloats(HipRegion.VALUE, c.stages.values());
      c.putBf16(HipRegion.STATE, c.stages.states());
      run(c, HipKernel.VALUE_GRAPH, 0);
      float[] scores = c.getFloats(HipRegion.SCORES, c.stages.scores().length);
      HipReference reference = c.reference();
      reference.transitions = c.stages.transitions().clone();
      reference.values = c.stages.values().clone();
      reference.state = c.stages.states().clone();
      reference.valueGraph(c.input(reference));
      eagerFp32(failures, c, "K7 s", c.stages.scores(), scores);
      check(failures, c, "K7 s vs reference", Diff.fp32(reference.scores, scores), SCORE_REFERENCE_LIMIT);
    });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 形ごとに確かめ、例外もその形の失敗として集める（ある形が失敗しても、ほかの形の確かめは止めない）。 */
  private void eachCase(List<String> failures, Consumer<GpuCase> check) {
    for (GpuCase c : cases) {
      try {
        check.accept(c);
      } catch (RuntimeException failure) {
        log.info("PRISM-HIP-K {} stopped: {}", c.name, failure.getMessage());
        failures.add(c.name + " stopped: " + failure.getMessage());
      }
    }
  }

  /**
   * 新しい重みのスロット（候補層の 2 段目、型ごとの FFN と最後の正規化のゲイン）のどの項目でも、真ん中の要素 1 つを NaN にすると、重みの公開は
   * そのスロットと項目を名指しした例外で止まる（四人打ちの v3 の形）。止まった後も計画は前の重みのままで、s はビット単位で変わらない。
   */
  public void publishRejectsNanInTheNewWeightSlots() {
    for (GpuCase c : cases) {
      if (!c.name.equals(HipFixtures.Arm.V3.label())) {
        continue;
      }
      float[] before = c.forward();
      HipPack.Header header = c.layout.header();
      int checked = 0;
      for (HipWeight weight : HipWeight.values()) {
        for (int entry = 0; entry < weight.index().entries(); entry++) {
          long offset = header.weight(weight, entry);
          if (offset < 0 || !newSlot(weight, entry)) {
            continue;
          }
          ByteBuffer broken = ByteBuffer.allocateDirect(c.pack.buffer().remaining()).order(ByteOrder.LITTLE_ENDIAN);
          broken.put(c.pack.buffer()).flip();
          int middle = Math.toIntExact(offset + header.elements(weight, entry) / 2 * weight.element().bytes());
          switch (weight.element()) {
            case FP32 -> broken.putInt(middle, 0x7fc00000);
            case BF16 -> broken.putShort(middle, (short) 0x7fc0);
            default -> throw new IllegalStateException(weight + " holds no floating point values");
          }
          String named = "weight slot " + weight.ordinal() + " entry " + entry + " contains NaN";
          RuntimeException failure = Assert.expectThrows(RuntimeException.class, () -> c.plan().uploadWeights(broken, 2));
          Assert.assertTrue(String.valueOf(failure.getMessage()).contains(named),
              weight + " entry " + entry + ": " + failure.getMessage());
          checked++;
        }
      }
      Assert.assertTrue(HipParity.bitwiseEqual(before, c.forward()), c.name + ": a rejected publish changed the scores");
      log.info("PRISM-HIP-K {} NaN in each of {} entries of the new weight slots stops the publish naming the slot; the plan"
          + " keeps its weights", c.name, checked);
    }
  }

  /** 2 段目のスロットか、型ごとのグループ（局・席、牌）の FFN と最後の正規化の項目なら true。 */
  private static boolean newSlot(HipWeight weight, int entry) {
    return switch (weight.index()) {
      case SINGLE, LAYER -> weight.reread();
      case FFN -> entry != HipWeight.CANDIDATE_FFN && entry % TokenGroup.values().length != TokenGroup.ALL.ordinal();
      case GROUP -> entry != TokenGroup.ALL.ordinal();
    };
  }

  /** 全トークンの層 {@code layer} で K0 → K2 を確かめる。 */
  private static void attentionAll(List<String> failures, GpuCase c, int layer) {
    float[] input = layer == 0 ? c.stages.embedded() : c.stages.layers().get(layer - 1);
    HipReference reference = c.reference();
    HipReference.Input in = c.input(reference);
    c.putInput();
    run(c, HipKernel.RELATION, 0);
    if (layer == 0) {
      reference.embed(in);
      c.putBf16(HipRegion.QKV0, reference.layer0Qkv());
    } else {
      c.putBf16(HipRegion.NORMALIZED, reference.attentionInput(input, layer));
    }
    run(c, HipKernel.ATTENTION_ALL, layer);
    float[] o = c.getBf16(HipRegion.ATTENTION, c.tokens() * attention(c));
    reference.residual = input.clone();
    reference.attention(in, layer);
    eager(failures, c, "K2 o layer " + layer, c.stages.attention().get(layer), o);
    check(failures, c, "K2 o layer " + layer + " vs reference", Diff.bf16(reference.attentionOut, o), REFERENCE_LIMIT);
  }

  /**
   * 局・席・牌の層 {@code layer} で K0 → K4 を確かめる。参照のイベントの k・v は、最後の全トークンの層の x から作る。最初の局・席・牌の層では、
   * 全てのトークンの n（局・席・牌は前の層の x から、イベントは最後の全トークンの層の x から）を NORMALIZED に置き、K4 がイベントの k・v も
   * 射影する。後の層では、局・席・牌の n を NORMALIZED に、参照の 2 つ目からの層のイベントの k・v を EVENT_KEY_VALUE に置く。
   */
  private static void attentionFixed(List<String> failures, GpuCase c, int layer) {
    int fixedValues = c.fixedTokens() * hidden(c);
    float[] backbone = c.stages.layers().get(c.architecture.allTokenLayers() - 1);
    float[] input = Arrays.copyOf(c.stages.layers().get(layer - 1), fixedValues);
    HipReference reference = c.reference();
    HipReference.Input in = c.input(reference);
    c.putInput();
    run(c, HipKernel.RELATION, 0);
    reference.residual = backbone.clone();
    reference.eventKeyValue(in);
    System.arraycopy(input, 0, reference.residual, 0, fixedValues);
    boolean first = c.architecture.fixedIndex(layer) == 0;
    c.putBf16(HipRegion.NORMALIZED, reference.attentionInput(first ? reference.residual : input, layer));
    if (c.architecture.copiedEventLayers() > 0) {
      c.putBf16(HipRegion.EVENT_KEY_VALUE,
          concatenate(Arrays.asList(reference.eventKeyValues).subList(1, reference.eventKeyValues.length)));
    }
    run(c, HipKernel.ATTENTION_FIXED, layer);
    float[] o = c.getBf16(HipRegion.ATTENTION, c.fixedTokens() * attention(c));
    reference.attention(in, layer);
    eager(failures, c, "K4 o layer " + layer, c.stages.attention().get(layer), o);
    check(failures, c, "K4 o layer " + layer + " vs reference",
        Diff.bf16(Arrays.copyOf(reference.attentionOut, o.length), o), REFERENCE_LIMIT);
  }

  private static void run(GpuCase c, HipKernel kernel, int layer) {
    log.info("PRISM-HIP-K start {} layer {} {}", kernel, layer, c.name);
    c.plan().run(kernel, layer, c.hipBatch);
    c.plan().synchronize();
    log.info("PRISM-HIP-K done {} layer {} {}", kernel, layer, c.name);
  }

  /** 参照とビット単位で同じことを求める（違う要素の数もログの行に出す）。 */
  private static void bitwise(List<String> failures, GpuCase c, String label, float[] expected, float[] actual) {
    int differ = 0;
    for (int i = 0; i < expected.length; i++) {
      differ += Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(actual[i]) ? 1 : 0;
    }
    log.info("PRISM-HIP-K {} {}, bits differ {}", c.name, Diff.bf16(expected, actual).line(label), differ);
    if (differ != 0) {
      failures.add(c.name + " " + label + " differs in " + differ + " of " + expected.length + " values");
    }
  }

  /** eager の BF16 の段と比べる。 */
  private static void eager(List<String> failures, GpuCase c, String label, float[] expected, float[] actual) {
    check(failures, c, label + " vs eager", Diff.bf16(expected, actual), EAGER_LIMIT);
  }

  private static void eagerFp32(List<String> failures, GpuCase c, String label, float[] expected, float[] actual) {
    check(failures, c, label + " vs eager", Diff.fp32(expected, actual), EAGER_LIMIT);
  }

  /** 値の列を順につなげる。 */
  private static float[] concatenate(List<float[]> parts) {
    float[] out = new float[parts.stream().mapToInt(part -> part.length).sum()];
    int at = 0;
    for (float[] part : parts) {
      System.arraycopy(part, 0, out, at, part.length);
      at += part.length;
    }
    return out;
  }

  private static float[] minus(float[] a, float[] b) {
    float[] out = new float[a.length];
    for (int i = 0; i < a.length; i++) {
      out[i] = a[i] - b[i];
    }
    return out;
  }

  private static int hidden(GpuCase c) {
    return c.architecture.hidden();
  }

  private static int attention(GpuCase c) {
    return c.architecture.attention();
  }

  private static void check(List<String> failures, GpuCase c, String label, Diff diff, double limit) {
    report(c, label, diff);
    if (!(diff.relativeRms() <= limit)) {
      failures.add(c.name + " " + label + " " + diff.relativeRms());
    }
  }

  private static void report(GpuCase c, String label, Diff diff) {
    log.info("PRISM-HIP-K {} {}", c.name, diff.line(label));
  }
}

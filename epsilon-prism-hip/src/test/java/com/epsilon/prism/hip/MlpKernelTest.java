package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;

import java.lang.ref.Reference;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * K3・K5 を GPU 1 枚で動かし、CPU の参照と比べる。型を分けない FFN（区間 1 つ）と、型ごとの FFN（区間 2 つ。局・席の幅は 1536 か
 * 512、牌の幅は 768）の両方を、四人打ちと三人打ちの数で試す。バッチは 37 行とイベント 100 個で、局・席の区間の終わり（四人打ち 185、
 * 三人打ち 148）が 48 の倍数にならないようにしてある。
 */
public class MlpKernelTest {

  private static final Logger log = LoggerFactory.getLogger(MlpKernelTest.class);

  private static final int ROWS = 37;
  private static final int EVENTS = 100;

  /** 前に足すバッチの大きさ（{@link MlpCase#prepend}）。 */
  private static final int PREPENDED_ROWS = 11;
  private static final int PREPENDED_EVENTS = 23;

  private static final double RELATIVE_RMS = 1e-2;
  private static final double WITHIN_ONE_ULP = 0.99;

  private static final HipArchitecture SHARED = TestPack.architecture(512, TestPack.FOUR_PLAYERS);
  private static final HipArchitecture SHARED_DEPTH3 = MlpCase.sharedDepth3(512);
  private static final HipArchitecture TYPED_FOUR = MlpCase.typed(TestPack.FOUR_PLAYERS, 1536);
  private static final HipArchitecture TYPED_FOUR_CONTEXT_512 = MlpCase.typed(TestPack.FOUR_PLAYERS, 512);
  private static final HipArchitecture TYPED_THREE = MlpCase.typed(MlpCase.THREE_PLAYERS, 1536);

  private HipLibrary library;
  private PtStream stream;
  private long version;

  @BeforeClass(groups = "rocm")
  public void openStream() {
    library = HipLibrary.load();
    stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
  }

  @AfterClass(groups = "rocm", alwaysRun = true)
  public void closeStream() {
    if (stream != null) {
      stream.close();
    }
  }

  /** K3 と K5 の全ての形（名前、形の記述、カーネル、層）。 */
  @DataProvider
  public Object[][] kernels() {
    return new Object[][] {
      {"v2 K3", SHARED, HipKernel.MLP_ALL, 0},
      {"v2 depth 3 K3", SHARED_DEPTH3, HipKernel.MLP_ALL, 0},
      {"v2 K5", SHARED, HipKernel.MLP_LAST, 1},
      {"V3 4p all-token K3", TYPED_FOUR, HipKernel.MLP_ALL, 0},
      {"V3 3p all-token K3", TYPED_THREE, HipKernel.MLP_ALL, 0},
      {"V3 4p K3", TYPED_FOUR, HipKernel.MLP_FIXED, 1},
      {"V3 4p context 512 K3", TYPED_FOUR_CONTEXT_512, HipKernel.MLP_FIXED, 1},
      {"V3 4p K5", TYPED_FOUR, HipKernel.MLP_LAST, 2},
      {"V3 3p K3", TYPED_THREE, HipKernel.MLP_FIXED, 1},
      {"V3 3p K5", TYPED_THREE, HipKernel.MLP_LAST, 2}
    };
  }

  /** x を書き戻す K3 の形。 */
  @DataProvider
  public Object[][] residualKernels() {
    return new Object[][] {
      {"v2 K3", SHARED, HipKernel.MLP_ALL, 0},
      {"V3 4p all-token K3", TYPED_FOUR, HipKernel.MLP_ALL, 0},
      {"V3 4p K3", TYPED_FOUR, HipKernel.MLP_FIXED, 1},
      {"V3 4p context 512 K3", TYPED_FOUR_CONTEXT_512, HipKernel.MLP_FIXED, 1},
      {"V3 3p K3", TYPED_THREE, HipKernel.MLP_FIXED, 1}
    };
  }

  /**
   * 出力が参照と合う（相対 RMS ≤ 1e-2、1 ulp 以内の割合 ≥ 0.99）。K3 では x と次の層の Attention の入力 n を比べ、x を書き戻さない
   * トークン（最後の全トークンの層と、局・席・牌の層のイベント）はビットで入力のままであることも確かめる。最後の全トークンの層の K3 では、
   * イベントのキーとバリューの組も比べる。K5 では、組ごとのゲインで最後に正規化した y と、Cross-Attention のキーとバリューを比べる。
   */
  @Test(groups = "rocm", dataProvider = "kernels")
  public void matchesReference(String name, HipArchitecture architecture, HipKernel kernel, int layer) {
    MlpCase c = MlpCase.random(1, architecture, kernel, layer, ROWS, EVENTS);
    MlpReference.Output expected = MlpReference.run(c);
    MlpReference.Output actual;
    try (HipPlan plan = HipPlan.create(library, stream, c.shape())) {
      actual = run(plan, c);
    }
    log.info(
        "PRISM-HIP-MLP {}: {} tokens, FFN segments {}",
        name,
        c.tokens(),
        c.group(0) == TokenGroup.ALL
            ? "[0, " + c.processedTokens() + ")"
            : "[0, " + c.contextTokens() + ") and [" + c.contextTokens() + ", " + c.fixedTokens() + ")");
    List<String> failures = new ArrayList<>();
    int h = architecture.hidden();
    if (kernel == HipKernel.MLP_LAST) {
      compare(failures, name, "y", expected.state(), actual.state());
      compare(failures, name, "KV", expected.memory(), actual.memory());
    } else {
      int written = c.writtenTokens() * h;
      compare(failures, name, "x", MlpReference.head(expected.x(), written), MlpReference.head(actual.x(), written));
      compare(failures, name, "n", expected.normalized(), actual.normalized());
      for (int j = 0; j < expected.events().length; j++) {
        compare(failures, name, "event keys and values " + j, expected.events()[j], actual.events()[j]);
      }
      assertEquals(
          Arrays.copyOfRange(actual.x(), written, actual.x().length),
          Arrays.copyOfRange(c.x, written, c.x.length),
          name + " tokens that K3 does not write back");
    }
    assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 残差の枝（O 射影と、全ての組の下りの射影）が 0 なら、x はビットで変わらない（x1 = ⌊x + 0⌉ = x、x = ⌊x1 + 0⌉）。 */
  @Test(groups = "rocm", dataProvider = "residualKernels")
  public void zeroBranchKeepsResidual(String name, HipArchitecture architecture, HipKernel kernel, int layer) {
    MlpCase c = MlpCase.random(2, architecture, kernel, layer, ROWS, EVENTS).withoutResidualBranch();
    try (HipPlan plan = HipPlan.create(library, stream, c.shape())) {
      assertEquals(run(plan, c).x(), c.x, name);
    }
  }

  /**
   * 前に別の行とイベントを足して、バッチの中の位置と区間の境が変わっても、同じトークンの出力はビットで同じになる。比べるのは、K3 では
   * x と n とイベントのキーとバリュー、K5 では y とキーとバリュー。
   */
  @Test(groups = "rocm", dataProvider = "kernels")
  public void independentOfBatchPosition(String name, HipArchitecture architecture, HipKernel kernel, int layer) {
    MlpCase alone = MlpCase.random(3, architecture, kernel, layer, ROWS, EVENTS);
    MlpCase other = MlpCase.random(4, architecture, kernel, layer, PREPENDED_ROWS, PREPENDED_EVENTS);
    MlpCase joined = alone.prepend(other);
    int h = architecture.hidden();
    MlpReference.Output single;
    MlpReference.Output shifted;
    try (HipPlan plan = HipPlan.create(library, stream, joined.shape())) {
      single = run(plan, alone);
      shifted = run(plan, joined);
    }
    if (kernel == HipKernel.MLP_LAST) {
      assertEquals(alone.unprepend(shifted.state(), h, alone.fixedTokens(), other), single.state(), name + " y");
      assertEquals(alone.unprependRows(shifted.memory(), other), single.memory(), name + " KV");
    } else {
      assertEquals(alone.unprepend(shifted.x(), h, alone.tokens(), other), single.x(), name + " x");
      assertEquals(
          alone.unprepend(shifted.normalized(), h, alone.normalizedTokens(), other), single.normalized(), name + " n");
      int width = 2 * architecture.attention();
      for (int j = 0; j < single.events().length; j++) {
        assertEquals(
            Arrays.copyOfRange(shifted.events()[j], other.events * width, (other.events + alone.events) * width),
            single.events()[j],
            name + " event keys and values " + j);
      }
    }
  }

  /** 入力の重みを GPU に写してカーネルを回し、カーネルが書いた領域を読み出す。写し元のバッファは、写し終わるまで回収させない。 */
  private MlpReference.Output run(HipPlan plan, MlpCase c) {
    ByteBuffer residual = c.input(HipRegion.RESIDUAL);
    ByteBuffer attention = c.input(HipRegion.ATTENTION);
    plan.uploadWeights(c.pack(), ++version);
    plan.copyIn(HipRegion.RESIDUAL, residual, c.bytes(HipRegion.RESIDUAL));
    plan.copyIn(HipRegion.ATTENTION, attention, c.bytes(HipRegion.ATTENTION));
    plan.run(c.kernel, c.layer, c.batch());
    plan.synchronize();
    Reference.reachabilityFence(residual);
    Reference.reachabilityFence(attention);
    int h = c.architecture.hidden();
    if (c.kernel == HipKernel.MLP_LAST) {
      return new MlpReference.Output(
          new short[0], new short[0], read(plan, c, HipRegion.STATE), read(plan, c, HipRegion.MEMORY), new short[0][]);
    }
    short[] normalized = MlpReference.head(read(plan, c, HipRegion.NORMALIZED), c.normalizedTokens() * h);
    return new MlpReference.Output(
        read(plan, c, HipRegion.RESIDUAL), normalized, new short[0], new short[0], events(plan, c));
  }

  /**
   * 最後の全トークンの層の K3 が書いた、イベントのキーとバリューの組 [組][イベント, 2A] を読む。局・席・牌の層の分は EVENT_KEY_VALUE
   * の領域に、2 段目の分は REREAD_MEMORY の領域の局・席・牌の後ろにある。
   */
  private static short[][] events(HipPlan plan, MlpCase c) {
    int sets = c.eventKeyValues().size();
    if (sets == 0) {
      return new short[0][];
    }
    int count = c.events * 2 * c.architecture.attention();
    short[][] events = new short[sets][];
    short[] layers = read(plan, c, HipRegion.EVENT_KEY_VALUE);
    for (int f = 0; f < c.fixedAttention.size(); f++) {
      events[f] = Arrays.copyOfRange(layers, f * count, (f + 1) * count);
    }
    if (c.reread != null) {
      short[] memory = read(plan, c, HipRegion.REREAD_MEMORY);
      events[sets - 1] = Arrays.copyOfRange(memory, memory.length - count, memory.length);
    }
    return events;
  }

  /** 領域のうち、このバッチの分を読み出す（BF16 のビット）。 */
  private static short[] read(HipPlan plan, MlpCase c, HipRegion region) {
    long bytes = c.bytes(region);
    ByteBuffer out = TestPack.direct(bytes);
    plan.copyOut(region, out, bytes);
    plan.synchronize();
    return MlpReference.shorts(out, Math.toIntExact(bytes / Short.BYTES));
  }

  private static void compare(List<String> failures, String name, String output, short[] expected, short[] actual) {
    MlpReference.Comparison comparison = MlpReference.Comparison.of(expected, actual);
    log.info(
        "PRISM-HIP-MLP {} {} vs reference: relative RMS {}, within 1 ulp {}, max ulp {}",
        name,
        output,
        String.format("%.3e", comparison.relativeRms()),
        String.format("%.5f", comparison.withinOneUlp()),
        comparison.maxUlp());
    if (!(comparison.relativeRms() <= RELATIVE_RMS) || !(comparison.withinOneUlp() >= WITHIN_ONE_ULP)) {
      failures.add(name + " " + output + " " + comparison);
    }
  }
}

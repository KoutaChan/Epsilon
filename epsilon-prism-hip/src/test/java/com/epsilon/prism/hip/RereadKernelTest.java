package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;

import com.epsilon.prism.hip.RereadCase.Row;

import java.lang.ref.Reference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * K9（候補層の 2 段目と B）を GPU 1 枚で動かし、CPU の参照と比べる。行ごとのイベントの数には 0・1・15〜17・上限を入れ、キーの数
 * （局・席・牌とイベント）が 16 の倍数の前後になる行を、四人打ちと三人打ちの両方の数で用意する。行ごとの遷移は 1〜40 個。
 */
@Test(groups = "rocm")
public class RereadKernelTest {

  private static final Logger log = LoggerFactory.getLogger(RereadKernelTest.class);

  /** z と B の相対 RMS の上限。 */
  private static final double LIMIT = 1e-2;

  /** 出力の枝 z − z1 の相対 RMS の上限。z1 に埋もれさせずに、Attention の結果が参照と合うかを見る。 */
  private static final double BRANCH_LIMIT = 2e-2;

  /** テストする形。 */
  enum Arm {
    /** 四人打ちの数（F 39）、G 256、ゲートの種類 6・手の扱いの種類 8。 */
    FOUR_PLAYERS(RereadCase.architecture(TestPack.FOUR_PLAYERS, 256, 6, 8)),
    /** 四人打ちの数、G 512。 */
    FOUR_PLAYERS_WIDE(RereadCase.architecture(TestPack.FOUR_PLAYERS, 512, 6, 8)),
    /** 三人打ちの数（F 31・キーのコード 107・イベントの上限 91）、G 512、ゲートの種類 7・手の扱いの種類 9。 */
    THREE_PLAYERS(RereadCase.architecture(new HipCounts(3, 27, 107, 23, 91), 512, 7, 9));

    final HipArchitecture architecture;

    Arm(HipArchitecture architecture) {
      this.architecture = architecture;
    }
  }

  private static final int ROW_CAPACITY = 32;
  private static final int EVENT_CAPACITY = 2048;
  private static final int TRANSITION_CAPACITY = 512;

  private final Map<Arm, HipPlan> plans = new EnumMap<>(Arm.class);
  private PtStream stream;
  private long version;

  @BeforeClass(groups = "rocm")
  public void openPlans() {
    HipLibrary library = HipLibrary.load();
    stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
    for (Arm arm : Arm.values()) {
      HipShape shape = RereadCase.shape(arm.architecture, ROW_CAPACITY, EVENT_CAPACITY, TRANSITION_CAPACITY);
      plans.put(arm, HipPlan.create(library, stream, shape));
    }
  }

  @AfterClass(groups = "rocm", alwaysRun = true)
  public void closePlans() {
    for (HipPlan plan : plans.values()) {
      plan.close();
    }
    plans.clear();
    if (stream != null) {
      stream.close();
    }
  }

  @DataProvider
  public Object[][] arms() {
    return Arrays.stream(Arm.values()).map(arm -> new Object[] {arm}).toArray(Object[][]::new);
  }

  /** z と B が参照と合う（相対 RMS ≤ 1e-2）。出力の枝 z − z1 も参照と合う（相対 RMS ≤ 2e-2）。 */
  @Test(groups = "rocm", dataProvider = "arms")
  public void matchesReference(Arm arm) {
    RereadCase c = RereadCase.random(11, arm.architecture, rows(arm.architecture.counts()));
    RereadReference.Result expected = RereadReference.run(c);
    RereadReference.Result actual = run(arm, c);
    MlpReference.Comparison z = MlpReference.Comparison.of(expected.transitions(), actual.transitions());
    double branch = relativeRms(branch(c, expected.transitions()), branch(c, actual.transitions()));
    double value = relativeRms(expected.values(), actual.values());
    log.info(
        "PRISM-HIP-K9 {} ({} rows, {} events, {} transitions): z relative RMS {}, within 1 ulp {}, max ulp {}; "
            + "branch z - z1 relative RMS {}; B relative RMS {}",
        arm,
        c.rows.size(),
        c.events,
        c.transitions,
        String.format(Locale.ROOT, "%.3e", z.relativeRms()),
        String.format(Locale.ROOT, "%.5f", z.withinOneUlp()),
        z.maxUlp(),
        String.format(Locale.ROOT, "%.3e", branch),
        String.format(Locale.ROOT, "%.3e", value));
    assertTrue(z.relativeRms() <= LIMIT, "z relative RMS " + z.relativeRms());
    assertTrue(branch <= BRANCH_LIMIT, "branch relative RMS " + branch);
    assertTrue(value <= LIMIT, "B relative RMS " + value);
  }

  /** 前に別の行を足してバッチの中の位置を変えても、同じ遷移の z と B はビットで同じになる。 */
  @Test(groups = "rocm", dataProvider = "arms")
  public void independentOfBatchPosition(Arm arm) {
    HipCounts counts = arm.architecture.counts();
    RereadCase alone = RereadCase.random(12, arm.architecture, rows(counts));
    List<Row> before = List.of(new Row(3, 5), new Row(0, 9), new Row(counts.maxEvents() / 2, 23));
    RereadCase shifted = alone.prepend(RereadCase.random(13, arm.architecture, before));
    RereadReference.Result expected = run(arm, alone);
    RereadReference.Result actual = run(arm, shifted);
    int offset = shifted.transitions - alone.transitions;
    int hidden = arm.architecture.hidden();
    assertEquals(
        Arrays.copyOfRange(actual.transitions(), offset * hidden, actual.transitions().length),
        expected.transitions(),
        "z of the shifted transitions");
    assertEquals(
        bits(Arrays.copyOfRange(actual.values(), offset, actual.values().length)),
        bits(expected.values()),
        "B of the shifted transitions");
  }

  /** 出力の射影 W_O と b_O が 0 なら、z は z1 とビットで同じになる。 */
  @Test(groups = "rocm", dataProvider = "arms")
  public void zeroOutputKeepsTheFirstStage(Arm arm) {
    RereadCase c = RereadCase.random(14, arm.architecture, rows(arm.architecture.counts())).withoutOutput();
    assertEquals(run(arm, c).transitions(), c.start);
  }

  /**
   * 行ごとのイベントと遷移の数を返す。キーの数（F + イベントの数）が 16 の倍数の前後になるイベントの数を、四人打ち（F 39）と三人打ち
   * （F 31）の両方で含む。
   *
   * @param counts 四人打ちか三人打ちかで決まる数
   * @return 行
   */
  static List<Row> rows(HipCounts counts) {
    int[] events = {0, 1, 15, 16, 17, counts.maxEvents(), 9, 10, 24, 25, 33, 41, 57, 2};
    int[] transitions = {1, 40, 3, 7, 13, 2, 21, 5, 11, 1, 17, 6, 9, 4};
    List<Row> rows = new ArrayList<>();
    for (int r = 0; r < events.length; r++) {
      rows.add(new Row(events[r], transitions[r]));
    }
    return rows;
  }

  private RereadReference.Result run(Arm arm, RereadCase c) {
    HipPlan plan = plans.get(arm);
    ByteBuffer pack = c.pack();
    ByteBuffer blob = c.blob();
    ByteBuffer start = c.input(HipRegion.TRANSITION);
    ByteBuffer memory = c.input(HipRegion.REREAD_MEMORY);
    plan.uploadWeights(pack, ++version);
    plan.copyIn(HipRegion.INPUT, blob, c.bytes(HipRegion.INPUT));
    plan.copyIn(HipRegion.TRANSITION, start, c.bytes(HipRegion.TRANSITION));
    plan.copyIn(HipRegion.REREAD_MEMORY, memory, c.bytes(HipRegion.REREAD_MEMORY));
    log.info("PRISM-HIP-K9 start {} {} transitions", arm, c.transitions);
    plan.run(HipKernel.REREAD, 0, c.batch());
    ByteBuffer z = TestPack.direct(c.bytes(HipRegion.TRANSITION));
    ByteBuffer b = TestPack.direct(c.bytes(HipRegion.VALUE));
    plan.copyOut(HipRegion.TRANSITION, z, c.bytes(HipRegion.TRANSITION));
    plan.copyOut(HipRegion.VALUE, b, c.bytes(HipRegion.VALUE));
    plan.synchronize();
    Reference.reachabilityFence(blob);
    Reference.reachabilityFence(start);
    Reference.reachabilityFence(memory);
    float[] values = new float[c.transitions];
    b.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(0).asFloatBuffer().get(values);
    return new RereadReference.Result(MlpReference.shorts(z, c.start.length), values);
  }

  /** 出力の枝 z − z1 を FP32 で返す。 */
  private static float[] branch(RereadCase c, short[] z) {
    float[] out = new float[z.length];
    for (int i = 0; i < z.length; i++) {
      out[i] = MlpReference.value(z[i]) - MlpReference.value(c.start[i]);
    }
    return out;
  }

  private static double relativeRms(float[] expected, float[] actual) {
    double error = 0;
    double norm = 0;
    for (int i = 0; i < expected.length; i++) {
      double d = (double) actual[i] - expected[i];
      error += d * d;
      norm += (double) expected[i] * expected[i];
    }
    return Math.sqrt(error / norm);
  }

  private static int[] bits(float[] values) {
    int[] out = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = Float.floatToRawIntBits(values[i]);
    }
    return out;
  }
}

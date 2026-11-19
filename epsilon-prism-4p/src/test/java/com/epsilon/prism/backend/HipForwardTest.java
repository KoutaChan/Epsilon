package com.epsilon.prism.backend;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipPlan;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputFormat.Role;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
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
    eachArm(
        failures,
        HipFixtures.Arm.selected(),
        arm -> {
          try (NDManager manager = NDManager.newBaseManager(Device.cpu());
              PtStream stream = GpuCase.newStream()) {
            PrismNetwork network = HipFixtures.network(manager, arm.shape(), 50 + arm.ordinal());
            ParameterStore store = new ParameterStore(manager, false);
            WeightLayout layout = WeightLayout.of(network);
            WeightPack pack = WeightPack.build(layout, network, 3);
            Scores scores = new Scores();
            try (HipBackend backend =
                HipBackend.open(
                    HipLibrary.load(), stream, layout, GpuCase.capacity(GpuCase.ROWS))) {
              backend.publish(pack);
              for (int b = 0; b < BATCHES; b++) {
                HostBatch batch =
                    HipFixtures.encode(
                        HipFixtures.mixedRows(GpuCase.ROWS, 600 + 100L * arm.ordinal() + 13L * b));
                float[] s = backend.forward(batch, 3);
                EagerStages.Stages eager = scores.add(s, network, store, manager, pack, batch);
                if (b == 0) {
                  trace(
                      backend.plan(),
                      eager,
                      arm.label() + " vs eager BF16",
                      layout.architecture().counts().fixed() * batch.plan().rows(),
                      batch.plan().count(Axis.TRANSITION));
                }
              }
              log.info(
                  "PRISM-HIP-T5 arm {} largest score differences vs eager BF16:{}",
                  arm.label(),
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
    eachArm(
        failures,
        HipFixtures.Arm.selected(),
        arm -> {
          try (NDManager manager = NDManager.newBaseManager(Device.cpu());
              PtStream stream = GpuCase.newStream()) {
            PrismNetwork network = HipFixtures.network(manager, arm.shape(), 90 + arm.ordinal());
            ParameterStore store = new ParameterStore(manager, false);
            WeightLayout layout = WeightLayout.of(network);
            WeightPack pack = WeightPack.build(layout, network, 6);
            HostBatch batch = HipFixtures.encode(rows);
            Scores scores = new Scores();
            try (HipBackend backend =
                HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(rows.size()))) {
              backend.publish(pack);
              scores.add(backend.forward(batch, 6), network, store, manager, pack, batch);
            }
            scores.check(failures, arm.label(), "long ");
            HipParity.Actions longOnly = subset(HipParity.Actions.of(batch), 2, rows.size());
            HipParity.Report report = HipParity.compare(scores.hip(), scores.bf16(), longOnly);
            log.info(
                "PRISM-HIP-T5 long-only arm {} HIP vs eager BF16 (CPU): {} flips: {}",
                arm.label(),
                report.summary(),
                flips(scores.hip(), scores.bf16(), longOnly));
            argmax(
                failures,
                "long-only " + arm.label(),
                scores.hip(),
                scores.bf16(),
                scores.fp32(),
                longOnly);
          }
        });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 同じ行の各合法手のビット列は、行の順序、直前のバッチの長さ、ストリームに依存しない。 */
  @Test(groups = "rocm", dataProvider = "scoreShapes", dataProviderClass = HipFixtures.class)
  public void rebatchingKeepsRowScoresBitwise(PrismShape shape) {
    List<long[]> rows = HipReferenceTest.mixedRows(GpuCase.ROWS);
    List<long[]> shuffled = new ArrayList<>(rows);
    Collections.shuffle(shuffled, new Random(9));
    List<long[]> longer = new ArrayList<>(HipFixtures.longRows(7));
    longer.addAll(shuffled);
    HostBatch first = HipFixtures.encode(rows);
    HostBatch second = HipFixtures.encode(shuffled);
    HostBatch longBatch = HipFixtures.encode(longer);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream left = GpuCase.newStream();
        PtStream right = GpuCase.newStream()) {
      PrismNetwork network = HipFixtures.network(manager, shape, 61);
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, 4);
      float[] eager =
          EagerStages.scores(
              network,
              new ParameterStore(manager, false),
              EagerStages.inputs(manager, first),
              Precision.BFLOAT16);
      float[] expected = null;
      for (PtStream stream : List.of(left, right)) {
        try (HipBackend backend =
            HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(longer.size()))) {
          backend.publish(pack);
          float[] original = backend.forward(first, 4);
          if (expected == null) {
            Diff diff = Diff.fp32(eager, original);
            Assert.assertTrue(diff.relativeRms() <= EAGER_LIMIT, diff.line(shape.toString()));
            expected = original;
          }
          Assert.assertTrue(HipParity.bitwiseEqual(original, expected), "stream scores differ");
          float[] reordered = backend.forward(second, 4);
          backend.forward(longBatch, 4);
          for (int r = 0; r < rows.size(); r++) {
            int firstAction = HipFixtures.firstAction(first, r);
            int moved = HipFixtures.firstAction(second, shuffled.indexOf(rows.get(r)));
            float[] alone = backend.forward(HipFixtures.encode(List.of(rows.get(r))), 4);
            Assert.assertEquals(alone.length, HipFixtures.actions(first, r));
            for (int k = 0; k < alone.length; k++) {
              int bits = Float.floatToRawIntBits(expected[firstAction + k]);
              Assert.assertEquals(
                  Float.floatToRawIntBits(reordered[moved + k]), bits, "shuffled row " + r);
              Assert.assertEquals(Float.floatToRawIntBits(alone[k]), bits, "single row " + r);
            }
          }
          Assert.assertTrue(
              HipParity.bitwiseEqual(backend.forward(first, 4), expected),
              "repeated scores differ");
        }
      }
    }
  }

  @DataProvider
  public Object[][] v4KernelArms() {
    return new Object[][] {
      {"V4_S", new PrismShape.V4()},
      {"V4_M", new PrismShape.V4(384)},
      {"V4_S_BIDIRECTIONAL", com.epsilon.prism.model.ModelShapes.bidirectional(256)},
      {"V4_M_BIDIRECTIONAL", com.epsilon.prism.model.ModelShapes.bidirectional(384)}
    };
  }

  @Test(groups = "rocm", dataProvider = "v4KernelArms")
  public void v4KernelScoresAndSpeed(String arm, PrismShape.V4 shape) throws IOException {
    List<long[]> rows = new ArrayList<>(HipFixtures.mixedRows(24, 800));
    rows.addAll(HipFixtures.longRows(2));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager eager = NDManager.newBaseManager(Device.gpu(0), "PyTorch");
        PtStream stream = GpuCase.newStream()) {
      PrismNetwork random = HipFixtures.network(manager, shape, 76);
      checkV4KernelScores(arm + "/random", random, eager, stream, rows, true);
      String checkpoints = System.getProperty("prism.v4.checkpoints");
      if (checkpoints != null) {
        Path checkpoint = Path.of(checkpoints, arm);
        PrismNetwork trained = PrismNetwork.load(checkpoint, manager);
        Assert.assertEquals(trained.shape(), shape);
        checkV4KernelScores(arm + "/trained", trained, eager, stream, rows, false);
      }
    }
  }

  @Test(groups = "rocm")
  public void v4DepthThreeScoresMatchEager() {
    List<long[]> rows = HipFixtures.longMixedRows();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager eager = NDManager.newBaseManager(Device.gpu(0), "PyTorch");
        PtStream stream = GpuCase.newStream()) {
      for (int hidden : new int[] {256, 384}) {
        var shape = new PrismShape.V4(hidden, 2 * hidden, 3, PrismShape.EventAttention.CAUSAL);
        checkV4KernelScores(
            "V4/" + hidden + "/depth3",
            HipFixtures.network(manager, shape, 77),
            eager,
            stream,
            rows,
            false);
      }
    }
  }

  @Test(groups = "rocm")
  public void v4CausalKeysKeepTheExistingEventPrefixBitwise() {
    var variant = FOUR_PLAYER.variant();
    var before = new com.epsilon.agent.observe.Observation(variant);
    var after = new com.epsilon.agent.observe.Observation(variant);
    var rows = HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 96);
    long[] prefix = null;
    long[] extended = null;
    for (int i = 0; i < rows.size() && prefix == null; i++) {
      before.bind(rows.get(i), 0);
      if (before.publicEvents() < 16 || before.publicEvents() + 39 > 64) {
        continue;
      }
      for (int j = i + 1; j < rows.size(); j++) {
        after.bind(rows.get(j), 0);
        if (before.kyoku() != after.kyoku() || before.honba() != after.honba()) {
          break;
        }
        if (after.observer() == before.observer() && after.publicEvents() + 39 > 64) {
          prefix = rows.get(i);
          extended = rows.get(j);
          break;
        }
      }
    }
    Assert.assertNotNull(
        prefix, "fixture needs two decisions from the same perspective in one round");
    for (ShapeArm arm : List.of(ShapeArm.V4_S, ShapeArm.V4_M)) {
      HostBatch shortRow = HipFixtures.encode(List.of(prefix), arm.shape());
      HostBatch longRow = HipFixtures.encode(List.of(extended), arm.shape());
      try (NDManager cpu = NDManager.newBaseManager(Device.cpu());
          PtStream stream = GpuCase.newStream()) {
        PrismNetwork network = HipFixtures.network(cpu, arm.shape(), 229);
        WeightLayout layout = WeightLayout.of(network);
        try (HipBackend hip =
            HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(1))) {
          hip.publish(WeightPack.build(layout, network, 1));
          int fixed = layout.architecture().counts().fixed();
          int tokens = fixed + shortRow.plan().events(0);
          int stride = network.shape().hidden() * Short.BYTES;
          ByteBuffer left = normalizedPrefix(hip, shortRow, tokens * stride);
          ByteBuffer right = normalizedPrefix(hip, longRow, tokens * stride);
          Assert.assertEquals(
              left.position(fixed * stride).mismatch(right.position(fixed * stride)),
              -1,
              arm + " normalized event states at the next attention layer");
        }
      }
    }
  }

  private static ByteBuffer normalizedPrefix(HipBackend backend, HostBatch row, int bytes) {
    backend.forward(row, 1);
    ByteBuffer out = ByteBuffer.allocateDirect(bytes);
    backend.plan().copyOut(HipRegion.NORMALIZED, out, bytes);
    backend.synchronize();
    return out;
  }

  private static void checkV4KernelScores(
      String label,
      PrismNetwork network,
      NDManager eager,
      PtStream stream,
      List<long[]> rows,
      boolean speed) {
    HostBatch host = HipFixtures.encode(rows, network.shape());
    WeightLayout layout = WeightLayout.of(network);
    int capacity = speed && Boolean.getBoolean("prism.v4.speed") ? SPEED_ROWS : rows.size();
    try (HipBackend backend =
        HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(capacity))) {
      backend.publish(WeightPack.build(layout, network, 2));
      float[] hip = backend.forward(host, 2);
      var identity = V4BitFixture.compare(label, "inference", hip);
      log.info(
          "PRISM-V4-BITS {} inference different={}/{}",
          label,
          identity.differing(),
          identity.elements());
      Assert.assertEquals(identity.differing(), 0L, label + " inference bits");
      try (NDManager work = eager.newSubManager()) {
        float[] reference =
            EagerStages.scores(
                network,
                new ParameterStore(work, false),
                EagerStages.inputs(work, host),
                Precision.BFLOAT16);
        Diff diff = Diff.fp32(reference, hip);
        HipParity.Report report = HipParity.compare(hip, reference, HipParity.Actions.of(host));
        log.info("PRISM-V4-PARITY {} {} {}", label, diff.line("scores"), report.summary());
        Assert.assertTrue(diff.relativeRms() <= EAGER_LIMIT, diff.line(label));
      }
      if (capacity == SPEED_ROWS) {
        HostBatch timed =
            HipFixtures.encode(HipFixtures.simulatedRows(SPEED_ROWS, 800), network.shape());
        measure(backend, timed.blob(), HipBackend.batchOf(timed), 2, label);
      }
    }
  }

  @DataProvider
  public Object[][] identityShapes() {
    return new Object[][] {{ShapeArm.V2, 70L}, {ShapeArm.V3, 72L}};
  }

  /** 保存した入力を再現し、同じ重みから得た各合法手の FP32 ビット列を読み戻す。 */
  @Test(groups = "rocm", dataProvider = "identityShapes")
  public void matchesSavedScoreBytes(ShapeArm arm, long seed) throws IOException {
    String base = System.getProperty("prism.hip.identity");
    if (base == null)
      throw new SkipException("set prism.hip.identity to the saved score fixture directory");
    Path fixture = Path.of(base, arm.name());
    HostBatch host = HipFixtures.encode(HipFixtures.simulatedRows(SPEED_ROWS, 800));
    ByteBuffer input = host.blob().duplicate().position(0).limit(host.bytes());
    Assert.assertEquals(
        input.mismatch(ByteBuffer.wrap(Files.readAllBytes(fixture.resolve("input.bin")))),
        -1,
        "saved input differs");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream = GpuCase.newStream()) {
      PrismNetwork network = HipFixtures.network(manager, arm.shape(), seed);
      WeightLayout layout = WeightLayout.of(network);
      try (HipBackend backend =
          HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(SPEED_ROWS))) {
        WeightPack pack = WeightPack.build(layout, network, 5);
        ByteBuffer archived = ByteBuffer.wrap(Files.readAllBytes(fixture.resolve("weights.bin")));
        ByteBuffer actual = pack.buffer();
        Assert.assertEquals(actual.remaining(), archived.remaining(), "saved weight pack length");
        Assert.assertEquals(actual.mismatch(archived), -1, "saved weight bytes differ");
        backend.publish(pack);
        float[] scores = backend.forward(host, 5);
        ByteBuffer saved =
            ByteBuffer.wrap(Files.readAllBytes(fixture.resolve("scores.bin")))
                .order(ByteOrder.LITTLE_ENDIAN);
        Assert.assertEquals(saved.remaining(), scores.length * Float.BYTES);
        for (int i = 0; i < scores.length; i++) {
          Assert.assertEquals(
              Float.floatToRawIntBits(scores[i]), saved.getInt(), "score at action " + i);
        }
        log.info(
            "Legacy identity arm={} actions={} bytes={}", arm, scores.length, saved.capacity());
      }
    }
  }

  public void validatesRowEventSpans() {
    HostBatch host = HipFixtures.encode(HipFixtures.longRows(6));
    HipBatch batch = HipBackend.batchOf(host);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream = GpuCase.newStream()) {
      PrismNetwork network = HipFixtures.network(manager, ShapeArm.V3.shape(), 72);
      WeightLayout layout = WeightLayout.of(network);
      try (HipBackend backend =
          HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(batch.rows()))) {
        backend.publish(WeightPack.build(layout, network, 1));
        float[] expected = backend.forward(host, 1);
        int row =
            Math.toIntExact(batch.offset(HipSection.ROW_COLUMN))
                + (batch.rows() - 1)
                    * layout.header().sectionWidth(HipSection.ROW_COLUMN)
                    * Integer.BYTES;
        int first = row + RowColumn.FIRST_EVENT.ordinal() * Integer.BYTES;
        int count = row + RowColumn.EVENTS.ordinal() * Integer.BYTES;
        ByteBuffer blob = host.blob();
        int savedFirst = blob.getInt(first);
        int savedCount = blob.getInt(count);
        int limit = layout.architecture().counts().maxEvents();
        Assert.assertTrue(batch.events() > limit);
        for (int[] span :
            new int[][] {
              {0, -1},
              {-1, 0},
              {0, limit + 1},
              {batch.events(), 1},
              {batch.events() + 1, 0},
              {Integer.MAX_VALUE, 2},
              {1, Integer.MAX_VALUE}
            }) {
          blob.putInt(first, span[0]).putInt(count, span[1]);
          Assert.expectThrows(IllegalArgumentException.class, () -> backend.forward(host, 1));
          backend.plan().copyIn(HipRegion.INPUT, blob, batch.bytes());
          Assert.expectThrows(
              IllegalArgumentException.class, () -> backend.plan().run(HipKernel.REREAD, 0, batch));
        }
        blob.putInt(first, batch.events()).putInt(count, 0);
        backend.forward(host, 1);
        backend.plan().run(HipKernel.REREAD, 0, batch);
        blob.putInt(first, savedFirst).putInt(count, savedCount);
        Assert.assertTrue(HipParity.bitwiseEqual(backend.forward(host, 1), expected));
      }
    }
  }

  /** 初期化したままのネットワークでは s が全て +0.0f。古いバージョンで forward すると例外になる（全てのアーム）。 */
  public void blankScoresZeroAndStaleVersionsFail() {
    List<String> failures = new ArrayList<>();
    eachArm(
        failures,
        HipFixtures.Arm.selected(),
        arm -> {
          try (NDManager manager = NDManager.newBaseManager(Device.cpu());
              PtStream stream = GpuCase.newStream()) {
            manager.getEngine().setRandomSeed(7);
            PrismNetwork network = new PrismNetwork(FOUR_PLAYER, arm.shape());
            network.initialize(manager);
            WeightLayout layout = WeightLayout.of(network);
            try (HipBackend backend =
                HipBackend.open(
                    HipLibrary.load(), stream, layout, GpuCase.capacity(GpuCase.ROWS))) {
              backend.publish(WeightPack.build(layout, network, 0));
              HostBatch batch = HipFixtures.encode(HipReferenceTest.mixedRows(GpuCase.ROWS));
              float[] s = backend.forward(batch, 0);
              for (int a = 0; a < s.length; a++) {
                Assert.assertEquals(
                    Float.floatToRawIntBits(s[a]),
                    0,
                    arm.label() + " blank score " + a + " = " + s[a]);
              }
              Assert.expectThrows(IllegalStateException.class, () -> backend.forward(batch, 1));
              log.info(
                  "PRISM-HIP-T5 blank arm {}: {} scores are +0.0f; stale version rejected",
                  arm.label(),
                  s.length);
            }
          }
        });
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 1 判断あたりの GPU の時間を測る（全てのアーム）。バッチは模擬の半荘の 2048 行で、アームごとに 1 行をログに書く。 */
  public void measureSpeed() {
    List<String> failures = new ArrayList<>();
    HostBatch batch = HipFixtures.encode(HipFixtures.simulatedRows(SPEED_ROWS, 800));
    eachArm(
        failures,
        HipFixtures.Arm.selected(),
        arm -> {
          try (NDManager manager = NDManager.newBaseManager(Device.cpu());
              PtStream stream = GpuCase.newStream()) {
            PrismNetwork network = HipFixtures.network(manager, arm.shape(), 70 + arm.ordinal());
            WeightLayout layout = WeightLayout.of(network);
            WeightPack pack = WeightPack.build(layout, network, 5);
            int rows = batch.plan().rows();
            try (HipBackend backend =
                HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(rows))) {
              backend.publish(pack);
              measure(
                  backend, batch.blob(), HipBackend.batchOf(batch), pack.version(), arm.label());
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
  private static void trace(
      HipPlan plan, EagerStages.Stages expected, String label, int fixedTokens, int transitions) {
    int hidden = expected.states().length / fixedTokens;
    log.info(
        "PRISM-HIP-T5 trace {}: {}; {}; {}; {}",
        label,
        Diff.bf16(expected.states(), read(plan, HipRegion.STATE, expected.states().length, true))
            .line("y"),
        Diff.bf16(expected.memory(), read(plan, HipRegion.MEMORY, expected.memory().length, true))
            .line("KV"),
        Diff.bf16(
                expected.transitions(),
                read(plan, HipRegion.TRANSITION, transitions * hidden, true))
            .line("z"),
        Diff.fp32(expected.values(), read(plan, HipRegion.VALUE, transitions, false)).line("B"));
  }

  private static float[] read(HipPlan plan, HipRegion region, int count, boolean bf16) {
    ByteBuffer buffer =
        ByteBuffer.allocateDirect(count * (bf16 ? Short.BYTES : Float.BYTES))
            .order(ByteOrder.LITTLE_ENDIAN);
    plan.copyOut(region, buffer, buffer.capacity());
    plan.synchronize();
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] =
          bf16
              ? Float.intBitsToFloat((buffer.getShort(i * Short.BYTES) & 0xFFFF) << 16)
              : buffer.getFloat(i * Float.BYTES);
    }
    return values;
  }

  /** 差の大きい手を k 個選び、行・手の種類・参照の s・HIP の s を並べて返す。 */
  private static String largest(float[] hip, float[] reference, HipParity.Actions actions, int k) {
    Integer[] order = new Integer[actions.roles().length];
    for (int a = 0; a < order.length; a++) {
      order[a] = a;
    }
    Arrays.sort(
        order,
        (x, y) ->
            Double.compare(
                Math.abs((double) hip[y] - reference[y]),
                Math.abs((double) hip[x] - reference[x])));
    int[] rowOf = new int[order.length];
    for (int r = 0; r < actions.rowFirst().length; r++) {
      for (int a = actions.rowFirst()[r]; a < actions.rowFirst()[r] + actions.rowCount()[r]; a++) {
        rowOf[a] = r;
      }
    }
    StringBuilder out = new StringBuilder();
    for (int i = 0; i < Math.min(k, order.length); i++) {
      int a = order[i];
      out.append(
          String.format(
              Locale.ROOT,
              " [row %d %s ref %.4f hip %.4f]",
              rowOf[a],
              actions.roles()[a].name(),
              reference[a],
              hip[a]));
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
  private static void measure(
      HipBackend backend, ByteBuffer blob, HipBatch batch, long version, String arm) {
    int decisions = batch.rows();
    ByteBuffer scores =
        ByteBuffer.allocateDirect(batch.actions() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
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
        continue; // この形では使わないカーネル
      }
      sum += median;
      line.append(
          String.format(
              Locale.ROOT, " %s=%.3fms(%.3fus)", kernel, median, median * 1e3 / decisions));
    }
    Arrays.sort(totals);
    int fixed = backend.layout().architecture().counts().fixed();
    log.info(
        "PRISM-HIP-SPEED arm {} rows {} tokens/row {} transitions/row {} actions/row {}:{}"
            + " kernels={}ms ({} us/decision) wall={}ms ({} us/decision)",
        arm,
        decisions,
        String.format(
            Locale.ROOT, "%.1f", ((double) fixed * decisions + batch.events()) / decisions),
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
        float[] s,
        PrismNetwork network,
        ParameterStore store,
        NDManager manager,
        WeightPack pack,
        HostBatch batch) {
      EagerStages.Stages half;
      EagerStages.Stages full;
      try (NDManager scope = manager.newSubManager()) {
        half =
            EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
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
      log.info(
          "PRISM-HIP-T5 {}arm {} HIP vs eager BF16 (CPU): {} passes={}",
          prefix,
          arm,
          half.summary(),
          half.passes());
      log.info(
          "PRISM-HIP-T5 {}arm {} HIP vs eager FP32 (CPU): {} passes={}",
          prefix,
          arm,
          full.summary(),
          full.passes());
      log.info("PRISM-HIP-T5 {}arm {} HIP vs CPU reference: {}", prefix, arm, cpu.summary());
      Diff half16 = Diff.fp32(floats(bf16), floats(hip));
      Diff full32 = Diff.fp32(floats(fp32), floats(hip));
      Diff mirror = Diff.fp32(floats(reference), floats(hip));
      log.info("PRISM-HIP-T5 {}arm {} {}", prefix, arm, half16.line("s vs eager BF16"));
      log.info("PRISM-HIP-T5 {}arm {} {}", prefix, arm, full32.line("s vs eager FP32"));
      log.info("PRISM-HIP-T5 {}arm {} {}", prefix, arm, mirror.line("s vs CPU reference"));
      String label = prefix + arm;
      log.info(
          "PRISM-HIP-T5 {}arm {} argmax flips vs eager BF16: {}",
          prefix,
          arm,
          flips(floats(hip), floats(bf16), joined));
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
      List<String> failures,
      String label,
      float[] hip,
      float[] bf16,
      float[] fp32,
      HipParity.Actions actions) {
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
        "PRISM-HIP-T5 {} argmax vs eager BF16 {} on all {} rows (line {} {}); vs eager FP32 {} on"
            + " {} rows that eager's own BF16 rounding does not reorder",
        label,
        String.format(Locale.ROOT, "%.5f", half.argmaxAgreement()),
        half.rows(),
        HipParity.ARGMAX_LIMIT,
        half.argmaxAgreement() >= HipParity.ARGMAX_LIMIT ? "met" : "missed",
        String.format(Locale.ROOT, "%.5f", rate),
        decided);
    if (rate < HipParity.ARGMAX_LIMIT) {
      failures.add(
          label
              + " argmax on rows that BF16 rounding does not reorder "
              + rate
              + " ("
              + decided
              + " rows)");
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
      out.append(
          String.format(
              Locale.ROOT,
              " [row %d gap %.3e max|ds| %.3e]",
              r,
              (double) reference[b] - reference[a],
              delta));
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
        Arrays.copyOfRange(actions.rowFirst(), from, to),
        Arrays.copyOfRange(actions.rowCount(), from, to),
        actions.roles());
  }

  private static final class Worst {
    double error = -1;
    int row;
    int action;
    float hip;
    float eager;

    void update(int[] rowIds, HipParity.Actions actions, float[] scores, float[] reference) {
      for (int r = 0; r < rowIds.length; r++) {
        for (int k = 0; k < actions.rowCount()[r]; k++) {
          int i = actions.rowFirst()[r] + k;
          double delta = Math.abs((double) scores[i] - reference[i]);
          if (delta > error) {
            error = delta;
            row = rowIds[r];
            action = k;
            hip = scores[i];
            eager = reference[i];
          }
        }
      }
    }
  }

  private static final int COHORT_BATCH = 32;

  @DataProvider
  public Object[][] cohortArms() {
    List<Object[]> cases = new ArrayList<>();
    for (ShapeArm arm : List.of(ShapeArm.V3, ShapeArm.V4_S, ShapeArm.V4_M)) {
      cases.add(new Object[] {arm, null});
    }
    String checkpoint = System.getProperty("prism.hip.checkpoint");
    if (checkpoint != null) cases.add(new Object[] {ShapeArm.V4_S, checkpoint});
    return cases.toArray(Object[][]::new);
  }

  /** コホート全体に等間隔に散らしたラベル行で、推論の入口から合法手のスコアまでを照合する。 */
  @Test(groups = "rocm", dataProvider = "cohortArms")
  public void cohortScoresMatchEager(ShapeArm arm, String checkpoint) throws IOException {
    String directory = System.getProperty("prism.hip.cohort");
    if (directory == null)
      throw new SkipException("set prism.hip.cohort to a saved cohort directory");
    int count = checkpoint == null ? Integer.getInteger("prism.hip.parityRows", 4096) : 20_000;
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream = GpuCase.newStream();
        CohortStore cohort = new CohortStore()) {
      cohort.addCohort(Path.of(directory));
      int iteration = cohort.cohorts().getFirst();
      int[] labels = cohort.select(iteration, iteration, index -> index.source().label());
      Assert.assertTrue(count > 0 && labels.length >= count);
      PrismNetwork network =
          checkpoint == null
              ? HipFixtures.network(manager, arm.shape(), 72)
              : PrismNetwork.load(Snapshot.model(Path.of(checkpoint)), manager);
      if (checkpoint != null)
        Assert.assertTrue(network.shape() instanceof PrismShape.V4, "checkpoint must contain V4");
      Worst worst = new Worst();
      ParameterStore store = new ParameterStore(manager, false);
      WeightLayout layout = WeightLayout.of(network);
      HipParity.Rounding result = new HipParity.Rounding(0, 0, 0, 0, 0, 0, 0);
      try (HipBackend backend =
          HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(COHORT_BATCH))) {
        backend.publish(WeightPack.build(layout, network, 1));
        for (int start = 0; start < count; start += COHORT_BATCH) {
          List<long[]> observations = new ArrayList<>();
          int[] rowIds = new int[Math.min(COHORT_BATCH, count - start)];
          for (int r = 0; r < rowIds.length; r++) {
            rowIds[r] = labels[(int) ((long) (start + r) * labels.length / count)];
            observations.add(cohort.read(rowIds[r]).observation());
          }
          HostBatch host = HipFixtures.encode(observations);
          try (NDManager scope = manager.newSubManager()) {
            NetworkInputs inputs = EagerStages.inputs(scope, host);
            float[] bf16 = EagerStages.scores(network, store, inputs, Precision.BFLOAT16);
            float[] fp32 = EagerStages.scores(network, store, inputs, Precision.FLOAT32);
            float[] hip = backend.forward(host, 1);
            var actions = HipParity.Actions.of(host);
            result =
                result.plus(
                    HipParity.rounding(
                        hip,
                        bf16,
                        fp32,
                        actions,
                        (row, eager, actual, gap, noise) ->
                            log.info(
                                "Stable disagreement arm={} row={} eagerAction={} hipAction={}"
                                    + " gap={} noise={}",
                                arm,
                                rowIds[row],
                                eager,
                                actual,
                                gap,
                                noise)));
            worst.update(rowIds, actions, hip, bf16);
          }
        }
      }
      log.info(
          "Worst score arm={} checkpoint={} row={} action={} hip={} eager={} abs={}",
          arm,
          checkpoint,
          worst.row,
          worst.action,
          worst.hip,
          worst.eager,
          worst.error);
      log.info(
          "Cohort parity arm={} rows={} actions={} relativeRms={} maxAbs={} stable={} agreed={}",
          arm,
          result.rows(),
          result.actions(),
          result.relativeRms(),
          result.maximum(),
          result.stable(),
          result.agreed());
      Assert.assertTrue(result.relativeRms() <= 0.04, result.toString());
      Assert.assertTrue(
          result.stable() > 0 && result.agreed() >= 0.99 * result.stable(), result.toString());
    }
  }
}

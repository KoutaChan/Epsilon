package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.observe.EventRecords.Prefix;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.EventCacheSettings;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFeature;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.kernel.CollectionCapabilities;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.ShapeArm;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 公開の順に決まるイベントのページと射影表のテスト。CPU のテストはページの配置と予約の決まりを確かめる。rocm のグループのテストは、製品の
 * 入口（{@link NativeLane}）でページを使う提出と入力の全量の提出が全ての合法手で同じビットになることを、両ルールの形で確かめる。
 */
public class EventCacheTest {
  private static final int ROWS = EventReplay.ROWS;
  private static final Tensor[] COLUMNS = {
    Tensor.EVENT_CATEGORY, Tensor.EVENT_SCALAR, Tensor.RELATION
  };

  /** 基点の JAR が入力の全量で求めた、両ルールの V4_S・V4_M の判断と全ての合法手のスコア。 */
  private static final String FROZEN_SCORES = "frozen-event-scores.gz";

  private static final long GENERATION = 3;

  @Test
  public void pageMapPreservesEventIdentityAcrossRoundsAndEvictions() {
    EventReplay replay = new EventReplay(ShapeArm.V4_S);
    var slots = replay.capacity.events();
    var resident = new EventPages.Resident(slots);
    int total = 0, misses = 0;
    byte[][] pool = new byte[COLUMNS.length][];
    for (long[] words : HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 420)) {
      replay.encode(words);
      HostBatch full = replay.full.input();
      var plan = full.plan();
      int events = plan.count(Axis.EVENT);
      total += events;
      int changed = place(replay, plan, resident);
      misses += changed;
      int sent = replay.pages.stage(full, changed);
      int payload = Integer.BYTES * (ROWS + events + changed);
      IntBuffer map = replay.pages.map().asIntBuffer();
      for (int c = 0; c < COLUMNS.length; c++) {
        Tensor tensor = COLUMNS[c];
        int width = replay.format.width(tensor) * (tensor == Tensor.EVENT_SCALAR ? 2 : 1);
        int offset =
            plan.offset(tensor)
                + (tensor == Tensor.RELATION ? width * plan.tokenStart(Axis.EVENT) : 0);
        if (pool[c] == null) pool[c] = new byte[width * slots];
        for (int m = 0; m < changed; m++) {
          int dense = map.get(ROWS + events + m);
          replay
              .pages
              .map()
              .get(payload + m * width, pool[c], map.get(ROWS + dense) * width, width);
        }
        payload += changed * width;
        for (int e = 0; e < events; e++) {
          int at = map.get(ROWS + e) * width;
          for (int k = 0; k < width; k++) {
            Assert.assertEquals(pool[c][at + k], full.blob().get(offset + e * width + k));
          }
        }
      }
      int basic = replay.format.fixedTokens();
      int code = plan.offset(Tensor.TRANSITION_CODE);
      for (int t = 0; t < plan.count(Axis.TRANSITION); t++) {
        Assert.assertEquals(
            replay.pages.map().slice(payload, basic).mismatch(full.blob().slice(code, basic)), -1);
        payload += basic;
        code += replay.format.candidateCodes().codeWidth();
      }
      Assert.assertEquals(sent, payload);
    }
    Assert.assertTrue(
        misses > slots && misses < total / 2, "expected both eviction and prefix reuse");
  }

  private static int place(EventReplay replay, BatchPlan plan, EventPages.Resident resident) {
    var reservation = replay.pages.reserve(plan, resident);
    Assert.assertNotNull(reservation);
    reservation.submit(() -> {});
    reservation.complete();
    return reservation.misses();
  }

  @Test
  public void siblingPlacementsCopyOnlyTheirSharedPrefix() {
    EventReplay replay = new EventReplay(ShapeArm.V4_S);
    var resident = new EventPages.Resident(replay.capacity.events());
    Prefix root = null;
    int prefix = 0;
    for (long[] words : HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 240)) {
      replay.encode(words);
      var plan = replay.full.plan();
      int length = plan.events(0);
      if (length <= prefix) continue;
      if (prefix == 0) {
        root = Prefix.append(null, length, 0);
        for (int row = 0; row < ROWS; row++) replay.pages.observe(row, root);
        Assert.assertEquals(place(replay, plan, resident), length);
        prefix = length;
        continue;
      }
      replay.pages.observe(0, Prefix.append(root, length, 0));
      replay.pages.observe(1, Prefix.append(root, length, 0));
      Assert.assertEquals(place(replay, plan, resident), 2 * (length - prefix));
      IntBuffer map = replay.pages.map().asIntBuffer();
      for (int e = 0; e < length; e++) {
        Assert.assertEquals(map.get(ROWS + e) == map.get(ROWS + length + e), e < prefix);
      }
      Assert.assertEquals(place(replay, plan, resident), 0);
      Assert.assertEquals(map.get(0), 0);
      Assert.assertEquals(map.get(1), 0);
      resident.clear();
      Assert.assertEquals(place(replay, plan, resident), 2 * length - prefix);
      return;
    }
    Assert.fail("Expected two increasing event counts");
  }

  @Test
  public void aLaneOnlyAcceptsItsOwnPlacementGeneration() {
    EventReplay replay = new EventReplay(ShapeArm.V4_S);
    var first = new EventPages.Resident(replay.capacity.events());
    var second = new EventPages.Resident(replay.capacity.events());
    for (long[] words : HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 100)) {
      replay.encode(words);
      var plan = replay.full.plan();
      int length = plan.events(0);
      if (length == 0) continue;
      Assert.assertEquals(place(replay, plan, first), length);
      Assert.assertEquals(place(replay, plan, second), length);
      Assert.assertEquals(place(replay, plan, first), 0);
      first.clear();
      Assert.assertEquals(place(replay, plan, first), length);
      Assert.assertEquals(place(replay, plan, second), 0);
      return;
    }
    Assert.fail("Expected at least one event");
  }

  @Test
  public void finishedPathsFreeTheirSlotsBeforeLivePaths() {
    EventReplay replay = new EventReplay(ShapeArm.V4_S);
    for (long[] words : HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 100)) {
      replay.encode(words);
      var plan = replay.full.plan();
      int length = plan.events(0);
      if (length == 0) continue;
      var resident = new EventPages.Resident(3 * length);
      Prefix live = Prefix.append(null, length, 0);
      Prefix ended = Prefix.append(null, length, 0);
      Prefix child = Prefix.append(ended, length + 1, 0);
      for (Prefix tip : List.of(live, ended)) {
        for (int row = 0; row < ROWS; row++) replay.pages.observe(row, tip);
        Assert.assertEquals(place(replay, plan, resident), length);
      }
      ended.release();
      Assert.assertEquals(ended.references(), ROWS + 1);
      child.release();
      Assert.assertEquals(ended.references(), ROWS);
      replay.pages.clear(ROWS);
      Assert.assertEquals(ended.references(), 0);
      Prefix next = Prefix.append(null, length, 0);
      for (int row = 0; row < ROWS; row++) replay.pages.observe(row, next);
      Assert.assertEquals(place(replay, plan, resident), length);
      next.release();
      replay.pages.clear(ROWS);
      for (int row = 0; row < ROWS; row++) replay.pages.observe(row, live);
      Assert.assertEquals(place(replay, plan, resident), 0);
      live.release();
      replay.pages.clear(ROWS);
      Assert.assertEquals(live.references(), 0);
      return;
    }
    Assert.fail("Expected at least one event");
  }

  @Test
  public void slotsStayReservedUntilTheirLastReaderCompletes() {
    EventReplay replay = new EventReplay(ShapeArm.V4_S);
    for (long[] words : HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 100)) {
      replay.encode(words);
      var plan = replay.full.plan();
      int length = plan.events(0);
      if (length == 0) continue;
      var resident = new EventPages.Resident(plan.count(Axis.EVENT));
      var reader = new EventPages(replay.format, replay.capacity, ByteBuffer::allocateDirect);
      var next = new EventPages(replay.format, replay.capacity, ByteBuffer::allocateDirect);
      for (int r = 0; r < ROWS; r++) {
        Prefix tip = Prefix.append(null, length, 0);
        reader.observe(r, tip);
        replay.pages.observe(r, tip);
        tip.release();
        next.observeSnapshot(r, length, 0);
      }
      var write = replay.pages.reserve(plan, resident);
      var read = reader.reserve(plan, resident);
      Assert.assertEquals(write.misses(), ROWS * length);
      Assert.assertEquals(read.misses(), 0);
      Assert.assertNull(next.reserve(plan, resident));
      Assert.expectThrows(IllegalStateException.class, read::complete);
      write.submit(() -> {});
      read.submit(() -> {});
      Assert.expectThrows(IllegalStateException.class, () -> reader.reserve(plan, resident));
      reader.clear(ROWS);
      replay.pages.clear(ROWS);
      write.complete();
      Assert.assertNull(next.reserve(plan, resident));
      read.complete();
      var available = next.reserve(plan, resident);
      Assert.assertEquals(available.misses(), ROWS * length);
      available.submit(() -> {});
      next.complete();
      next.clear(ROWS);
      return;
    }
    Assert.fail("Expected a nonempty history");
  }

  @Test
  public void incompleteAndFailedReservationsCannotSupplyHits() {
    EventReplay replay = new EventReplay(ShapeArm.V4_S);
    for (long[] words : HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 100)) {
      replay.encode(words);
      BatchPlan plan = replay.full.plan();
      int length = plan.events(0);
      if (length == 0) continue;
      var resident = new EventPages.Resident(plan.count(Axis.EVENT));
      Prefix valid = replay.history[replay.view.observer()];
      replay.pages.observe(1, null);
      Assert.expectThrows(IllegalStateException.class, () -> replay.pages.reserve(plan, resident));
      replay.pages.observe(1, valid);
      var first = replay.pages.reserve(plan, resident);
      Assert.assertEquals(first.misses(), length);
      Assert.expectThrows(IllegalStateException.class, () -> replay.pages.reserve(plan, resident));
      var following = new EventPages(replay.format, replay.capacity, ByteBuffer::allocateDirect);
      for (int r = 0; r < ROWS; r++) following.observe(r, valid);
      var second = following.reserve(plan, resident);
      Assert.assertEquals(second.misses(), 0);
      Assert.expectThrows(IllegalStateException.class, () -> second.submit(Assert::fail));
      Assert.expectThrows(
          IllegalArgumentException.class,
          () ->
              first.submit(
                  () -> {
                    throw new IllegalArgumentException("enqueue failed");
                  }));
      Assert.assertFalse(second.submit(Assert::fail));
      Assert.assertEquals(place(replay, plan, resident), length);
      Assert.assertEquals(place(replay, plan, resident), 0);
      var prepared = following.reserve(plan, resident);
      resident.clear();
      Assert.assertFalse(prepared.submit(Assert::fail));
      Assert.assertEquals(place(replay, plan, resident), length);
      following.clear(ROWS);
      return;
    }
    Assert.fail("Expected a nonempty history");
  }

  @DataProvider
  public static Object[][] causalShapes() {
    return java.util.Arrays.stream(ShapeArm.values())
        .filter(arm -> WeightLayout.supports(arm.shape()))
        .filter(arm -> arm.shape().inputFeatures().contains(InputFeature.PUBLIC_ORDER_EVENTS))
        .flatMap(
            arm ->
                java.util.Arrays.stream(Variant.values())
                    .filter(
                        variant -> {
                          var format = InputSchema.of(variant);
                          return CollectionCapabilities.supportsRule(
                              format.seats(), format.tiles());
                        })
                    .map(variant -> new Object[] {variant, arm}))
        .toArray(Object[][]::new);
  }

  /**
   * 同じ局の判断を順に流し、ページを使う提出のスコアが入力の全量の提出と全ての合法手で同じビットになる。最初の判断の後に次の世代を公開すると、
   * 前の世代で準備した提出は通らず、新しい世代でもビットは同じ。途中で別の重みを公開すると、前の重みで作った射影は使われない。
   */
  @Test(groups = "rocm", dataProvider = "causalShapes")
  public void scoresMatchAcrossReusedPrefixes(Variant variant, ShapeArm arm) throws Exception {
    var base = InputSchema.of(variant);
    var inputs = HipFixtures.observations(variant, 800, 1, TableWalk.Policy.CALLS, 240);
    EventReplay replay = replay(arm.shape().inputFormat(base), inputs);
    try (NDManager cpu = NDManager.newBaseManager(Device.cpu())) {
      String trained =
          variant == Variant.TENHOU_4P ? System.getProperty("prism.v4.checkpoints") : null;
      var network =
          trained == null
              ? PerturbedNetwork.of(cpu, base, arm.shape(), 229)
              : PrismNetwork.load(Path.of(trained, arm.name()), cpu);
      try (var lanes = new Lanes(network, Device.gpu(0), replay)) {
        int count = 0, actions = 0;
        boolean changedWeights = false;
        for (long[] words : inputs) {
          replay.encode(words);
          actions += replay.full.plan().count(Axis.ACTION);
          float[] expected = lanes.full(replay);
          Assert.assertEquals(expected.length, replay.full.plan().count(Axis.ACTION));
          Assert.assertTrue(HipParity.bitwiseEqual(expected, lanes.paged(replay)), arm + " score");
          count += expected.length;
          if (count == expected.length) {
            var stale =
                lanes.paged.prepare(
                    replay.full.input(), NativeLane.scores(replay.full.input()), null);
            lanes.publish(network, GENERATION + 1);
            Assert.expectThrows(IllegalStateException.class, stale::run);
            Assert.assertTrue(HipParity.bitwiseEqual(expected, lanes.paged(replay)), "republish");
          } else if (!changedWeights
              && count > 300
              && replay.full.plan().count(Axis.ACTION) > ROWS) {
            // 入力を変えずに別の重みを公開し、前の重みで計算した埋め込みや射影が残らないことを確かめる。
            var changed = PerturbedNetwork.of(cpu, base, network.shape(), 230);
            lanes.publish(changed, GENERATION + 2);
            float[] fresh = lanes.full(replay);
            Assert.assertFalse(
                HipParity.bitwiseEqual(expected, fresh),
                "changed weights must change at least one score");
            changedWeights = true;
            Assert.assertTrue(
                HipParity.bitwiseEqual(fresh, lanes.paged(replay)), "changed weights");
          }
        }
        Assert.assertTrue(changedWeights);
        Assert.assertEquals(lanes.paged.generation(), GENERATION + 2);
        Assert.assertEquals(count, actions);
      }
    }
  }

  /**
   * 基点の JAR が保存した両ルールの V4_S・V4_M の判断を流し、入力の全量の提出とページを使う提出の両方で、全ての合法手のコードと
   * スコアのビットが保存した値と同じになる。
   */
  @Test(groups = "rocm")
  public void allLegalScoresMatchFrozenObservations() throws Exception {
    InputStream resource = EventCacheTest.class.getResourceAsStream(FROZEN_SCORES);
    Assert.assertNotNull(resource, FROZEN_SCORES);
    try (var in = new DataInputStream(new BufferedInputStream(new GZIPInputStream(resource)))) {
      for (Variant variant : Variant.values())
        for (ShapeArm arm : List.of(ShapeArm.V4_S, ShapeArm.V4_M)) frozenScores(variant, arm, in);
      Assert.assertEquals(in.read(), -1);
    }
  }

  /** 1 つのルールと形の保存した判断を読み、全量とページの両方のスコアを保存した値と照らす。 */
  private static void frozenScores(Variant variant, ShapeArm arm, DataInputStream in)
      throws IOException {
    var base = InputSchema.of(variant);
    var inputs = HipFixtures.observations(variant, 800, 1, TableWalk.Policy.CALLS, 240);
    Assert.assertEquals(in.readUTF(), variant.name() + "/" + arm.name());
    Assert.assertEquals(in.readInt(), inputs.size());
    var replay = replay(arm.shape().inputFormat(base), inputs);
    try (var cpu = NDManager.newBaseManager(Device.cpu());
        var lanes =
            new Lanes(PerturbedNetwork.of(cpu, base, arm.shape(), 229), Device.gpu(0), replay)) {
      for (var words : inputs) {
        Assert.assertEquals(in.readInt(), words.length);
        long[] saved = new long[words.length];
        for (int i = 0; i < saved.length; i++) saved[i] = in.readLong();
        Assert.assertEquals(saved, words);
        replay.encode(saved);
        int count = replay.full.plan().count(Axis.ACTION);
        var full = lanes.full(replay);
        var paged = lanes.paged(replay);
        Assert.assertEquals(full.length, count);
        Assert.assertEquals(paged.length, count);
        Assert.assertEquals(in.readInt(), count);
        for (int r = 0; r < ROWS; r++) {
          for (int j = 0; j < replay.full.plan().actions(r); j++) {
            int i = replay.full.plan().firstAction(r) + j;
            checkAction(
                in,
                replay.view.action(j).code(),
                Float.floatToRawIntBits(full[i]),
                Float.floatToRawIntBits(paged[i]));
          }
        }
      }
    }
  }

  /** 保存した手のコードとスコアのビットを読み、全量とページの両方のスコアがそれと同じことを確かめる。 */
  static void checkAction(DataInputStream in, int code, int full, int paged) throws IOException {
    Assert.assertEquals(in.readInt(), code);
    int expected = in.readInt();
    Assert.assertEquals(full, expected);
    Assert.assertEquals(paged, expected);
  }

  @Test
  public void scoreFixtureRejectsReorderingLostDuplicatesAndBitChanges() throws IOException {
    var buffer = ByteBuffer.allocate(3 * 2 * Integer.BYTES);
    for (int code : new int[] {10, 5, 10}) buffer.putInt(code).putInt(0x80000000);
    checkScores(buffer.array(), new int[] {10, 5, 10}, 0x80000000);
    Assert.expectThrows(
        AssertionError.class, () -> checkScores(buffer.array(), new int[] {5, 10, 10}, 0x80000000));
    Assert.expectThrows(
        AssertionError.class, () -> checkScores(buffer.array(), new int[] {10, 5}, 0x80000000));
    Assert.expectThrows(
        AssertionError.class, () -> checkScores(buffer.array(), new int[] {10, 5, 10}, 0x80000001));
    Assert.expectThrows(
        java.io.EOFException.class,
        () ->
            checkScores(
                java.util.Arrays.copyOf(buffer.array(), buffer.capacity() - 1),
                new int[] {10, 5, 10},
                0x80000000));
  }

  private static void checkScores(byte[] fixture, int[] codes, int bits) throws IOException {
    try (var in = new DataInputStream(new ByteArrayInputStream(fixture))) {
      for (int code : codes) checkAction(in, code, 0x80000000, bits);
      Assert.assertEquals(in.read(), -1);
    }
  }

  @DataProvider
  public static Object[][] projectionShapes() {
    return java.util.Arrays.stream(causalShapes())
        .filter(
            row ->
                CollectionCapabilities.projectionRowLimit(((ShapeArm) row[1]).shape().hidden()) > 0)
        .toArray(Object[][]::new);
  }

  private static long[][] boundaryRows(EventReplay replay) {
    long[][] rows = new long[3][];
    for (long[] words : boundaryInputs(replay.format.variant())) {
      var view = replay.view.bind(words, 0);
      int end = replay.format.fixedTokens() + view.events() + view.kanDoraIndicators();
      int index = (end + 1) % 16;
      if (index < rows.length && rows[index] == null && view.legalActions() > 1)
        rows[index] = words;
      if (rows[0] != null && rows[1] != null && rows[2] != null) return rows;
    }
    throw new AssertionError("Missing mixed-length inputs around an attention tile boundary");
  }

  @Test(dataProvider = "causalShapes")
  public void observationsCoverAttentionTileEdges(Variant variant, ShapeArm arm) {
    var replay = new EventReplay(arm.shape().inputFormat(InputSchema.of(variant)));
    var rows = boundaryRows(replay);
    for (int i = 0; i < rows.length; i++) {
      replay.encodePair(rows[i], rows[i], null, null);
      Assert.assertEquals((1 + replay.format.fixedTokens() + replay.full.plan().events(0)) % 16, i);
    }
  }

  @Test(groups = "rocm", dataProvider = "causalShapes")
  public void regroupedTileEdgesKeepEveryLegalScore(Variant variant, ShapeArm arm)
      throws Exception {
    var base = InputSchema.of(variant);
    var replay = new EventReplay(arm.shape().inputFormat(base));
    var rows = boundaryRows(replay);
    Prefix[] tips = new Prefix[rows.length];
    replay.other(longest(variant, boundaryInputs(variant)));
    try (var cpu = NDManager.newBaseManager(Device.cpu());
        var lanes =
            new Lanes(PerturbedNetwork.of(cpu, base, arm.shape(), 229), Device.gpu(0), replay)) {
      float[][] expected = new float[rows.length][];
      for (int i = 0; i < rows.length; i++) {
        var view = replay.view.bind(rows[i], 0);
        tips[i] = Prefix.append(null, view.events(), view.kanDoraIndicators());
        replay.encodePair(rows[i], rows[i], tips[i], tips[i]);
        expected[i] = java.util.Arrays.copyOf(lanes.full(replay), view.legalActions());
      }
      for (int left = 0; left < rows.length; left++)
        for (int right = 0; right < rows.length; right++) {
          replay.encodePair(rows[left], rows[right], tips[left], tips[right]);
          var full = lanes.full(replay);
          var paged = lanes.paged(replay);
          Assert.assertTrue(HipParity.bitwiseEqual(full, paged));
          Assert.assertEquals(paged.length, expected[left].length + expected[right].length);
          int at = 0;
          for (int index : new int[] {left, right})
            for (float score : expected[index]) {
              Assert.assertEquals(
                  Float.floatToRawIntBits(paged[at++]), Float.floatToRawIntBits(score));
            }
        }
    } finally {
      replay.pages.clear(ROWS);
      for (var p : tips) if (p != null) p.release();
    }
  }

  private static int[] projectionCapacities(ShapeArm arm) {
    int limit = CollectionCapabilities.projectionRowLimit(arm.shape().hidden());
    int boundary = Integer.highestOneBit(limit) / 2;
    return new int[] {boundary - 1, boundary, limit};
  }

  /**
   * 射影表の容量の境界の前後と上限（能力表の上限から決める）で、入力の違うイベントを続けて流し、同じ入力をもう一度読み、重みを公開し直して
   * からも読む。どの提出でも、ページを使う提出のスコアは入力の全量の提出と全ての合法手で同じビットになる。
   */
  @Test(groups = "rocm", dataProvider = "projectionShapes")
  public void projectionTableKeepsScoresAcrossRepeatedReads(Variant variant, ShapeArm arm)
      throws Exception {
    var schema = InputSchema.of(variant);
    EventReplay replay = projectionReplay(variant, arm);
    try (NDManager cpu = NDManager.newBaseManager(Device.cpu())) {
      var network = PerturbedNetwork.of(cpu, schema, arm.shape(), 739);
      for (int slots : projectionCapacities(arm)) {
        var pages = new EventCacheSettings(EventCacheSettings.Mode.CAUSAL, slots, 0);
        try (var lanes = new Lanes(network, Device.gpu(0), replay, pages)) {
          Assert.assertEquals(lanes.slots, slots);
          short[] keys = {0x2a11, 0x3b25, 0x2c37};
          for (short key : keys) projectionPass(lanes, replay, key);
          for (short key : keys) projectionPass(lanes, replay, key);
          lanes.publish(network, GENERATION + 1);
          for (short key : keys) projectionPass(lanes, replay, key);
        }
      }
    }
  }

  private static EventReplay projectionReplay(Variant variant, ShapeArm arm) {
    var inputs = HipFixtures.observations(variant, 800, 1, TableWalk.Policy.CALLS, 100);
    EventReplay replay = replay(arm.shape().inputFormat(InputSchema.of(variant)), inputs);
    for (long[] words : inputs) {
      replay.encode(words);
      if (replay.full.plan().events(0) > 0 && replay.full.plan().actions(0) > 1) break;
    }
    Assert.assertTrue(replay.full.plan().events(0) > 0 && replay.full.plan().actions(0) > 1);
    return replay;
  }

  /** 各行の最初のイベントの値を書き換え、ページの履歴を新しくしてから、全量とページの両方のスコアを比べる。 */
  private static void projectionPass(Lanes lanes, EventReplay replay, short scalar) {
    var full = replay.full.input();
    var plan = full.plan();
    int scalarWidth = Short.BYTES * replay.format.width(Tensor.EVENT_SCALAR);
    for (int r = 0; r < ROWS; r++) {
      full.blob()
          .putShort(plan.offset(Tensor.EVENT_SCALAR) + plan.firstEvent(r) * scalarWidth, scalar);
      replay.pages.observeSnapshot(r, plan.events(r), 0);
    }
    float[] expected = lanes.full(replay);
    Assert.assertEquals(expected.length, plan.count(Axis.ACTION));
    Assert.assertTrue(HipParity.bitwiseEqual(expected, lanes.paged(replay)));
  }

  @Test(groups = "rocm", dataProvider = "causalShapes")
  public void cacheCapacityControlsResidencyWithoutChangingScores(Variant variant, ShapeArm arm)
      throws Exception {
    var base = InputSchema.of(variant);
    var format = arm.shape().inputFormat(base);
    var inputs = HipFixtures.observations(variant, 800, 1, TableWalk.Policy.CALLS, 100);
    var view = new Observation(variant);
    var counts =
        inputs.stream()
            .mapToInt(words -> ROWS * view.bind(words, 0).publicEvents())
            .filter(events -> events > 0)
            .summaryStatistics();
    Assert.assertTrue(counts.getMin() < counts.getMax());
    int pageSlots = counts.getMin() + (counts.getMax() - counts.getMin()) / 2;
    try (NDManager cpu = NDManager.newBaseManager(Device.cpu())) {
      var network = PerturbedNetwork.of(cpu, base, arm.shape(), 229);
      for (var settings :
          List.of(
              new EventCacheSettings(EventCacheSettings.Mode.OFF, pageSlots, 0.0),
              new EventCacheSettings(EventCacheSettings.Mode.CAUSAL, 0, 1.0),
              new EventCacheSettings(EventCacheSettings.Mode.CAUSAL, pageSlots, 1.0))) {
        int slots = settings.mode() == EventCacheSettings.Mode.OFF ? 0 : settings.eventsPerLane();
        EventReplay replay = replay(format, inputs);
        try (var lanes = new Lanes(network, Device.gpu(0), replay, settings)) {
          Assert.assertEquals(lanes.slots, slots);
          boolean fits = false, exceeds = false;
          for (long[] words : inputs) {
            replay.encode(words);
            int events = replay.full.plan().count(Axis.EVENT);
            fits |= events > 0 && events <= slots;
            exceeds |= events > slots;
            replay.full.input().blob().position(1);
            float[] expected = lanes.full(replay);
            Assert.assertTrue(HipParity.bitwiseEqual(expected, lanes.paged(replay)));
            if (exceeds && (slots == 0 || fits)) break;
          }
          Assert.assertTrue(exceeds && (slots == 0 || fits));
        }
      }
    }
  }

  @Test(groups = "rocm", dataProvider = "projectionShapes")
  public void pageAndProjectionGenerationsBelongToTheirDevice(Variant variant, ShapeArm arm)
      throws Exception {
    if (Engine.getInstance().getGpuCount() < 2) throw new SkipException("two GPUs are required");
    var base = InputSchema.of(variant);
    EventReplay replay = projectionReplay(variant, arm);
    try (var cpu = NDManager.newBaseManager(Device.cpu())) {
      var network = PerturbedNetwork.of(cpu, base, arm.shape(), 229);
      try (var gpu1 = new Lanes(network, Device.gpu(1), replay);
          var gpu0 = new Lanes(network, Device.gpu(0), replay);
          var default0 =
              new Lanes(
                  NativeLane.onDefaultStream(network, Device.gpu(0), replay.capacity, GENERATION),
                  NativeLane.onDefaultStream(network, Device.gpu(0), replay.capacity, GENERATION),
                  pages(replay))) {
        float[] expected = gpu1.full(replay);
        Assert.assertTrue(HipParity.bitwiseEqual(expected, gpu1.paged(replay)));
        Assert.assertTrue(HipParity.bitwiseEqual(expected, gpu0.paged(replay)));
        Assert.assertTrue(HipParity.bitwiseEqual(expected, default0.paged(replay)));
        var changed = PerturbedNetwork.of(cpu, base, arm.shape(), 230);
        gpu0.publish(changed, GENERATION + 1);
        float[] other = gpu0.full(replay);
        Assert.assertFalse(HipParity.bitwiseEqual(expected, other));
        Assert.assertTrue(HipParity.bitwiseEqual(other, gpu0.paged(replay)));
        Assert.assertTrue(HipParity.bitwiseEqual(expected, default0.paged(replay)));
        Assert.assertTrue(HipParity.bitwiseEqual(expected, gpu1.paged(replay)));
      }
    }
  }

  /** 2 局ぶんの判断。Attention のタイルの境界をまたぐ長さの判断を探すときに使う。 */
  private static List<long[]> boundaryInputs(Variant variant) {
    return HipFixtures.observations(variant, 800, 2, TableWalk.Policy.CALLS, 800);
  }

  /** 判断の中で公開のイベントがいちばん多い判断を返す。 */
  private static long[] longest(Variant variant, List<long[]> inputs) {
    var view = new Observation(variant);
    long[] longest = null;
    int events = -1;
    for (long[] words : inputs) {
      int count = view.bind(words, 0).publicEvents();
      if (count > events) {
        events = count;
        longest = words;
      }
    }
    return longest;
  }

  /** 判断を流す再生を作る。ページを使う提出の前に全量で流すバッチは、判断の中でいちばん長い判断から作る。 */
  private static EventReplay replay(InputFormat format, List<long[]> inputs) {
    var replay = new EventReplay(format);
    replay.other(longest(format.variant(), inputs));
    return replay;
  }

  /** バッチの全てのイベントが入るページ。 */
  private static EventCacheSettings pages(EventReplay replay) {
    return new EventCacheSettings(EventCacheSettings.Mode.CAUSAL, replay.capacity.events(), 0);
  }

  /**
   * 同じ重みを公開した 2 本のレーン。入力の全量の提出は一方に、ページを使う提出はもう一方に流す。ページを使う提出の前には、別の長い判断の
   * バッチを全量で同じレーンに流しておく（前の提出が残した入力を読んでしまうと、スコアが変わって分かる）。
   */
  private static final class Lanes implements AutoCloseable {
    final NativeLane full;
    final NativeLane paged;

    /** ページを使うレーンが確保できたスロットの数。 */
    final int slots;

    Lanes(PrismNetwork network, Device device, EventReplay replay) {
      this(network, device, replay, pages(replay));
    }

    Lanes(PrismNetwork network, Device device, EventReplay replay, EventCacheSettings pages) {
      this(
          NativeLane.open(network, device, replay.capacity, GENERATION),
          NativeLane.open(network, device, replay.capacity, GENERATION),
          pages);
    }

    Lanes(NativeLane full, NativeLane paged, EventCacheSettings pages) {
      this.full = full;
      this.paged = paged;
      slots = paged.configurePages(pages);
    }

    void publish(PrismNetwork network, long generation) {
      full.publish(network, generation);
      paged.publish(network, generation);
    }

    float[] full(EventReplay replay) {
      return full.forward(replay.full.input());
    }

    float[] paged(EventReplay replay) {
      paged.forward(replay.other().input());
      return paged.forward(replay.full.input(), replay.pages);
    }

    @Override
    public void close() {
      try {
        full.close();
      } finally {
        paged.close();
      }
    }
  }
}

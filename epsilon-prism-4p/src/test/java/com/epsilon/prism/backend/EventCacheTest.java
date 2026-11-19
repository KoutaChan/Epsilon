package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.observe.EventRecords.Prefix;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.EventCacheSettings;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.ShapeArm;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class EventCacheTest {
  private static final int ROWS = 2;
  private static final Tensor[] COLUMNS = {
    Tensor.EVENT_CATEGORY, Tensor.EVENT_SCALAR, Tensor.RELATION
  };

  private static final class Replay {
    final InputFormat format;
    final BatchPlan.Capacity capacity = GpuCase.capacity(ROWS);
    final Prefix[] history;
    final HostInputs full;
    final EventPages pages;
    final InputFormat.RowWriter writer;
    final Observation view;
    int kyoku = -1, honba = -1;

    Replay(ShapeArm arm) {
      this(arm.shape().inputFormat(InputSchema.FOUR_PLAYER));
    }

    Replay(InputFormat format) {
      this.format = format;
      history = new Prefix[format.seats()];
      full = HostInputs.allocate(format, capacity, ByteBuffer::allocateDirect);
      pages = new EventPages(format, capacity, ByteBuffer::allocateDirect);
      writer = format.writer();
      view = new Observation(format.variant());
    }

    void encode(long[] words) {
      view.bind(words, 0);
      if (kyoku != view.kyoku() || honba != view.honba()) {
        for (Prefix previous : history) if (previous != null) previous.release();
        java.util.Arrays.fill(history, null);
      }
      kyoku = view.kyoku();
      honba = view.honba();
      int seat = view.observer();
      full.plan().clear();
      for (int r = 0; r < ROWS; r++) Assert.assertEquals(full.plan().add(view), r);
      full.plan().seal();
      // 2 行目は親がこのバッチで追記したイベントも共有する。どちらの行もまだ GPU に送っていない。
      Prefix previous = history[seat];
      history[seat] = Prefix.append(previous, view.events(), view.kanDoraIndicators());
      if (previous != null && previous != history[seat]) previous.release();
      for (int r = 0; r < ROWS; r++) {
        writer.write(view, full, r);
        pages.observe(r, history[seat]);
      }
    }
  }

  @Test
  public void pageMapPreservesEventIdentityAcrossRoundsAndEvictions() {
    Replay replay = new Replay(ShapeArm.V4_S);
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
      int changed = replay.pages.pack(plan, resident);
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

  @Test
  public void siblingPlacementsCopyOnlyTheirSharedPrefix() {
    Replay replay = new Replay(ShapeArm.V4_S);
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
        Assert.assertEquals(replay.pages.pack(plan, resident), length);
        prefix = length;
        continue;
      }
      replay.pages.observe(0, Prefix.append(root, length, 0));
      replay.pages.observe(1, Prefix.append(root, length, 0));
      Assert.assertEquals(replay.pages.pack(plan, resident), 2 * (length - prefix));
      IntBuffer map = replay.pages.map().asIntBuffer();
      for (int e = 0; e < length; e++) {
        Assert.assertEquals(map.get(ROWS + e) == map.get(ROWS + length + e), e < prefix);
      }
      Assert.assertEquals(replay.pages.pack(plan, resident), 0);
      Assert.assertEquals(map.get(0), 0);
      Assert.assertEquals(map.get(1), 0);
      resident.clear();
      Assert.assertEquals(replay.pages.pack(plan, resident), 2 * length - prefix);
      return;
    }
    Assert.fail("Expected two increasing event counts");
  }

  @Test
  public void aLaneOnlyAcceptsItsOwnPlacementGeneration() {
    Replay replay = new Replay(ShapeArm.V4_S);
    var first = new EventPages.Resident(replay.capacity.events());
    var second = new EventPages.Resident(replay.capacity.events());
    for (long[] words : HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 100)) {
      replay.encode(words);
      var plan = replay.full.plan();
      int length = plan.events(0);
      if (length == 0) continue;
      Assert.assertEquals(replay.pages.pack(plan, first), length);
      Assert.assertEquals(replay.pages.pack(plan, second), length);
      Assert.assertEquals(replay.pages.pack(plan, first), 0);
      first.clear();
      Assert.assertEquals(replay.pages.pack(plan, first), length);
      Assert.assertEquals(replay.pages.pack(plan, second), 0);
      return;
    }
    Assert.fail("Expected at least one event");
  }

  @Test
  public void finishedPathsFreeTheirSlotsBeforeLivePaths() {
    Replay replay = new Replay(ShapeArm.V4_S);
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
        Assert.assertEquals(replay.pages.pack(plan, resident), length);
      }
      ended.release();
      Assert.assertEquals(ended.references(), ROWS + 1);
      child.release();
      Assert.assertEquals(ended.references(), ROWS);
      replay.pages.clear(ROWS);
      Assert.assertEquals(ended.references(), 0);
      Prefix next = Prefix.append(null, length, 0);
      for (int row = 0; row < ROWS; row++) replay.pages.observe(row, next);
      Assert.assertEquals(replay.pages.pack(plan, resident), length);
      next.release();
      replay.pages.clear(ROWS);
      for (int row = 0; row < ROWS; row++) replay.pages.observe(row, live);
      Assert.assertEquals(replay.pages.pack(plan, resident), 0);
      live.release();
      replay.pages.clear(ROWS);
      Assert.assertEquals(live.references(), 0);
      return;
    }
    Assert.fail("Expected at least one event");
  }

  @DataProvider
  public static Object[][] causalShapes() {
    return java.util.Arrays.stream(Variant.values())
        .flatMap(
            variant ->
                java.util.stream.Stream.of(ShapeArm.V4_S, ShapeArm.V4_M)
                    .map(arm -> new Object[] {variant, arm}))
        .toArray(Object[][]::new);
  }

  @Test(groups = "rocm", dataProvider = "causalShapes")
  public void scoresMatchAcrossReusedPrefixes(Variant variant, ShapeArm arm) throws Exception {
    var base = InputSchema.of(variant);
    var inputs = HipFixtures.observations(variant, 800, 1, TableWalk.Policy.CALLS, 240);
    Replay replay = new Replay(arm.shape().inputFormat(base));
    try (NDManager cpu = NDManager.newBaseManager(Device.cpu());
        PtStream stream = GpuCase.newStream()) {
      String trained =
          variant == Variant.TENHOU_4P ? System.getProperty("prism.v4.checkpoints") : null;
      var network =
          trained == null
              ? PerturbedNetwork.of(cpu, base, arm.shape(), 229)
              : PrismNetwork.load(Path.of(trained, arm.name()), cpu);
      WeightLayout layout = WeightLayout.of(network);
      // M は収集時と同じ上限まで射影を保存し、小さい入力でも表の上位の行を使う。
      var capacity = replay.capacity;
      if (arm == ShapeArm.V4_M) {
        capacity =
            new BatchPlan.Capacity(ROWS, 131_072, capacity.transitions(), capacity.actions());
      }
      try (HipBackend hip = HipBackend.open(HipLibrary.load(), stream, layout, capacity)) {
        var pack = WeightPack.build(layout, network, 3);
        hip.publish(pack);
        int count = 0;
        for (long[] words : inputs) {
          replay.encode(words);
          float[] expected = hip.forward(replay.full.input(), pack.version());
          Assert.assertTrue(
              HipParity.bitwiseEqual(expected, pagedScores(hip, replay)), arm + " score");
          count += expected.length;
          if (count == expected.length) {
            hip.publish(pack);
            Assert.assertTrue(
                HipParity.bitwiseEqual(expected, pagedScores(hip, replay)), "republish");
          } else if (count > 300 && count - expected.length <= 300) {
            // 入力を変えずに別の重みを公開し、前の重みで計算した埋め込みや射影が残らないことを確かめる。
            var changed = PerturbedNetwork.of(cpu, base, network.shape(), 230);
            pack = WeightPack.build(layout, changed, 4);
            hip.publish(pack);
            float[] fresh = hip.forward(replay.full.input(), pack.version());
            Assert.assertTrue(
                HipParity.bitwiseEqual(fresh, pagedScores(hip, replay)), "changed weights");
          }
        }
        Assert.assertEquals(pack.version(), 4L);
        System.out.println(
            "Event cache " + variant + " " + arm + " matched " + count + " action score bits");
      }
    }
  }

  @Test(groups = "rocm")
  public void cacheCapacityControlsResidencyWithoutChangingScores() throws Exception {
    var inputs = HipFixtures.observations(800, 1, TableWalk.Policy.CALLS, 100);
    try (NDManager cpu = NDManager.newBaseManager(Device.cpu());
        PtStream stream = GpuCase.newStream()) {
      var network = HipFixtures.network(cpu, ShapeArm.V4_M.shape(), 229);
      WeightLayout layout = WeightLayout.of(network);
      var pack = WeightPack.build(layout, network, 3);
      for (int slots : new int[] {0, 32}) {
        Replay replay = new Replay(ShapeArm.V4_M);
        try (HipBackend hip = HipBackend.open(HipLibrary.load(), stream, layout, replay.capacity)) {
          hip.publish(pack);
          var settings = new EventCacheSettings(EventCacheSettings.Mode.CAUSAL, slots, 1.0);
          Assert.assertEquals(hip.configureEventPages(1, settings).slots(), slots);
          boolean fits = false, exceeds = false;
          for (long[] words : inputs) {
            replay.encode(words);
            int events = replay.full.plan().count(Axis.EVENT);
            fits |= events > 0 && events <= slots;
            exceeds |= events > slots;
            replay.full.input().blob().position(1);
            float[] expected = hip.forward(replay.full.input(), 3);
            Assert.assertTrue(HipParity.bitwiseEqual(expected, pagedScores(hip, replay)));
            if (exceeds && (slots == 0 || fits)) break;
          }
          Assert.assertTrue(exceeds && (slots == 0 || fits));
        }
      }
    }
  }

  private static float[] pagedScores(HipBackend hip, Replay replay) {
    int n = replay.full.plan().count(Axis.ACTION);
    ByteBuffer out =
        ByteBuffer.allocateDirect(n * Float.BYTES).order(java.nio.ByteOrder.LITTLE_ENDIAN);
    // 通常経路が直前に送った符号を誤って読んでも試験が通ることを防ぐ。
    var plan = replay.full.plan();
    int bytes = plan.bytes();
    ByteBuffer restored = ByteBuffer.allocateDirect(bytes);
    for (int i = 0; i < bytes; i++) restored.put(i, (byte) 0xa5);
    hip.plan().copyIn(HipRegion.INPUT, restored, bytes);
    hip.submit(replay.full.input(), out, hip.weightsVersion(), replay.pages);
    hip.plan().copyOut(HipRegion.INPUT, restored, bytes);
    hip.synchronize();
    ByteBuffer full = replay.full.input().blob();
    for (Tensor tensor : Tensor.values()) {
      int start = plan.offset(tensor);
      int length =
          plan.count(tensor.axis()) * replay.format.width(tensor) * tensor.element().bytes();
      Assert.assertEquals(
          restored.slice(start, length).mismatch(full.slice(start, length)), -1, tensor.name());
    }
    float[] scores = new float[n];
    out.asFloatBuffer().get(scores);
    return scores;
  }
}

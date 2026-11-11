package com.epsilon.agent.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.expectThrows;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.PairList;
import com.epsilon.agent.learn.DataParallelGroup.Lane;
import com.epsilon.agent.learn.DataParallelGroup.LaneWork;
import com.epsilon.agent.learn.DataParallelGroup.Update;
import com.epsilon.agent.runtime.NetworkRuntime;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.testng.annotations.Test;

/**
 * 出力のヘッドが 2 つある小さなネットワークで、データ並列の更新を確かめる。1 レーンと 2 レーン（CPU）で更新の結果が同じになる。
 * 全てのパラメーターのグループの勾配を確かめてから更新するので、1 つのグループでも勾配が不正なら、どのパラメーターも変わらない。
 * 勾配の無いパラメーターは、グループが許すときだけ勾配 0 として扱う。
 */
public class DataParallelGroupTest {

  private static final int FEATURES = 3;
  private static final int OUTPUTS = 2;
  private static final int ROWS = 8;

  /** テスト用のパラメーターのグループ。名前に決まった文字列を含むパラメーターが入る。 */
  private enum Group implements DataParallelGroup.ParameterGroup {
    ALL("", true),
    ACTOR("actor", true),
    VALUE("value", true),
    STRICT_VALUE("value", false);

    private final String part;
    private final boolean missing;

    Group(String part, boolean missing) {
      this.part = part;
      this.missing = missing;
    }

    @Override
    public boolean includes(String parameter) {
      return parameter.contains(part);
    }

    @Override
    public boolean allowsMissingGradients() {
      return missing;
    }

    @Override
    public boolean allowsZeroGradient() {
      return false;
    }
  }

  @Test(groups = "native")
  public void oneAndTwoLanesTakeTheSameSteps() throws Exception {
    try (NetworkRuntime runtime = new NetworkRuntime(1, 1, 1);
        Model reference = model();
        Model canonical = model();
        Model replica = model();
        var single = new DataParallelGroup<>(List.of(new Lane(reference, false, runtime)));
        var dual =
            new DataParallelGroup<>(List.of(new Lane(canonical, false, runtime), new Lane(replica, false, runtime)))) {
      Optimizer singleOptimizer = adam(0.01f);
      Optimizer dualOptimizer = adam(0.01f);
      for (int step = 0; step < 3; step++) {
        float[][] batch = batch(step);
        LaneWork<Lane, Boolean> whole = lane -> learn(lane, batch, 0, ROWS, 1f, 1f);
        LaneWork<Lane, Boolean> head = lane -> learn(lane, batch, 0, ROWS / 2, 1f, 1f);
        LaneWork<Lane, Boolean> tail = lane -> learn(lane, batch, ROWS / 2, ROWS, 1f, 1f);
        single.execute(List.of(whole));
        single.step(new int[] {0}, List.of(new Update(Group.ALL, singleOptimizer, 1f)));
        dual.execute(List.of(head, tail));
        dual.step(new int[] {0, 1}, List.of(new Update(Group.ALL, dualOptimizer, 1f)));
        assertValues(values(canonical), values(reference), 1e-6f);
        assertValues(values(replica), values(canonical), 0f);
      }
      LaneWork<Lane, Boolean> idle = lane -> Boolean.TRUE;
      expectThrows(IllegalArgumentException.class, () -> dual.execute(new int[] {1, 1}, List.of(idle, idle)));
      List<Update> all = List.of(new Update(Group.ALL, dualOptimizer, 1f));
      expectThrows(IllegalArgumentException.class, () -> dual.step(new int[] {2}, all));
    }
  }

  @Test(groups = "native")
  public void everyGroupIsCheckedBeforeAnyUpdate() throws Exception {
    try (NetworkRuntime runtime = new NetworkRuntime(1, 1, 1);
        Model canonical = model();
        Model replica = model();
        var group =
            new DataParallelGroup<>(List.of(new Lane(canonical, false, runtime), new Lane(replica, false, runtime)))) {
      Map<String, float[]> before = values(canonical);
      List<Update> both = List.of(new Update(Group.ACTOR, adam(0.01f), 1f), new Update(Group.VALUE, adam(0.01f), 1f));
      float[][] batch = batch(1);
      int[] lanes = {0, 1};
      LaneWork<Lane, Boolean> head = lane -> learn(lane, batch, 0, ROWS / 2, 1f, 1f);
      LaneWork<Lane, Boolean> tail = lane -> learn(lane, batch, ROWS / 2, ROWS, 1f, 1f);

      // actor の勾配は有限だが、value の勾配に NaN がある
      LaneWork<Lane, Boolean> poisoned = lane -> learn(lane, batch, 0, ROWS / 2, 1f, Float.NaN);
      group.execute(List.of(poisoned, tail));
      expectThrows(IllegalStateException.class, () -> group.step(lanes, both));
      assertValues(values(canonical), before, 0f);
      assertValues(values(replica), before, 0f);

      // value の勾配は有限だが、actor の勾配が全て 0
      LaneWork<Lane, Boolean> valueOnly = lane -> learn(lane, batch, 0, ROWS / 2, 0f, 1f);
      group.execute(List.of(valueOnly, valueOnly));
      expectThrows(IllegalStateException.class, () -> group.step(lanes, both));
      assertValues(values(canonical), before, 0f);
      assertValues(values(replica), before, 0f);

      group.execute(List.of(head, tail));
      group.step(lanes, both);
      Map<String, float[]> after = values(canonical);
      for (Map.Entry<String, float[]> entry : before.entrySet()) {
        assertFalse(Arrays.equals(after.get(entry.getKey()), entry.getValue()), entry.getKey());
      }
      assertValues(values(replica), after, 0f);
    }
  }

  @Test(groups = "native")
  public void aMissingGradientCountsAsZeroOnlyWhereTheGroupAllowsIt() throws Exception {
    try (NetworkRuntime runtime = new NetworkRuntime(1, 1, 1);
        Model model = model();
        var group = new DataParallelGroup<>(List.of(new Lane(model, false, runtime)))) {
      Map<String, float[]> before = values(model);
      float[][] batch = batch(2);
      LaneWork<Lane, Boolean> actorOnly = lane -> learn(lane, batch, 0, ROWS, 1f, 0f);
      group.execute(List.of(actorOnly));
      List<Update> strict = List.of(new Update(Group.STRICT_VALUE, adam(0f), 1f));
      expectThrows(IllegalStateException.class, () -> group.step(new int[] {0}, strict));
      group.clearGradients();
      assertValues(values(model), before, 0f);

      group.execute(List.of(actorOnly));
      group.step(new int[] {0}, List.of(new Update(Group.ALL, adam(0f), 1f)));
      Map<String, float[]> after = values(model);
      for (Map.Entry<String, float[]> entry : before.entrySet()) {
        boolean changed = !Arrays.equals(after.get(entry.getKey()), entry.getValue());
        assertEquals(changed, entry.getKey().contains("actor"), entry.getKey());
      }
    }
  }

  @Test
  public void anUpdateNeedsAFinitePositiveScale() {
    Optimizer optimizer = adam(0.01f);
    for (float scale : new float[] {0f, -1f, Float.NaN, Float.POSITIVE_INFINITY}) {
      expectThrows(IllegalArgumentException.class, () -> new Update(Group.ALL, optimizer, scale));
    }
    expectThrows(IllegalArgumentException.class, () -> new DataParallelGroup<Lane>(List.of()));
  }

  /**
   * 行 [from, to) の損失 actorWeight·Σ(actor − y)² + valueWeight·Σ(value − y)² の勾配を溜める。重みが 0 の項は損失に入れないので、
   * そのヘッドのパラメーターには勾配が付かない。
   */
  private static Boolean learn(Lane lane, float[][] batch, int from, int to, float actorWeight, float valueWeight) {
    try (NDManager scope = lane.manager().newSubManager();
        GradientCollector collector = lane.newGradientCollector()) {
      NDArray inputs =
          scope.create(Arrays.copyOfRange(batch[0], from * FEATURES, to * FEATURES), new Shape(to - from, FEATURES));
      NDArray targets =
          scope.create(Arrays.copyOfRange(batch[1], from * OUTPUTS, to * OUTPUTS), new Shape(to - from, OUTPUTS));
      NDList outputs = lane.model().getBlock().forward(new ParameterStore(scope, false), new NDList(inputs), true);
      NDArray loss = scope.zeros(new Shape());
      if (actorWeight != 0f) {
        loss = loss.add(outputs.get(0).sub(targets).square().sum().mul(actorWeight));
      }
      if (valueWeight != 0f) {
        loss = loss.add(outputs.get(1).sub(targets).square().sum().mul(valueWeight));
      }
      collector.backward(loss);
    }
    return Boolean.TRUE;
  }

  /** 入力と目標の 2 つの配列を作る。乱数はステップごとに変える。 */
  private static float[][] batch(int step) {
    Random random = new Random(100 + step);
    float[] inputs = new float[ROWS * FEATURES];
    float[] targets = new float[ROWS * OUTPUTS];
    for (int index = 0; index < inputs.length; index++) {
      inputs[index] = random.nextFloat() * 2 - 1;
    }
    for (int index = 0; index < targets.length; index++) {
      targets[index] = random.nextFloat() * 2 - 1;
    }
    return new float[][] {inputs, targets};
  }

  /** 同じ乱数で初期化した、actor と value の 2 つの線形のヘッドを持つネットワーク。 */
  private static Model model() {
    Model model = Model.newInstance("data-parallel", Device.cpu(), "PyTorch");
    TwoHeads block = new TwoHeads();
    model.setBlock(block);
    block.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, FEATURES));
    Random random = new Random(7);
    for (Parameter parameter : block.getParameters().values()) {
      NDArray array = parameter.getArray();
      float[] values = new float[Math.toIntExact(array.size())];
      for (int index = 0; index < values.length; index++) {
        values[index] = (float) random.nextGaussian() * 0.5f;
      }
      try (NDArray initial = model.getNDManager().create(values, array.getShape())) {
        initial.copyTo(array);
      }
    }
    return model;
  }

  private static Optimizer adam(float weightDecay) {
    return Optimizer.adamW().optLearningRateTracker(Tracker.fixed(0.01f)).optWeightDecays(weightDecay).build();
  }

  private static Map<String, float[]> values(Model model) {
    Map<String, float[]> values = new LinkedHashMap<>();
    for (var entry : model.getBlock().getParameters()) {
      values.put(entry.getKey(), entry.getValue().getArray().toFloatArray());
    }
    return values;
  }

  private static void assertValues(Map<String, float[]> actual, Map<String, float[]> expected, float tolerance) {
    assertEquals(actual.keySet(), expected.keySet());
    for (Map.Entry<String, float[]> entry : expected.entrySet()) {
      float[] got = actual.get(entry.getKey());
      float[] want = entry.getValue();
      assertEquals(got.length, want.length, entry.getKey());
      for (int index = 0; index < want.length; index++) {
        assertEquals(got[index], want[index], tolerance, entry.getKey() + "[" + index + "]");
      }
    }
  }

  /** 入力を 2 つの線形のヘッド（actor と value）に通す。 */
  private static final class TwoHeads extends AbstractBlock {
    private final Linear actor = addChildBlock("actor", Linear.builder().setUnits(OUTPUTS).build());
    private final Linear value = addChildBlock("value", Linear.builder().setUnits(OUTPUTS).build());

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType type, Shape... shapes) {
      actor.initialize(manager, type, shapes);
      value.initialize(manager, type, shapes);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore store, NDList inputs, boolean training, PairList<String, Object> parameters) {
      return new NDList(
          actor.forward(store, inputs, training, parameters).singletonOrThrow(),
          value.forward(store, inputs, training, parameters).singletonOrThrow());
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      return new Shape[] {actor.getOutputShapes(inputShapes)[0], value.getOutputShapes(inputShapes)[0]};
    }
  }
}

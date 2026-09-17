package com.epsilon.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.PairList;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup.ParameterScope;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 小さいPyTorch モデルで、連続バッファ更新とパラメーターごとの独立AdamWを比較する。 */
public class DataParallelGroupTest {
  @DataProvider
  public Object[][] scopes() {
    return new Object[][] {{ParameterScope.ONLINE}, {ParameterScope.PRETRAIN_ALL}};
  }

  @Test(groups = "native", dataProvider = "scopes")
  public void reductionReplicaTailAndPausedActorMatchIndependentAdamW(ParameterScope scope)
      throws Exception {
    verifyReductionAndResume(scope, Device.cpu(), Device.cpu());
  }

  @DataProvider
  public Object[][] gpuLanes() {
    return new Object[][] {
      {ParameterScope.ONLINE, Device.gpu(0), Device.gpu(0)},
      {ParameterScope.ONLINE, Device.gpu(0), Device.gpu(1)},
      {ParameterScope.PRETRAIN_ALL, Device.gpu(0), Device.gpu(0)},
      {ParameterScope.PRETRAIN_ALL, Device.gpu(0), Device.gpu(1)}
    };
  }

  @Test(groups = "rocm", dataProvider = "gpuLanes")
  public void gpuReductionReplicaTailAndPausedActorMatchCpuAdamW(
      ParameterScope scope, Device canonicalDevice, Device replicaDevice) throws Exception {
    verifyReductionAndResume(scope, canonicalDevice, replicaDevice);
  }

  private static void verifyReductionAndResume(
      ParameterScope scope, Device canonicalDevice, Device replicaDevice) throws Exception {
    try (DecisionExecutionContext context = new DecisionExecutionContext();
        Model canonical = model(canonicalDevice);
        Model replica = model(replicaDevice);
        Model reference = model(Device.cpu());
        Group group = new Group(canonical, replica, scope, context)) {
      Optimizer actor = adam();
      Optimizer value = adam();
      Optimizer referenceActor = adam();
      Optimizer referenceValue = adam();

      // 学習ワーカーごとの重みが違っても、加算後に方策/価値をそれぞれ更新する。
      group.executeWork(
          List.of(
              lane -> {
                gradients(lane.model(), true, true, 0.25f);
                return true;
              },
              lane -> {
                gradients(lane.model(), true, true, 0.75f);
                return true;
              }));
      gradients(reference, true, true, 1.0f);
      fused(group, scope, actor, value);
      referenceStep(reference, referenceActor, true);
      referenceStep(reference, referenceValue, false);
      assertModelsEqual(canonical, reference);
      assertModelsEqual(replica, reference);

      // 強制行動の端数バッチは複製モデル 1だけを使い、方策のモーメントと重み減衰を動かさない。
      group.setActiveGradientLanes(new int[] {1});
      group.executeWork(
          List.of(
              lane -> {
                // 使用しない学習ワーカーの古い勾配が混入しないことも確認する。
                gradients(lane.model(), true, true, 99.0f);
                return true;
              },
              lane -> {
                gradients(lane.model(), false, true, -0.4f);
                return true;
              }));
      gradients(reference, false, true, -0.4f);
      group.applyFlattenedOptimizerStep(
          value,
          scope == ParameterScope.ONLINE
              ? ParameterScope.ONLINE_VALUE
              : ParameterScope.PRETRAIN_VALUE);
      referenceStep(reference, referenceValue, false);
      assertModelsEqual(canonical, reference);
      assertModelsEqual(replica, reference);

      // 方策の再開後も一致することで、価値のみのステップ中に更新回数が進んでいないと確認する。
      group.executeAssigned(
          new int[] {1},
          List.of(
              lane -> {
                gradients(lane.model(), true, true, -0.6f);
                return true;
              }));
      gradients(reference, true, true, -0.6f);
      fused(group, scope, actor, value);
      referenceStep(reference, referenceActor, true);
      referenceStep(reference, referenceValue, false);
      assertModelsEqual(canonical, reference);
      assertModelsEqual(replica, reference);
    }
  }

  private static void fused(Group group, ParameterScope scope, Optimizer actor, Optimizer value)
      throws Exception {
    if (scope == ParameterScope.ONLINE)
      group.applyFlattenedOnlineOptimizerSteps(actor, 1, value, 1);
    else group.applyFlattenedPretrainOptimizerSteps(actor, value);
  }

  private static Model model(Device device) {
    Model model = Model.newInstance("optimizer-contract", device, "PyTorch");
    TinyBlock block = new TinyBlock();
    model.setBlock(block);
    block.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, 3));
    int ordinal = 1;
    for (var parameter : block.getParameters().values()) {
      parameter.getArray().fillI(ordinal++ * 0.125f);
      parameter.getArray().setRequiresGradient(true);
    }
    return model;
  }

  private static void gradients(Model model, boolean actor, boolean value, float scale) {
    try (GradientCollector collector = model.getNDManager().getEngine().newGradientCollector()) {
      int ordinal = 1;
      for (var parameter : model.getBlock().getParameters()) {
        String name = parameter.getKey();
        if (!(name.contains("actor") ? actor : value)) continue;
        try (NDArray sum = parameter.getValue().getArray().sum();
            NDArray loss = sum.mul(scale * ordinal++)) {
          collector.backward(loss);
        }
      }
    }
  }

  private static void referenceStep(Model model, Optimizer optimizer, boolean actor) {
    for (var parameter : model.getBlock().getParameters()) {
      if (parameter.getKey().contains("actor") != actor) continue;
      NDArray weights = parameter.getValue().getArray();
      try (NDArray gradient = weights.getGradient()) {
        optimizer.update(parameter.getKey(), weights, gradient);
        gradient.fillI(0);
      }
    }
  }

  private static void assertModelsEqual(Model actual, Model expected) {
    for (var parameter : expected.getBlock().getParameters()) {
      float[] want = parameter.getValue().getArray().toFloatArray();
      float[] got =
          actual.getBlock().getParameters().get(parameter.getKey()).getArray().toFloatArray();
      Assert.assertEquals(got.length, want.length);
      for (int i = 0; i < want.length; i++)
        Assert.assertEquals(got[i], want[i], 2e-6f, parameter.getKey());
    }
  }

  private static Optimizer adam() {
    return Optimizer.adamW()
        .optLearningRateTracker(Tracker.fixed(0.003f))
        .optWeightDecays(0.01f)
        .optClipGrad(1000f)
        .build();
  }

  private static final class Group extends DataParallelGroup<DataParallelGroup.Lane> {
    Group(Model canonical, Model replica, ParameterScope scope, DecisionExecutionContext context) {
      super(
          List.of(
              new Lane(canonical, false, false, context),
              new Lane(
                  replica,
                  false,
                  replica.getNDManager().getDevice().isGpu()
                      && replica
                          .getNDManager()
                          .getDevice()
                          .equals(canonical.getNDManager().getDevice()),
                  context)),
          scope,
          null,
          DataType.FLOAT32,
          context,
          (selected, name) ->
              switch (selected) {
                case ONLINE, PRETRAIN_ALL -> true;
                case ONLINE_ACTOR, PRETRAIN_POLICY -> name.contains("actor");
                case ONLINE_VALUE, PRETRAIN_VALUE -> name.contains("value");
              });
      recordParameterReadiness();
    }
  }

  private static final class TinyBlock extends AbstractBlock {
    private final Linear actor = addChildBlock("actor", Linear.builder().setUnits(2).build());
    private final Linear value = addChildBlock("value", Linear.builder().setUnits(2).build());

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType type, Shape... shapes) {
      actor.initialize(manager, type, shapes);
      value.initialize(manager, type, shapes);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore store,
        NDList inputs,
        boolean training,
        PairList<String, Object> parameters) {
      return actor.forward(store, inputs, training, parameters);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      return actor.getOutputShapes(inputShapes);
    }
  }
}

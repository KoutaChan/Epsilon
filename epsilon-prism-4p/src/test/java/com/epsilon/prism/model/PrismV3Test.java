package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.Limits;
import com.epsilon.prism.backend.HipBackend;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.backend.WeightLayout;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.hip.HipShape;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism4p.input.InputSchema;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 型ごとの FFN を持つ形（v3 の構造）を確かめる（CPU、FP32）。見るのは、アームのパラメーターの数、初期化したままで s ≡ 0 になること、
 * 2 ステップで全てのパラメーターに勾配が届くこと、イベント 0 個と最長の行で値が有限なこと、初期化したままの候補層の 2 段目が 1 段の候補層と
 * 同じ関数になること、保存と読み込み、HIP の形の記述・重みの束・計画（GPU のグループでは計画を開いて回す）。
 */
public class PrismV3Test {
  private static final Logger log = LoggerFactory.getLogger(PrismV3Test.class);

  /** CPU で速く回す小さな v3 の形。 */
  private static final PrismShape SMALL =
      new PrismShape(64, 128, 64, 3, new PrismShape.Typed(128, 256, 128, true));

  @Test
  public void armsHaveTheDesignedParameterCounts() {
    assertCounts(ShapeArm.V2.shape(2), 1_939_604, 1_766_610);
    assertCounts(ShapeArm.V3.shape(3), 4_885_780, 4_712_786);
    assertCounts(ShapeArm.V3_ONE_STAGE.shape(3), 4_719_508, 4_546_514);
    assertCounts(ShapeArm.V3_CONTEXT_512.shape(3), 3_835_156, 3_662_162);
    Assert.assertThrows(IllegalArgumentException.class, () -> ShapeArm.V3.shape(2));
  }

  private static void assertCounts(PrismShape shape, long total, long inference) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
      network.initialize(manager);
      long all = count(network.getParameters().values());
      long played = count(network.inferenceParameters().values());
      log.info("PRISM-V3-COUNT shape={} total={} inference={}", shape, all, played);
      Assert.assertEquals(all, total, "total of " + shape);
      Assert.assertEquals(played, inference, "inference of " + shape);
    }
  }

  private static long count(List<Parameter> parameters) {
    long sum = 0;
    for (Parameter parameter : parameters) {
      sum += parameter.getArray().size();
    }
    return sum;
  }

  @Test
  public void blankScoresZeroAndEveryParameterReceivesGradientWithinTwoSteps() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager scope = manager.newSubManager()) {
      manager.getEngine().setRandomSeed(5);
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, SMALL);
      network.initialize(manager);
      ParameterStore store = new ParameterStore(manager, false);
      Batch batch = Batches.build(Batches.allKinds(6), 19);
      for (float score : scores(network, store, scope, batch)) {
        Assert.assertEquals(score, 0f, "blank score");
      }
      List<ValueLabel> labels = Batches.values(batch, 23);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 29);
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      Optimizer optimizer = Batches.adam(1e-3f);
      Set<String> reached = new HashSet<>();
      for (int step = 0; step < 2; step++) {
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              network.forward(
                  store,
                  batch.inputs(stepScope),
                  Pass.training(Precision.FLOAT32, measured, auxiliary));
          collector.backward(loss.total(stepScope, outputs, labels, auxiliary, 1f, labels.size(), 1f));
        }
        for (Pair<String, Parameter> pair : network.getParameters()) {
          try (NDArray gradient = pair.getValue().getArray().getGradient()) {
            if (Batches.maxAbs(gradient.toFloatArray()) > 0) {
              reached.add(pair.getKey());
            }
          }
        }
        Batches.step(network, optimizer);
      }
      List<String> missing = new ArrayList<>();
      for (Pair<String, Parameter> pair : network.getParameters()) {
        if (!reached.contains(pair.getKey())) {
          missing.add(pair.getKey());
        }
      }
      Assert.assertTrue(missing.isEmpty(), "no gradient: " + missing);
    }
  }

  @Test
  public void scoresStayFiniteWithoutEventsAndOnTheLongestRow() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager scope = manager.newSubManager()) {
      manager.getEngine().setRandomSeed(7);
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, SMALL);
      network.initialize(manager);
      Batches.perturb(network, 0.05f, 31);
      ParameterStore store = new ParameterStore(manager, false);
      List<Batch> batches =
          List.of(
              Batches.build(Batches.allKinds(0), 37),
              Batches.build(List.of(Batches.turn(Limits.MAX_EVENT_TOKENS), Batches.response(1)), 41));
      for (Batch batch : batches) {
        float[] scores = scores(network, store, scope, batch);
        float max = 0;
        for (float score : scores) {
          Assert.assertTrue(Float.isFinite(score), "score " + score);
          max = Math.max(max, Math.abs(score));
        }
        Assert.assertTrue(max > 0, "perturbed scores are all zero");
      }
    }
  }

  /** 2 段目の出力の重みが 0 なら、ほかの重みが同じ 1 段の候補層と s がビットで同じ。 */
  @Test
  public void blankSecondStageKeepsTheOneStageFunction() {
    PrismShape oneStage =
        new PrismShape(64, 128, 64, 3, new PrismShape.Typed(128, 256, 128, false));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager scope = manager.newSubManager()) {
      manager.getEngine().setRandomSeed(11);
      PrismNetwork two = new PrismNetwork(InputSchema.FORMAT, SMALL);
      two.initialize(manager);
      Batches.perturb(two, 0.05f, 43);
      PrismNetwork one = new PrismNetwork(InputSchema.FORMAT, oneStage);
      one.initialize(manager);
      Map<String, Parameter> byName = new HashMap<>();
      for (Pair<String, Parameter> pair : two.getParameters()) {
        String name = pair.getKey().substring(2);
        if (name.startsWith("reread_output")) {
          pair.getValue().getArray().set(new float[(int) pair.getValue().getArray().size()]);
        }
        byName.put(name, pair.getValue());
      }
      for (Pair<String, Parameter> pair : one.getParameters()) {
        pair.getValue().getArray().set(byName.get(pair.getKey().substring(2)).getArray().toFloatArray());
      }
      ParameterStore store = new ParameterStore(manager, false);
      Batch batch = Batches.build(Batches.allKinds(5), 47);
      Assert.assertEquals(scores(two, store, scope, batch), scores(one, store, scope, batch));
    }
  }

  @Test
  public void savedV3LoadsWithTheSameScores() throws IOException {
    Path root = Files.createTempDirectory("prism-v3");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager scope = manager.newSubManager()) {
      manager.getEngine().setRandomSeed(13);
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, SMALL);
      network.initialize(manager);
      Batches.perturb(network, 0.05f, 53);
      Path directory = root.resolve("model");
      network.save(directory);
      JsonObject json =
          JsonParser.parseString(Files.readString(directory.resolve("model.json"))).getAsJsonObject();
      Assert.assertEquals(json.getAsJsonObject("shape").getAsJsonObject("typed").get("context").getAsInt(), 256);
      Assert.assertEquals(PrismNetwork.readShape(directory), SMALL);
      PrismNetwork loaded = PrismNetwork.load(directory, manager);
      ParameterStore store = new ParameterStore(manager, false);
      Batch batch = Batches.build(Batches.allKinds(4), 59);
      Assert.assertEquals(scores(loaded, store, scope, batch), scores(network, store, scope, batch));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** 型ごとの全てのアームで、HIP の形の記述・重みの束・計画の形を作れ、HIP の経路を選べる。 */
  @Test(dataProvider = "typedArms")
  public void typedArmsBuildTheHipDescriptionPackAndPlanShape(ShapeArm arm) {
    PrismShape shape = arm.shape(ShapeArm.V3_DEPTH);
    Assert.assertTrue(WeightLayout.supports(shape));
    Assert.assertEquals(InferencePath.HIP.resolve(Device.gpu(0), shape), InferencePath.HIP);
    Assert.assertEquals(InferencePath.HIP.resolve(Device.cpu(), shape), InferencePath.EAGER);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
      network.initialize(manager);
      WeightLayout layout = WeightLayout.of(network);
      HipArchitecture architecture = layout.architecture();
      Assert.assertEquals(architecture.layers(), shape.layers());
      Assert.assertEquals(architecture.reread(), shape.reread());
      Assert.assertEquals(architecture.candidateFfn(), shape.candidateFfn());
      WeightPack pack = WeightPack.build(layout, network, 1);
      Assert.assertEquals(pack.buffer().remaining(), layout.bytes());
      HipShape plan = layout.hipShape(capacity(4));
      Assert.assertEquals(plan.fixedTokens(), InputSchema.FORMAT.fixedTokens());
      Assert.assertTrue(HipRegion.EVENT_KEY_VALUE.capacityBytes(plan) > 0);
      Assert.assertEquals(HipRegion.REREAD_MEMORY.capacityBytes(plan) > 0, shape.reread());
    }
  }

  /**
   * 型ごとの全てのアームで、HIP の計画を作れ、重みを写せ、順伝播が最後まで回り（s が有限）、カーネルを 1 本ずつでも回せる（GPU 1 枚）。
   */
  @Test(groups = "rocm", dataProvider = "typedArms")
  public void typedArmsRunEveryKernel(ShapeArm arm) {
    PrismShape shape = arm.shape(ShapeArm.V3_DEPTH);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0))) {
      manager.getEngine().setRandomSeed(17);
      PrismNetwork typed = new PrismNetwork(InputSchema.FORMAT, shape);
      typed.initialize(manager);
      Batches.perturb(typed, 0.05f, 19);
      WeightLayout layout = WeightLayout.of(typed);
      HostBatch batch = HipFixtures.encode(HipFixtures.mixedRows(6, 300));
      HipBatch hipBatch = HipBackend.batchOf(batch);
      try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, capacity(6))) {
        backend.publish(WeightPack.build(layout, typed, 2));
        for (float score : backend.forward(batch, 2)) {
          Assert.assertTrue(Float.isFinite(score), "score " + score);
        }
        log.info("PRISM-V3-HIP {} runs the whole forward", arm);
        backend.plan().copyIn(HipRegion.INPUT, batch.blob().duplicate().position(0), hipBatch.bytes());
        backend.plan().run(HipKernel.RELATION, 0, hipBatch);
        backend.plan().run(HipKernel.MLP_ALL, 0, hipBatch);
        backend.plan().run(HipKernel.ATTENTION_FIXED, 1, hipBatch);
        backend.plan().run(HipKernel.MLP_FIXED, 1, hipBatch);
        backend.plan().run(HipKernel.ATTENTION_FIXED, 2, hipBatch);
        backend.plan().run(HipKernel.MLP_LAST, 2, hipBatch);
        backend.plan().synchronize();
        if (shape.reread()) {
          backend.plan().run(HipKernel.REREAD_KEY_VALUE, 0, hipBatch);
          backend.plan().run(HipKernel.CANDIDATE, 0, hipBatch);
          backend.plan().run(HipKernel.REREAD, 0, hipBatch);
          backend.plan().synchronize();
        }
      }
    }
  }

  @DataProvider
  public Object[][] typedArms() {
    return new Object[][] {{ShapeArm.V3}, {ShapeArm.V3_ONE_STAGE}, {ShapeArm.V3_CONTEXT_512}};
  }

  /** {@code rows} 行のバッチを入れる容量。 */
  private static BatchPlan.Capacity capacity(int rows) {
    return new BatchPlan.Capacity(rows, rows * 110, rows * 192, rows * 32);
  }

  private static float[] scores(
      PrismNetwork network, ParameterStore store, NDManager scope, Batch batch) {
    try (NDManager pass = scope.newSubManager()) {
      return network
          .forward(store, batch.inputs(pass), Pass.inference(Precision.FLOAT32))
          .scores()
          .toFloatArray();
    }
  }
}

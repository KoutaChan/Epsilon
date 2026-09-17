package com.epsilon.major.ai.decision.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.ai.grp.EpsilonGrpExample;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpTrainer;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.major.ai.belief.EpsilonBeliefCheckpointManager;
import com.epsilon.major.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.major.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.major.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.major.ai.network.NetworkDevices;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 互換性検証には、tools/compatibility が過去のソースから独立して生成した保存データを使用する。 */
@Test(groups = "legacy")
public class LegacyCompatibilityTest {
  private static final String SERIES = "major";
  private static final String COMMIT = "d051b3619e417162facc0a5d4bdf94e5f5a632c2";
  private static final Gson GSON = new Gson();

  public void oldCheckpointsPreserveParameterNamesInputsAndForward() throws Exception {
    Path fixture = fixture();
    JsonObject expected =
        JsonParser.parseString(Files.readString(fixture.resolve("fixture.json"))).getAsJsonObject();
    Assert.assertEquals(expected.get("sourceCommit").getAsString(), COMMIT);
    Assert.assertEquals(
        DecisionInputSchema.fingerprint(), expected.get("inputFingerprint").getAsString());
    Assert.assertEquals(DecisionInputSchema.VERSION, expected.get("inputVersion").getAsInt());
    try (Model decision =
        EpsilonDecisionCheckpointManager.load(fixture.resolve("decision"), Device.cpu(), false)) {
      compare(
          expected.get("decisionParameters"),
          GSON.toJsonTree(shapes(decision)),
          "decisionParameters");
      compare(expected.get("decisions"), GSON.toJsonTree(decisions(decision)), "decisions");
    }
    try (Model belief =
        EpsilonBeliefCheckpointManager.load(fixture.resolve("belief"), Device.cpu())) {
      // 旧マニフェストには隠れ層幅がないため、パラメーターヘッダーから幅32を復元する。
      // 系列に同梱された既定値とは異なる値であることを意図している。
      compare(
          expected.get("beliefParameters"), GSON.toJsonTree(shapes(belief)), "beliefParameters");
      compare(
          expected.get("beliefForward"), GSON.toJsonTree(beliefForward(belief)), "beliefForward");
    }
    try (Model grp = EpsilonGrpCheckpointManager.load(fixture.resolve("grp"), Device.cpu())) {
      compare(expected.get("grpParameters"), GSON.toJsonTree(shapes(grp)), "grpParameters");
      compare(expected.get("grpForward"), GSON.toJsonTree(grpForward(grp)), "grpForward");
    }
  }

  public void oldActorAndValueAdamMomentsResumeAtTheSameSecondStep() throws Exception {
    Path fixture = fixture();
    JsonObject metadata =
        JsonParser.parseString(Files.readString(fixture.resolve("fixture.json"))).getAsJsonObject();
    if (!metadata.get("optimizerAvailable").getAsBoolean()) {
      throw new IllegalStateException(
          "Independent old optimizer unavailable: "
              + metadata.get("optimizerUnavailableReason").getAsString());
    }
    try (Model model =
            EpsilonDecisionCheckpointManager.load(
                fixture.resolve("optimizer-step1"), Device.cpu(), false);
        Model expected =
            EpsilonDecisionCheckpointManager.load(
                fixture.resolve("optimizer-step2"), Device.cpu(), false);
        var execution = new DecisionExecutionContext();
        var parallel =
            EpsilonDecisionDataParallel.openOnline(
                model, NetworkDevices.of(Device.cpu()), execution)) {
      Optimizer actor = optimizer();
      Optimizer value = optimizer();
      actor.loadState(model.getNDManager(), fixture.resolve("actor.state"));
      value.loadState(model.getNDManager(), fixture.resolve("value.state"));
      gradientStep(model, parallel, actor, value, 2);
      var actualParameters = model.getBlock().getParameters();
      var expectedParameters = expected.getBlock().getParameters();
      Assert.assertEquals(actualParameters.keys(), expectedParameters.keys());
      for (String name : expectedParameters.keys()) {
        var wanted = expectedParameters.get(name).getArray();
        var actual = actualParameters.get(name).getArray();
        Assert.assertEquals(actual.getShape(), wanted.getShape(), name);
        float[] a = actual.toFloatArray();
        float[] b = wanted.toFloatArray();
        for (int i = 0; i < a.length; i++)
          Assert.assertEquals(a[i], b[i], 2e-7f, name + "[" + i + "]");
      }
    }
  }

  public void oldGrpAdamMomentsResumeAtTheSameSecondStep() throws Exception {
    Path fixture = fixture();
    JsonObject metadata =
        JsonParser.parseString(Files.readString(fixture.resolve("fixture.json"))).getAsJsonObject();
    JsonObject config = metadata.getAsJsonObject("grpOptimizer");
    var nextExample = GSON.fromJson(metadata.get("grpNextExample"), EpsilonGrpExample.class);
    try (Model model =
            EpsilonGrpCheckpointManager.load(fixture.resolve("grp-optimizer-step1"), Device.cpu());
        Model expected =
            EpsilonGrpCheckpointManager.load(fixture.resolve("grp-optimizer-step2"), Device.cpu());
        var trainer =
            new EpsilonGrpTrainer(
                model,
                config.get("batchSize").getAsInt(),
                config.get("learningRate").getAsFloat(),
                config.get("weightDecay").getAsFloat(),
                config.get("gradClip").getAsFloat())) {
      EpsilonGrpCheckpointManager.loadOptimizerState(
          trainer, fixture.resolve("grp-optimizer-step1"));
      trainer.trainExamples(List.of(nextExample), 1);
      var actualParameters = model.getBlock().getParameters();
      var expectedParameters = expected.getBlock().getParameters();
      Assert.assertEquals(actualParameters.keys(), expectedParameters.keys());
      for (String name : expectedParameters.keys()) {
        var wanted = expectedParameters.get(name).getArray();
        var actual = actualParameters.get(name).getArray();
        Assert.assertEquals(actual.getShape(), wanted.getShape(), name);
        float[] a = actual.toFloatArray();
        float[] b = wanted.toFloatArray();
        for (int i = 0; i < a.length; i++)
          Assert.assertEquals(a[i], b[i], 1e-6f, name + "[" + i + "]");
      }
    }
  }

  public void oldCompiledDatasetPreservesSlabsAndSampleIdentities() throws Exception {
    Path fixture = fixture();
    JsonObject expected =
        JsonParser.parseString(Files.readString(fixture.resolve("fixture.json"))).getAsJsonObject();
    var dataset = EpsilonDecisionCompiledDataset.open(fixture.resolve("compiled"));
    Assert.assertEquals(dataset.manifest().rows(), 5L);
    Assert.assertEquals(dataset.manifest().batches(), 2);
    Assert.assertEquals(
        dataset.manifest().compilerIdentity(), expected.get("compilerIdentity").getAsString());
    List<String> digests = new ArrayList<>();
    List<Long> sampleIds = new ArrayList<>();
    for (var shard : dataset.manifest().shards()) {
      for (var indexed : dataset.readIndexedShard(shard)) {
        digests.add(batchDigest(indexed.batch()));
        for (long id : indexed.sampleIds()) sampleIds.add(id);
      }
    }
    Assert.assertEquals(sampleIds, List.of(101L, 102L, 103L, 201L, 202L));
    compare(expected.get("compiledBatches"), GSON.toJsonTree(digests), "compiledSlabs");
    List<String> epoch = new ArrayList<>();
    try (var cursor = dataset.openEpoch(1, 2)) {
      DecisionHostBatch batch;
      while ((batch = cursor.next()) != null) epoch.add(batchDigest(batch));
    }
    Assert.assertEquals(epoch, digests);
  }

  static String batchDigest(DecisionHostBatch batch) throws Exception {
    var digest = java.security.MessageDigest.getInstance("SHA-256");
    try (var out =
        new java.io.DataOutputStream(
            new java.security.DigestOutputStream(
                java.io.OutputStream.nullOutputStream(), digest))) {
      out.writeInt(batch.size());
      out.writeInt(batch.bucket().legalActionCapacity());
      out.writeInt(batch.bucket().actionTransitionCapacity());
      for (short value : batch.inputs().denseCategories()) out.writeShort(value);
      for (float value : batch.inputs().denseNumerics()) out.writeFloat(value);
      for (int value : batch.trainingTargets().categoricalSlab()) out.writeInt(value);
      for (float value : batch.trainingTargets().numericSlab()) out.writeFloat(value);
    }
    return java.util.HexFormat.of().formatHex(digest.digest());
  }

  private static Path fixture() {
    String root = System.getProperty("epsilon.legacyFixtures");
    Assert.assertNotNull(
        root,
        "Run generate-legacy-fixtures.ps1 and set -Depsilon.legacyFixtures=<absolute output root>");
    Path fixture = Path.of(root).resolve(SERIES).resolve("fixture");
    Assert.assertTrue(
        Files.isRegularFile(fixture.resolve("provenance.json")),
        "Missing independent old-ref provenance: " + fixture);
    return fixture;
  }

  private static void compare(JsonElement expected, JsonElement actual, String path) {
    if (expected.isJsonObject()) {
      Assert.assertEquals(
          actual.getAsJsonObject().keySet(), expected.getAsJsonObject().keySet(), path);
      for (var field : expected.getAsJsonObject().entrySet()) {
        compare(
            field.getValue(),
            actual.getAsJsonObject().get(field.getKey()),
            path + "." + field.getKey());
      }
    } else if (expected.isJsonArray()) {
      Assert.assertEquals(actual.getAsJsonArray().size(), expected.getAsJsonArray().size(), path);
      for (int i = 0; i < expected.getAsJsonArray().size(); i++) {
        compare(
            expected.getAsJsonArray().get(i), actual.getAsJsonArray().get(i), path + "[" + i + "]");
      }
    } else if (expected.isJsonPrimitive() && expected.getAsJsonPrimitive().isNumber()) {
      Assert.assertEquals(actual.getAsDouble(), expected.getAsDouble(), 2e-5, path);
    } else Assert.assertEquals(actual, expected, path);
  }

  // この再現用の演算は tools/compatibility/LegacyFixture.java と同じ内容に保つ。
  // 旧版の演算は保存した旧ソースからコンパイルし、このテストは再構築後のモジュールだけを参照する。
  static Optimizer optimizer() {
    return Optimizer.adamW()
        .optLearningRateTracker(Tracker.fixed(2e-4f))
        .optWeightDecays(1e-4f)
        .optClipGrad(0.5f)
        .build();
  }

  static void gradientStep(
      Model model, EpsilonDecisionDataParallel parallel, Optimizer actor, Optimizer value, int step)
      throws Exception {
    try (var manager = model.getNDManager().newSubManager();
        var collector = manager.getEngine().newGradientCollector()) {
      NDArray loss = manager.create(0f);
      var parameters = model.getBlock().getParameters();
      for (String name : parameters.keys()) {
        var parameter = parameters.get(name);
        if (parameter.requiresGradient()) {
          // パラメーターと更新段階ごとに異なる勾配を使い、一次元配列への配置変更や
          // Adamの移動平均の欠落を検出する。定数勾配では学習再開の一致を十分に検証できない。
          float scale = (1 + Math.floorMod(name.hashCode() + step * 13, 31)) * 1e-5f;
          loss = loss.add(parameter.getArray().sum().mul(scale));
        }
      }
      collector.backward(loss);
    }
    parallel.applyFlattenedOnlineOptimizerSteps(actor, 1.0, value, 1.0);
  }

  static Map<String, long[]> shapes(Model model) {
    Map<String, long[]> shapes = new LinkedHashMap<>();
    var parameters = model.getBlock().getParameters();
    for (String name : parameters.keys()) {
      shapes.put(name, parameters.get(name).getArray().getShape().getShape());
    }
    return shapes;
  }

  static List<Map<String, Object>> decisions(Model model) {
    List<Map<String, Object>> rows = new ArrayList<>();
    try (var server = EpsilonDecisionInferenceServer.forModel(model, 8)) {
      for (long seed : new long[] {8181L, 9047L}) {
        GameEngine engine = new GameEngine(seed);
        GameStepResult step = engine.stepHanchan();
        int index = 0;
        while (index < 64) {
          if (!(step instanceof GameStepResult.AwaitingDecisions boundary)) {
            if (step instanceof GameStepResult.HanchanEnded) break;
            step = engine.stepHanchan();
            continue;
          }
          var selections = new EngineSelectionBuffer();
          for (var point : boundary.decisions()) {
            var builder =
                DecisionBatchBuilder.inference(
                    1, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
            builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
            var prediction = server.evaluateBatch(builder.build()).getFirst();
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("seed", seed);
            row.put("index", index);
            row.put("player", point.player());
            row.put("kind", point.kind().name());
            row.put("actions", point.legalActions().stream().map(Object::toString).toList());
            row.put("logPolicy", prediction.policyLogProbabilities());
            row.put("hasValue", prediction.hasValue());
            row.put("value", prediction.hasValue() ? prediction.valueUtility() : 0f);
            rows.add(row);
            int action = Math.floorMod(index * 17 + 3, point.legalActions().size());
            selections.add(point.id(), point.legalActions().get(action));
            index++;
          }
          step = engine.commitDecisions(selections);
        }
      }
    }
    return rows;
  }

  static float[] beliefForward(Model model) {
    try (var manager = model.getNDManager().newSubManager()) {
      var inputs =
          new NDList(
              manager.zeros(new Shape(1, DecisionInputSchema.STATE_INT_COUNT), DataType.INT16),
              manager.zeros(new Shape(1, DecisionInputSchema.STATE_FLOAT_COUNT)));
      return model
          .getBlock()
          .forward(new ParameterStore(manager, false), inputs, false, new ai.djl.util.PairList<>())
          .singletonOrThrow()
          .toFloatArray();
    }
  }

  static float[] grpForward(Model model) {
    try (var manager = model.getNDManager().newSubManager()) {
      var inputs =
          new NDList(
              manager.zeros(new Shape(1, 3, EpsilonGrpFeature.FEATURE_SIZE)),
              manager.create(new float[] {3}));
      return model
          .getBlock()
          .forward(new ParameterStore(manager, false), inputs, false, new ai.djl.util.PairList<>())
          .singletonOrThrow()
          .toFloatArray();
    }
  }
}

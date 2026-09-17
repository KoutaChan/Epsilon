package com.epsilon.pico.ai.decision.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.epsilon.ai.grp.EpsilonGrpExample;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpTrainer;
import com.epsilon.pico.ai.belief.EpsilonBeliefCheckpointManager;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.ai.grp.EpsilonGrpCheckpointManager;
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
  private static final String SERIES = "pico";
  private static final String COMMIT = "6c14915f4c4558faaf5fb68f8561a722c8efb522";
  private static final Gson GSON = new Gson();

  public void oldBeliefAndGrpCheckpointsPreserveParametersAndForward() throws Exception {
    Path fixture = fixture();
    JsonObject expected =
        JsonParser.parseString(Files.readString(fixture.resolve("fixture.json"))).getAsJsonObject();
    Assert.assertEquals(expected.get("sourceCommit").getAsString(), COMMIT);
    Assert.assertEquals(
        DecisionInputSchema.fingerprint(), expected.get("inputFingerprint").getAsString());
    Assert.assertEquals(DecisionInputSchema.VERSION, expected.get("inputVersion").getAsInt());
    try (Model belief =
        EpsilonBeliefCheckpointManager.load(fixture.resolve("belief"), Device.cpu())) {
      // 隠れ層を持たない旧マニフェストでも、パラメーターヘッダーから保存時の32を復元する。
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

  private static String batchDigest(DecisionHostBatch batch) throws Exception {
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

  private static Map<String, long[]> shapes(Model model) {
    Map<String, long[]> shapes = new LinkedHashMap<>();
    var parameters = model.getBlock().getParameters();
    for (String name : parameters.keys()) {
      shapes.put(name, parameters.get(name).getArray().getShape().getShape());
    }
    return shapes;
  }

  private static float[] beliefForward(Model model) {
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

  private static float[] grpForward(Model model) {
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

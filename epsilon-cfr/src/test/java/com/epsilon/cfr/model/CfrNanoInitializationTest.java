package com.epsilon.cfr.model;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.nano.ai.network.NetworkFactory;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Nanoの実重みを流用しながら、方策を累積後悔と混同しない初期化を検証する。 */
@Test(groups = "native")
public class CfrNanoInitializationTest {
  private static final Set<String> NEW_COMPONENTS = Set.of("candidateHidden", "output");

  public void copiesEverySharedTensorWithoutMutatingSourceOrNewComponents() {
    try (Model nano = nano(16);
        Model destination = Model.newInstance("cfr-conversion-test", Device.cpu(), "PyTorch")) {
      CfrNetwork target = new CfrNetwork(config());
      destination.setBlock(target);
      target.initialize(destination.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
      Map<String, float[]> sourceBefore = parameters(nano.getBlock());
      Map<String, float[]> targetBefore = parameters(target);
      CfrModel.NanoInitializationReport report =
          target.initializeRepresentationFromNano(nano.getBlock());
      int tensors = 0;
      long scalars = 0;
      Set<String> preserved = new HashSet<>();
      for (var child : target.getChildren()) {
        String name = child.getKey().replaceFirst("^[0-9]+_?", "");
        Block block = child.getValue();
        if (NEW_COMPONENTS.contains(name)) {
          Assert.assertFalse(block.getParameters().isEmpty(), name);
          preserved.add(name);
          for (var parameter : block.getParameters()) {
            String fullName = child.getKey() + "_" + parameter.getKey();
            Assert.assertEquals(
                parameter.getValue().getArray().toFloatArray(),
                targetBefore.get(fullName),
                fullName);
          }
          continue;
        }
        Block source =
            switch (name) {
              case "stateEncoder" -> child(nano.getBlock(), "stateEncoder");
              case "stateReadout" -> child(nano.getBlock(), "policyStateReadout");
              default -> child(child(nano.getBlock(), "policyHead"), name);
            };
        for (var parameter : block.getParameters()) {
          Assert.assertEquals(
              parameter.getValue().getArray().toFloatArray(),
              source.getParameters().get(parameter.getKey()).getArray().toFloatArray(),
              name + "/" + parameter.getKey());
          tensors++;
          scalars += parameter.getValue().getArray().size();
        }
      }
      Assert.assertEquals(report.copiedParameters(), tensors);
      Assert.assertEquals(preserved, NEW_COMPONENTS);
      Assert.assertEquals(report.copiedScalars(), scalars);
      Assert.assertTrue(report.copiedScalars() > report.totalScalars() * .8);
      assertParameters(nano.getBlock(), sourceBefore);
    }
  }

  public void freshModelsKeepZeroOutputsAndResumeTrainingAfterConversion() throws Exception {
    var context = CfrModelTest.context(19381);
    var observation = CfrObservationCodec.encodePayload(context);
    int[] actions = context.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
    double[] policy = new double[actions.length];
    Arrays.fill(policy, 1d / actions.length);
    var row =
        new CfrTrainingRow(
            new CfrSampleId(1, context.episodeId(), context.decisionId(), context.seat()),
            actions,
            0,
            policy,
            1,
            0,
            0,
            observation);
    var checkpoint = Files.createTempDirectory("epsilon-cfr-nano-conversion-");
    try (Model nano = nano(16)) {
      for (CfrModel.Role role : CfrModel.Role.values()) {
        try (CfrModel converted = new CfrModel(config(), role, Device.cpu())) {
          var report = converted.initializeRepresentationFromNano(nano);
          Assert.assertTrue(report.copiedParameters() > 0);
          Assert.assertEquals(converted.updates(), 0L);
          Assert.assertEquals(converted.rawScores(observation), new float[actions.length]);
          var directory = checkpoint.resolve(role.name());
          converted.saveTraining(directory);
          try (CfrModel resumed = CfrModel.loadTraining(directory, Device.cpu())) {
            Assert.assertEquals(resumed.rawScores(observation), converted.rawScores(observation));
            Assert.assertTrue(Double.isFinite(resumed.train(List.of(row)).loss()));
            Assert.assertEquals(resumed.updates(), 1L);
            Assert.expectThrows(
                IllegalStateException.class, () -> resumed.initializeRepresentationFromNano(nano));
          }
          Assert.assertEquals(converted.rawScores(observation), new float[actions.length]);
        }
      }
    }
  }

  public void rejectsIncompatibleNanoBeforeChangingAnyWeights() {
    try (Model nano = nano(32);
        Model target = Model.newInstance("cfr-invalid-conversion", Device.cpu(), "PyTorch")) {
      CfrNetwork network = new CfrNetwork(config());
      target.setBlock(network);
      network.initialize(target.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
      var before = parameters(network);
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> network.initializeRepresentationFromNano(nano.getBlock()));
      assertParameters(network, before);
      Assert.expectThrows(
          IllegalArgumentException.class, () -> network.initializeRepresentationFromNano(network));
      assertParameters(network, before);
    }
  }

  private static CfrModelConfig config() {
    return new CfrModelConfig(
        16,
        2,
        .002f,
        0,
        0,
        8391,
        2,
        DecisionComputePrecision.FLOAT32,
        DecisionTensorTransfer.DIRECT_BUFFER,
        EpsilonUtilityProfile.TENHOU);
  }

  private static Model nano(int hidden) {
    return NetworkFactory.createDecisionModel(
        Device.cpu(), false, hidden, EpsilonUtilityProfile.TOP);
  }

  private static Block child(Block parent, String name) {
    return parent.getChildren().stream()
        .filter(entry -> entry.getKey().endsWith(name))
        .findFirst()
        .orElseThrow()
        .getValue();
  }

  private static Map<String, float[]> parameters(Block block) {
    Map<String, float[]> result = new LinkedHashMap<>();
    for (var parameter : block.getParameters())
      result.put(parameter.getKey(), parameter.getValue().getArray().toFloatArray());
    return result;
  }

  private static void assertParameters(Block block, Map<String, float[]> expected) {
    for (var parameter : block.getParameters())
      Assert.assertEquals(
          parameter.getValue().getArray().toFloatArray(),
          expected.get(parameter.getKey()),
          parameter.getKey());
  }
}

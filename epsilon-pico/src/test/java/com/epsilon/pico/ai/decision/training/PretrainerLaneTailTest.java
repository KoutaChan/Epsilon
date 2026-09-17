package com.epsilon.pico.ai.decision.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.pico.ai.decision.benchmark.EpsilonDecisionPretrainBenchmark;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.model.EpsilonDecisionNetwork;
import com.epsilon.pico.ai.network.NetworkDevices;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.DecisionPretrainSettings;
import com.epsilon.pico.config.settings.DecisionSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class PretrainerLaneTailTest {
  @DataProvider
  public Object[][] precision() {
    return new Object[][] {{DecisionComputePrecision.FLOAT32}, {DecisionComputePrecision.BFLOAT16}};
  }

  /** 2つの並列処理単位で学習する間に、1つの処理単位で合法手が一つの判断のバッチを学習しても、方策を固定して価値関数だけを更新することを検証する。 */
  @Test(groups = "native", dataProvider = "precision")
  public void bothLanesResumeAfterForcedTailWithoutChangingPolicy(
      DecisionComputePrecision precision) throws Exception {
    var settings =
        new DecisionPretrainSettings(
            8,
            1,
            1,
            4,
            2,
            DecisionTensorTransfer.DIRECT_BUFFER,
            precision,
            8,
            1,
            1,
            60,
            0.001f,
            1f,
            1f,
            0.5f,
            2f,
            2f);
    var config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.decision.utilityProfile", "TOP", "epsilon.decision.weightDecay", "0.5"));
    var choice =
        EpsilonDecisionPretrainBenchmark.syntheticBatch(4, new DecisionBucket(4, 1), config);
    var forced =
        EpsilonDecisionPretrainBenchmark.syntheticBatch(1, new DecisionBucket(1, 1), config);
    try (var execution = new DecisionExecutionContext();
        Model model =
            NetworkFactory.createDecisionModel(Device.cpu(), false, 16, EpsilonUtilityProfile.TOP);
        var trainer =
            EpsilonDecisionPretrainer.open(
                model,
                settings,
                NetworkDevices.of(Device.cpu(), Device.cpu()),
                execution,
                config.bind(DecisionSettings.class))) {
      Assert.assertEquals(trainer.activeDevices().size(), 2);
      trainer.trainEpoch(List.of(choice));
      var policyBefore = snapshot(model, EpsilonDecisionNetwork::isOfflinePolicyParameterName);
      var valueBefore = snapshot(model, EpsilonDecisionNetwork::isOfflineValueParameterName);

      var tail = trainer.trainEpoch(List.of(forced));
      Assert.assertEquals(tail.optimizerSteps(), 1);
      Assert.assertEquals(tail.policyOptimizerSteps(), 0);
      Assert.assertEquals(tail.rows(), 1L);
      Assert.assertEquals(tail.policyRows(), 0L);
      assertSnapshotUnchanged(model, policyBefore);
      Assert.assertTrue(anyParameterChanged(model, valueBefore), "Value must learn from the tail");

      var resumed = trainer.trainEpoch(List.of(choice));
      Assert.assertEquals(resumed.optimizerSteps(), 1);
      Assert.assertEquals(resumed.policyOptimizerSteps(), 1);
      Assert.assertEquals(resumed.policyRows(), 4L);
      Assert.assertTrue(anyParameterChanged(model, policyBefore), "Policy must resume learning");
      for (var parameter : model.getBlock().getParameters()) {
        Assert.assertEquals(
            parameter.getValue().getArray().getDataType(), DataType.FLOAT32, parameter.getKey());
      }
    }
  }

  private static Map<String, float[]> snapshot(Model model, Predicate<String> included) {
    Map<String, float[]> values = new LinkedHashMap<>();
    for (var parameter : model.getBlock().getParameters()) {
      if (included.test(parameter.getKey()))
        values.put(parameter.getKey(), parameter.getValue().getArray().toFloatArray());
    }
    return values;
  }

  private static void assertSnapshotUnchanged(Model model, Map<String, float[]> expected) {
    for (var entry : expected.entrySet()) {
      Assert.assertEquals(
          model.getBlock().getParameters().get(entry.getKey()).getArray().toFloatArray(),
          entry.getValue(),
          entry.getKey());
    }
  }

  private static boolean anyParameterChanged(Model model, Map<String, float[]> before) {
    for (var entry : before.entrySet()) {
      if (!Arrays.equals(
          model.getBlock().getParameters().get(entry.getKey()).getArray().toFloatArray(),
          entry.getValue())) return true;
    }
    return false;
  }
}

package com.epsilon.pico.ai.model;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.pico.ai.network.NetworkFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PicoArchitectureTest {
  /** 初期化済み Pico モデルの総パラメータ数が、想定する約151万個であることを検証する。 */
  @Test(groups = "native")
  public void initializedPicoRetainsStrategicContextWithinItsParameterBudget() {
    try (Model model =
        NetworkFactory.createDecisionModel(Device.cpu(), false, 128, EpsilonUtilityProfile.TOP)) {
      var parameters = model.getBlock().getParameters();
      long count =
          parameters.values().stream().mapToLong(parameter -> parameter.getArray().size()).sum();
      Assert.assertEquals(count, 1_507_396L);
      Assert.assertTrue(
          parameters.keys().stream().anyMatch(name -> name.contains("strategicContextEncoder")));
      Assert.assertTrue(
          parameters.keys().stream().anyMatch(name -> name.endsWith("alternativeOffsets")));
      Assert.assertFalse(
          parameters.keys().stream()
              .anyMatch(
                  name ->
                      name.contains("policyNodeEmbedding")
                          || name.contains("alternativeEmbedding")
                          || name.endsWith("alternativeHidden_bias")));
    }
  }
}

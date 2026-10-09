package com.epsilon.cfr.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.google.gson.Gson;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class CfrModelConfigTest {
  @Test
  public void metadataMustDeclareExecutionSettings() {
    Assert.expectThrows(
        RuntimeException.class,
        () ->
            new Gson()
                .fromJson(
                    """
                    {"hiddenSize":16,"microBatchSize":2,"learningRate":0.001,
                     "weightDecay":0,"gradientClip":0,"seed":1}
                    """,
                    CfrModelConfig.class));
  }

  @Test
  public void trainingAndInferenceBatchLimitsAreIndependent() {
    var config =
        new CfrModelConfig(
            256,
            512,
            2e-4f,
            1e-4f,
            .5f,
            1,
            4096,
            DecisionComputePrecision.BFLOAT16,
            DecisionTensorTransfer.PINNED_BUFFER,
            EpsilonUtilityProfile.PLACEMENT);
    Assert.assertEquals(config.microBatchSize(), 512);
    Assert.assertEquals(config.inferenceBatchSize(), 4096);
    Assert.assertEquals(config.precision(Device.gpu(0)), DecisionComputePrecision.BFLOAT16);
    Assert.assertEquals(config.precision(Device.cpu()), DecisionComputePrecision.FLOAT32);
    Assert.assertEquals(config.utilityProfile(), EpsilonUtilityProfile.PLACEMENT);
    Assert.assertEquals(
        new Gson().fromJson(new Gson().toJson(config), CfrModelConfig.class), config);
  }

  @DataProvider
  public Object[][] invalidUtilityProfiles() {
    return new Object[][] {{null}, {"UNKNOWN"}, {"SCORE"}};
  }

  @Test(dataProvider = "invalidUtilityProfiles")
  public void savedMetadataRequiresSupportedUtilityProfile(String profile) {
    Gson json = new Gson();
    var metadata =
        json.toJsonTree(
                new CfrModelConfig(
                    16,
                    2,
                    .001f,
                    0,
                    0,
                    1,
                    2,
                    DecisionComputePrecision.FLOAT32,
                    DecisionTensorTransfer.DIRECT_BUFFER,
                    EpsilonUtilityProfile.TENHOU))
            .getAsJsonObject();
    if (profile == null) metadata.remove("utilityProfile");
    else metadata.addProperty("utilityProfile", profile);
    Assert.expectThrows(
        RuntimeException.class, () -> json.fromJson(metadata, CfrModelConfig.class));
  }
}

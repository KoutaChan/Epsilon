package com.epsilon.pico.ai.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** HL-Gauss 損失で定数を再利用する実装について、参照式と損失値・勾配が一致することを検証する。 */
public class HlGaussLossTest {
  @DataProvider
  public Object[][] profiles() {
    return java.util.Arrays.stream(EpsilonUtilityProfile.values())
        .filter(EpsilonUtilityProfile::rankBased)
        .map(profile -> new Object[] {profile})
        .toArray(Object[][]::new);
  }

  @Test(groups = "native", dataProvider = "profiles")
  public void cachedDeviceConstantsPreserveHostObjectiveAndLogitGradient(
      EpsilonUtilityProfile profile) {
    int bins = EpsilonDecisionHlGauss.BIN_COUNT;
    float[] logits = new float[3 * bins];
    for (int i = 0; i < logits.length; i++) logits[i] = (float) Math.sin(i * 0.13) * 2;
    float[] targets = {profile.utilityForRank(0), -0.13f, profile.utilityForRank(3)};
    float[] weights = {0.5f, 2, 1.5f};
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
        EpsilonDecisionHlGauss.DeviceConstants constants =
            new EpsilonDecisionHlGauss.DeviceConstants(manager, profile)) {
      int retainedArrays = manager.getManagedArrays().size();
      for (int iteration = 0; iteration < 2; iteration++) {
        try (NDManager batch = manager.newSubManager()) {
          targets[1] = -0.13f + 0.2f * iteration;
          NDArray rankProbabilities =
              batch.create(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, new Shape(1, 4));
          float expectedUtility = 0;
          for (int rank = 0; rank < 4; rank++) {
            expectedUtility += (rank + 1) * 0.1f * profile.utilityForRank(rank);
          }
          Assert.assertEquals(
              constants.priorUtility(rankProbabilities).getFloat(), expectedUtility, 1e-7f);
          NDArray directLogits = batch.create(logits, new Shape(3, bins));
          NDArray cachedLogits = batch.create(logits, new Shape(3, bins));
          directLogits.setRequiresGradient(true);
          cachedLogits.setRequiresGradient(true);
          NDArray target = batch.create(targets);
          target.setRequiresGradient(true);
          NDArray weight = batch.create(weights);
          float directLoss;
          try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
            NDArray loss =
                EpsilonDecisionLoss.computeValueLoss(directLogits, target, weight, profile);
            directLoss = loss.getFloat();
            collector.backward(loss);
          }
          float cachedLoss;
          try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
            NDArray loss =
                EpsilonDecisionLoss.computeValueLoss(cachedLogits, target, weight, constants);
            cachedLoss = loss.getFloat();
            collector.backward(loss);
          }
          double expected = 0;
          for (int row = 0; row < 3; row++) {
            expected +=
                weights[row]
                    * EpsilonDecisionHlGauss.crossEntropyFromLogits(
                        logits, row * bins, targets[row], profile)
                    / 4;
          }
          Assert.assertEquals(cachedLoss, (float) expected, 3e-5f);
          Assert.assertEquals(cachedLoss, directLoss, 2e-6f);
          float[] direct = directLogits.getGradient().toFloatArray();
          float[] cached = cachedLogits.getGradient().toFloatArray();
          for (int i = 0; i < direct.length; i++) Assert.assertEquals(cached[i], direct[i], 2e-6f);
          Assert.assertEquals(target.getGradient().abs().sum().getFloat(), 0f);
        }
        Assert.assertEquals(
            manager.getManagedArrays().size(),
            retainedArrays,
            "Batch temporaries must not remain in the device constants manager");
      }
    }
  }
}

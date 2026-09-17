package com.epsilon.pico.ai.decision.training;

import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.config.settings.DecisionSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 保存データに記録されたモデルの幅、価値の定義、系列を、実行時の設定に依存せず検証する。 */
public class CheckpointManifestTest {
  private static final Gson JSON = new Gson();

  @Test
  public void storedShapeAndValueDefinitionSurviveDifferentRuntimeSettings() throws Exception {
    SettingsLoader settings =
        EpsilonSettings.of(
            Map.of("epsilon.decision.hidden", "64", "epsilon.decision.utilityProfile", "TOP"));
    EpsilonDecisionCheckpointBundle bundle =
        new EpsilonDecisionCheckpointBundle(11, 3, 29, 20, EpsilonUtilityProfile.LAST_AVOIDANCE);
    Path directory = Files.createTempDirectory("decision-manifest-");
    try {
      Files.writeString(directory.resolve("manifest.json"), JSON.toJson(bundle));
      EpsilonDecisionCheckpointBundle read =
          EpsilonDecisionCheckpointManager.loadManifest(directory);
      Assert.assertEquals(read.hidden, 20);
      Assert.assertEquals(
          read.valueDefinition,
          EpsilonDecisionHlGauss.fingerprint(EpsilonUtilityProfile.LAST_AVOIDANCE));
      Assert.assertEquals(read.series, "epsilon-pico");
      Assert.assertEquals(read.globalStep, 11);
      Assert.assertEquals(settings.bind(DecisionSettings.class).hidden(), 64);
      Assert.assertEquals(
          settings.bind(DecisionSettings.class).utilityProfile(), EpsilonUtilityProfile.TOP);
    } finally {
      Files.deleteIfExists(directory.resolve("manifest.json"));
      Files.delete(directory);
    }
  }

  @Test
  public void legacyManifestIsAcceptedButExplicitOtherSeriesIsRejected() throws Exception {
    EpsilonDecisionCheckpointBundle bundle =
        new EpsilonDecisionCheckpointBundle(0, 0, 0, 16, EpsilonUtilityProfile.TENHOU);
    Path directory = Files.createTempDirectory("decision-series-");
    try {
      bundle.series = null;
      Files.writeString(directory.resolve("manifest.json"), JSON.toJson(bundle));
      Assert.assertNull(EpsilonDecisionCheckpointManager.loadManifest(directory).series);
      bundle.series = "epsilon-nano";
      Files.writeString(directory.resolve("manifest.json"), JSON.toJson(bundle));
      IOException error =
          Assert.expectThrows(
              IOException.class, () -> EpsilonDecisionCheckpointManager.loadManifest(directory));
      Assert.assertTrue(error.getMessage().contains("series mismatch"));
      bundle.series = "epsilon-pico";
      bundle.architecture = "incompatible-architecture";
      Files.writeString(directory.resolve("manifest.json"), JSON.toJson(bundle));
      Assert.expectThrows(
          IOException.class, () -> EpsilonDecisionCheckpointManager.loadManifest(directory));
    } finally {
      Files.deleteIfExists(directory.resolve("manifest.json"));
      Files.delete(directory);
    }
  }
}

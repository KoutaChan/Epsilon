package com.epsilon.nano.ai.decision.training;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 学習再開後も初期条件と乱数シードが保持され、レポートの JSON 項目へ反映されることを検証する。 */
public class DecisionSelectedPgReportWriterTest {

  @DataProvider
  public Object[][] seedModesAndStatuses() {
    return new Object[][] {
      {0L, "UUID_DERIVED", DecisionSelectedPgDuelResult.Status.PAUSED, "PAUSED"},
      {41L, "FIXED", DecisionSelectedPgDuelResult.Status.HARMFUL, "HARMFUL"}
    };
  }

  @Test(dataProvider = "seedModesAndStatuses")
  public void restoredRunKeepsSeedAndStatusLabelsInCampaignReport(
      long configuredSeed,
      String seedLabel,
      DecisionSelectedPgDuelResult.Status status,
      String statusLabel)
      throws Exception {
    Path directory = Files.createTempDirectory("selected-pg-report-");
    Path stateFile = directory.resolve("run.json");
    Path reportFile = directory.resolve(DecisionSelectedPgReportWriter.CAMPAIGN_FILE);
    try {
      DecisionSelectedPgRunContext run =
          DecisionSelectedPgRunContext.create(directory, 3, configuredSeed, new UUID(17L, 29L));
      DecisionSelectedPgCampaignState.writeJson(stateFile, run);
      JsonObject saved = JsonParser.parseString(Files.readString(stateFile)).getAsJsonObject();
      Assert.assertEquals(saved.get("seedMode").getAsString(), seedLabel);
      DecisionSelectedPgRunContext restored =
          DecisionSelectedPgCampaignState.readJson(stateFile, DecisionSelectedPgRunContext.class);
      Assert.assertEquals(restored, run);
      Assert.assertEquals(restored.lineageSeed(1), run.lineageSeed(1));

      Path champion = directory.resolve("champion");
      Path candidate = directory.resolve("candidate");
      DecisionSelectedPgDuelResult duel =
          new DecisionSelectedPgDuelResult(
              1, 1, 3, 1, 1, 1, 4, 8L, status, champion, candidate, "fixed test result");
      DecisionSelectedPgCampaignResult result =
          new DecisionSelectedPgCampaignResult(1, 1, 4, 8L, List.of(duel), status);
      SettingsLoader config = EpsilonSettings.defaults();
      DecisionSelectedPgReportWriter.writeCampaign(
          directory,
          config.bind(DecisionSelectedPgCampaignSettings.class),
          "test-contract",
          restored.seedBase(),
          restored.seedMode(),
          result,
          champion,
          champion,
          candidate,
          config);
      JsonObject report = JsonParser.parseString(Files.readString(reportFile)).getAsJsonObject();
      Assert.assertEquals(report.get("runSeedMode").getAsString(), seedLabel);
      Assert.assertEquals(report.get("runSeedBase").getAsLong(), run.seedBase());
      Assert.assertEquals(report.get("status").getAsString(), statusLabel);
      Assert.assertEquals(
          report.getAsJsonArray("duels").get(0).getAsJsonObject().get("status").getAsString(),
          statusLabel);
    } finally {
      Files.deleteIfExists(reportFile);
      Files.deleteIfExists(stateFile);
      Files.delete(directory);
    }
  }
}

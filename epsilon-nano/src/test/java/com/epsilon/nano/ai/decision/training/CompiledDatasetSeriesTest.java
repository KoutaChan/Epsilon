package com.epsilon.nano.ai.decision.training;

import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 変換済みデータセットについて、同一系列の旧形式を受理し、他系列のデータを拒否することを検証する。 */
public class CompiledDatasetSeriesTest {
  @DataProvider
  public Object[][] series() {
    return new Object[][] {{null, true}, {"epsilon-nano", true}, {"epsilon-pico", false}};
  }

  @Test(dataProvider = "series")
  public void legacyAndOwnSeriesOpenButExplicitOtherSeriesIsRejected(
      String series, boolean accepted) throws Exception {
    Path directory = Files.createTempDirectory("compiled-series-");
    JsonObject manifest = new JsonObject();
    manifest.addProperty("formatVersion", 11);
    manifest.addProperty("schemaFingerprint", DecisionInputSchema.fingerprint());
    manifest.addProperty("identity", "series-boundary");
    manifest.addProperty("rows", 0);
    manifest.addProperty("batches", 0);
    manifest.add("shards", new JsonArray());
    if (series != null) manifest.addProperty("series", series);
    try {
      Files.writeString(directory.resolve("manifest.json"), manifest.toString());
      if (accepted) {
        Assert.assertNotNull(EpsilonDecisionCompiledDataset.open(directory));
      } else {
        IOException error =
            Assert.expectThrows(
                IOException.class, () -> EpsilonDecisionCompiledDataset.open(directory));
        Assert.assertTrue(error.getMessage().contains("series mismatch"));
      }
    } finally {
      Files.deleteIfExists(directory.resolve("manifest.json"));
      Files.delete(directory);
    }
  }
}

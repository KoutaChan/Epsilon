package com.epsilon.major.ai.decision.training;

import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 現行形式の系列境界と、v47より前の形式拒否を検証する。 */
public class CompiledDatasetSeriesTest {
  @DataProvider
  public Object[][] series() {
    return new Object[][] {{null, true}, {"epsilon", true}, {"epsilon-pico", false}};
  }

  @Test(dataProvider = "series")
  public void unmarkedAndOwnSeriesOpenButExplicitOtherSeriesIsRejected(
      String series, boolean accepted) throws Exception {
    Path directory = Files.createTempDirectory("compiled-series-");
    JsonObject manifest = new JsonObject();
    manifest.addProperty("formatVersion", 12);
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

  @Test
  public void format11DatasetIsRejected() throws Exception {
    Path directory = Files.createTempDirectory("compiled-v11-");
    JsonObject manifest = new JsonObject();
    manifest.addProperty("formatVersion", 11);
    manifest.addProperty("schemaFingerprint", DecisionInputSchema.fingerprint());
    manifest.addProperty("identity", "old-format");
    manifest.addProperty("rows", 0);
    manifest.addProperty("batches", 0);
    manifest.add("shards", new JsonArray());
    try {
      Files.writeString(directory.resolve("manifest.json"), manifest.toString());
      Assert.expectThrows(IOException.class, () -> EpsilonDecisionCompiledDataset.open(directory));
    } finally {
      Files.deleteIfExists(directory.resolve("manifest.json"));
      Files.delete(directory);
    }
  }
}

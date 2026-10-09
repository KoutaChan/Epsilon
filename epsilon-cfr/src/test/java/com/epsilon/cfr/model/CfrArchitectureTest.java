package com.epsilon.cfr.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 盤面入力のモデルが、旧履歴構造の重みを読み込み始める前に拒否することを検証する。 */
public class CfrArchitectureTest {
  @Test
  public void byteSequencePolicyIsRejectedBeforeOpeningWeights() throws IOException {
    for (var role : CfrModel.Role.values()) {
      Path directory = Files.createTempDirectory("cfr-old-policy-");
      Path metadata = directory.resolve("model.json");
      try {
        JsonObject value = metadata(3, "epsilon-cfr-nano-feature-spine-history-gru-v2");
        value.addProperty("role", role.name());
        Files.writeString(metadata, value.toString());
        IOException error =
            Assert.expectThrows(
                IOException.class, () -> CfrModel.loadModel(directory, Device.cpu()));
        Assert.assertEquals(error.getMessage(), "CFR model architecture/schema mismatch");
      } finally {
        Files.deleteIfExists(metadata);
        Files.delete(directory);
      }
    }
  }

  private static JsonObject metadata(int version, String architecture) {
    var value = new JsonObject();
    value.addProperty("version", version);
    value.addProperty("architecture", architecture);
    value.addProperty("inputSchema", DecisionInputSchema.fingerprint());
    value.add(
        "config",
        new Gson()
            .toJsonTree(
                new CfrModelConfig(
                    16,
                    2,
                    .002f,
                    0,
                    0,
                    1337,
                    2,
                    DecisionComputePrecision.FLOAT32,
                    DecisionTensorTransfer.DIRECT_BUFFER,
                    EpsilonUtilityProfile.TENHOU)));
    value.addProperty("updates", 0);
    value.addProperty("trainingState", true);
    return value;
  }
}

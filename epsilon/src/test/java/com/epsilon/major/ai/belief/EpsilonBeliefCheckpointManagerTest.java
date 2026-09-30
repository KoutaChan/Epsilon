package com.epsilon.major.ai.belief;

import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Belief checkpointが方策固有入力ではなく状態入力だけへ結び付くことを検証する。 */
public class EpsilonBeliefCheckpointManagerTest {

  @Test
  public void stateFingerprintIsAcceptedAndFullDecisionFingerprintIsRejected() throws Exception {
    Path directory = Files.createTempDirectory("belief-state-fingerprint-");
    EpsilonBeliefCheckpointBundle bundle = new EpsilonBeliefCheckpointBundle(1, 2);
    bundle.series = "epsilon";
    bundle.hidden = 64;
    bundle.inputFingerprint = DecisionInputSchema.stateFingerprint();
    Path manifest = directory.resolve("manifest.json");
    try {
      Files.writeString(manifest, new Gson().toJson(bundle));
      Assert.assertEquals(
          EpsilonBeliefCheckpointManager.loadManifest(directory).inputFingerprint,
          DecisionInputSchema.stateFingerprint());

      bundle.inputFingerprint = DecisionInputSchema.fingerprint();
      Files.writeString(manifest, new Gson().toJson(bundle));
      Assert.expectThrows(
          IOException.class, () -> EpsilonBeliefCheckpointManager.loadManifest(directory));
    } finally {
      Files.deleteIfExists(manifest);
      Files.delete(directory);
    }
  }
}

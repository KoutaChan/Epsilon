package com.epsilon.cfr.training;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrReplayStore;
import com.epsilon.cfr.data.ObservationPayload;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrWarmStartRunnerTest {
  @Test
  public void teacherRowsKeepFullDistributionAndUseUniformSupervisedWeights() throws Exception {
    var episode =
        new CfrEpisode(
            5,
            List.of(
                new CfrDecision(
                    7,
                    2,
                    new int[] {1, 3},
                    1,
                    new double[] {.2, .8},
                    .8,
                    ObservationPayload.copyOf(new byte[] {4}))),
            new double[] {1, 0, -1, 0});
    var rows = CfrWarmStartRunner.supervisedRows(episode);
    Assert.assertEquals(rows.getFirst().targetPolicy(), new double[] {.2, .8});
    Assert.assertEquals(rows.getFirst().utility(), 0.0);
    Assert.assertEquals(rows.getFirst().logRegretWeight(), Double.NEGATIVE_INFINITY);
    Assert.assertEquals(rows.getFirst().logAverageWeight(), 0.0);
    Path directory = Files.createTempDirectory("cfr-teacher-corpus-");
    try (var corpus = new CfrReplayStore(directory, "teacher", 1, 2, 11, 12, 1024, 0, 1 << 20)) {
      Assert.assertTrue(corpus.appendEpisode(1, 5, rows));
      corpus.save(0);
    }
    try (var restored = CfrReplayStore.loadGeneration(directory, "teacher", 0)) {
      Assert.assertFalse(restored.appendEpisode(1, 5, rows));
      Assert.assertEquals(
          restored.averageSampler().sample(new SplittableRandom(1)).targetPolicy(),
          new double[] {.2, .8});
      Assert.assertEquals(restored.regretSampler().metrics().positiveWeightRows(), 0);
    }
    Assert.assertTrue(CfrWarmStartRunner.heldout(0));
    Assert.assertTrue(CfrWarmStartRunner.heldout(5));
    Assert.assertFalse(CfrWarmStartRunner.heldout(1));
  }

  @Test
  public void trainDoesNotOverwriteAnInterruptedInitialization() throws Exception {
    Path run = Files.createTempDirectory("cfr-initialization-guard-");
    CfrConfig.defaults().write(run.resolve("config.json"));
    String original = Files.readString(run.resolve("config.json"));
    Assert.expectThrows(IOException.class, () -> CfrCampaign.train(CfrConfig.defaults(), run));
    Assert.assertEquals(Files.readString(run.resolve("config.json")), original);
    Assert.assertFalse(Files.exists(run.resolve("current.json")));
  }
}

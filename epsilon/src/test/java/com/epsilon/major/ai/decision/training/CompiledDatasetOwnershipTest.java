package com.epsilon.major.ai.decision.training;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.benchmark.EpsilonDecisionPretrainBenchmark;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.config.settings.EpsilonSettings;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CompiledDatasetOwnershipTest {
  /** 復号したバッチを変更しても、復号用の作業領域、別のバッチ、保存済みの分割ファイルに影響しないことを検証する。 */
  @Test
  public void decodedSlabsStayIndependentAcrossScratchReuseAndRepeatedReads() throws Exception {
    SettingsLoader config = EpsilonSettings.of(Map.of("epsilon.decision.utilityProfile", "TOP"));
    DecisionHostBatch first =
        EpsilonDecisionPretrainBenchmark.syntheticBatch(3, new DecisionBucket(4, 1), config);
    DecisionHostBatch second =
        EpsilonDecisionPretrainBenchmark.syntheticBatch(2, new DecisionBucket(8, 4), config);
    Path directory = Files.createTempDirectory("pico-compiled-ownership-");
    try {
      try (var writer =
          EpsilonDecisionCompiledDataset.newWriter(
              directory, "ownership", "source", "teacher", "configuration", 8)) {
        writer.add(first, new long[] {11, 12, 13});
        writer.add(second, new long[] {21, 22});
        writer.finish();
      }
      var dataset = EpsilonDecisionCompiledDataset.open(directory);
      var shard = dataset.manifest().shards().getFirst();
      List<DecisionHostBatch> decoded = dataset.readShard(shard);
      assertBatchEquals(decoded.getFirst(), first);
      assertBatchEquals(decoded.get(1), second);

      Arrays.fill(decoded.getFirst().inputs().denseCategories(), (short) -1);
      Arrays.fill(decoded.getFirst().inputs().denseNumerics(), -1f);
      Arrays.fill(decoded.getFirst().trainingTargets().categoricalSlab(), -1);
      Arrays.fill(decoded.getFirst().trainingTargets().numericSlab(), -1f);

      assertBatchEquals(decoded.get(1), second);
      var repeated = dataset.readIndexedShard(shard);
      assertBatchEquals(repeated.getFirst().batch(), first);
      assertBatchEquals(repeated.get(1).batch(), second);
      Assert.assertEquals(repeated.getFirst().sampleIds(), new long[] {11, 12, 13});
      Assert.assertEquals(repeated.get(1).sampleIds(), new long[] {21, 22});
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  private static void assertBatchEquals(DecisionHostBatch actual, DecisionHostBatch expected) {
    Assert.assertEquals(actual.size(), expected.size());
    Assert.assertEquals(actual.bucket(), expected.bucket());
    Assert.assertEquals(actual.inputs().denseCategories(), expected.inputs().denseCategories());
    Assert.assertEquals(actual.inputs().denseNumerics(), expected.inputs().denseNumerics());
    Assert.assertEquals(
        actual.trainingTargets().categoricalSlab(), expected.trainingTargets().categoricalSlab());
    Assert.assertEquals(
        actual.trainingTargets().numericSlab(), expected.trainingTargets().numericSlab());
  }
}

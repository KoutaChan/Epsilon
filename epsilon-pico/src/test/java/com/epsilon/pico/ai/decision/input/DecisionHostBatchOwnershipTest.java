package com.epsilon.pico.ai.decision.input;

import com.epsilon.pico.ai.decision.benchmark.EpsilonDecisionPretrainBenchmark;
import com.epsilon.pico.config.settings.EpsilonSettings;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 復号済みの入力・教師データの配列を複製せずにバッチへ引き渡すことを検証する。 */
public class DecisionHostBatchOwnershipTest {
  @Test
  public void decodedTrainingBatchTakesAllFourSlabsWithoutCopying() {
    DecisionHostBatch source =
        EpsilonDecisionPretrainBenchmark.syntheticBatch(
            2,
            new DecisionBucket(4, 1),
            EpsilonSettings.of(Map.of("epsilon.decision.utilityProfile", "TOP")));
    short[] inputCategories = source.inputs().denseCategories();
    float[] inputNumerics = source.inputs().denseNumerics();
    int[] targetCategories = source.trainingTargets().categoricalSlab();
    float[] targetNumerics = source.trainingTargets().numericSlab();

    DecisionHostBatch restored =
        DecisionHostBatch.takeEncodedTrainingBatch(
            2,
            new DecisionBucket(4, 1),
            inputCategories,
            inputNumerics,
            targetCategories,
            targetNumerics);

    Assert.assertSame(restored.inputs().denseCategories(), inputCategories);
    Assert.assertSame(restored.inputs().denseNumerics(), inputNumerics);
    Assert.assertSame(restored.trainingTargets().categoricalSlab(), targetCategories);
    Assert.assertSame(restored.trainingTargets().numericSlab(), targetNumerics);
    Assert.assertEquals(restored.size(), 2);
    Assert.assertEquals(restored.sliceRows(0, 2).sampleWeightMass(), 2.0);
  }
}

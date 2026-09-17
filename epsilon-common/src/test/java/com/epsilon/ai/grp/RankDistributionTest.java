package com.epsilon.ai.grp;

import java.util.HashSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 24通りの順位順列の符号化、順位周辺確率の性質、一様分布、保存されたモデル構成の復元を検証する。 */
public class RankDistributionTest {
  @Test
  public void rankCodesEnumerateExactlyTheTwentyFourPermutations() {
    var permutations = new HashSet<String>();
    for (int code = 0; code < 24; code++) {
      int[] ranks = EpsilonGrpRanks.decode(code);
      Assert.assertEquals(EpsilonGrpRanks.encode(ranks), code);
      Assert.assertTrue(permutations.add(java.util.Arrays.toString(ranks)));
      var sorted = ranks.clone();
      java.util.Arrays.sort(sorted);
      Assert.assertEquals(sorted, new int[] {0, 1, 2, 3});
    }
    Assert.expectThrows(
        IllegalArgumentException.class, () -> EpsilonGrpRanks.encode(new int[] {0, 1, 1, 3}));
  }

  @Test
  public void marginalProbabilitiesIgnoreSeatAndRankGaugeOffsets() {
    float[] logits = {2, -1, 0, 3, 1, 4, -3, 2, 0, 1, 2, -2, -4, 2, 1, 0};
    float[] shifted = logits.clone();
    for (int seat = 0; seat < 4; seat++) {
      for (int rank = 0; rank < 4; rank++) shifted[seat * 4 + rank] += seat * 7 - rank * 3;
    }
    float[] baseline = EpsilonGrpSinkhorn.marginals(logits);
    float[] actual = EpsilonGrpSinkhorn.marginals(shifted);
    for (int i = 0; i < 16; i++) {
      Assert.assertEquals(actual[i], baseline[i], 1e-6f);
      Assert.assertTrue(actual[i] > 0 && actual[i] < 1);
    }
    for (int i = 0; i < 4; i++) {
      float row = 0, column = 0;
      for (int j = 0; j < 4; j++) {
        row += actual[i * 4 + j];
        column += actual[j * 4 + i];
      }
      Assert.assertEquals(row, 1f, 1e-6f);
      Assert.assertEquals(column, 1f, 1e-6f);
    }
  }

  @Test
  public void uniformEvidenceGivesEqualRankProbabilities() {
    for (float p : EpsilonGrpSinkhorn.marginals(new float[16])) Assert.assertEquals(p, .25f, 1e-7f);
  }

  @Test
  public void checkpointArchitectureRecoversStoredCapacity() throws Exception {
    var saved = new EpsilonGrpCheckpointBundle(11, 3, 48, 1);
    var config = saved.configuration();
    Assert.assertEquals(config, new EpsilonGrpCheckpointBundle.Configuration(48, 1));
    saved.architecture =
        saved.architecture.replace("marginalRepair=affine", "marginalRepair=other");
    Assert.expectThrows(java.io.IOException.class, saved::configuration);
  }
}

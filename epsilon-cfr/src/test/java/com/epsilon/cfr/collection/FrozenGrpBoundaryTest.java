package com.epsilon.cfr.collection;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class FrozenGrpBoundaryTest {
  @Test(groups = "native")
  public void sharedGrpProbabilitiesUseTheConfiguredRankUtility() {
    try (Model model = Model.newInstance("cfr-boundary-test", Device.cpu(), "PyTorch")) {
      var network = new EpsilonGrpNetwork(16, 1);
      model.setBlock(network);
      network.initialize(
          model.getNDManager(),
          DataType.FLOAT32,
          new Shape(-1, -1, EpsilonGrpFeature.FEATURE_SIZE),
          new Shape(-1));
      var inference = new EpsilonGrpInference(model);
      List<float[]> prefixes =
          List.of(
              RootState.initial().publicPrefix(),
              EpsilonGrpFeature.fromProgress(7, 1, 2, new int[] {42000, 13000, 27000, 16000}));
      List<float[]> probabilities = inference.predictMarginalProbabilities(prefixes);
      for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
        if (!profile.rankBased()) continue;
        var boundary = new FrozenGrpBoundary(inference, profile);
        List<double[]> actual = boundary.evaluateBatch(prefixes);
        for (int row = 0; row < prefixes.size(); row++) {
          for (int seat = 0; seat < 4; seat++) {
            double expected = 0;
            for (int rank = 0; rank < 4; rank++)
              expected +=
                  probabilities.get(row)[seat * 4 + rank] * profile.utilityForRankDouble(rank);
            Assert.assertEquals(actual.get(row)[seat], expected, 1e-12);
          }
        }
        double[] single = boundary.evaluate(prefixes.getFirst());
        for (int seat = 0; seat < 4; seat++)
          Assert.assertEquals(single[seat], actual.getFirst()[seat], 1e-6);
      }
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> new FrozenGrpBoundary(inference, EpsilonUtilityProfile.SCORE));
    }
  }
}

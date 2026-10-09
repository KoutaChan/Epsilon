package com.epsilon.cfr.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class CfrForcedPolicyNativeTest {
  @DataProvider
  public Object[][] roles() {
    return new Object[][] {{CfrModel.Role.AVERAGE}, {CfrModel.Role.REGRET}};
  }

  @Test(groups = "native", dataProvider = "roles")
  public void forcedPoliciesRetainPayloadAndLaterChoicesUseTheirOwnBoard(CfrModel.Role role) {
    DecisionContext base = CfrModelTest.context(877);
    int[] actions = base.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
    double[] teacher = new double[actions.length];
    if (role == CfrModel.Role.AVERAGE) teacher[0] = 1;
    else Arrays.fill(teacher, 1d / teacher.length);
    var training =
        new CfrTrainingRow(
            new CfrSampleId(1, base.episodeId(), base.decisionId(), base.seat()),
            actions,
            0,
            teacher,
            .5,
            0,
            0,
            CfrObservationCodec.encodePayload(base));
    try (var model =
        new CfrModel(
            new CfrModelConfig(
                16,
                8,
                .01f,
                0,
                0,
                919,
                8,
                DecisionComputePrecision.FLOAT32,
                DecisionTensorTransfer.DIRECT_BUFFER,
                EpsilonUtilityProfile.TENHOU),
            role,
            Device.cpu())) {
      for (int update = 0; update < 3; update++) model.train(List.of(training));
      try (var frozen = model.copyForInference(Device.cpu())) {
        var coldForced = withActions(base, true);
        assertParityAndPayload(frozen, List.of(coldForced));
        assertParityAndPayload(frozen, List.of(base));
        double[] trained = frozen.evaluate(List.of(base)).getFirst().targetPolicy();
        Assert.assertTrue(
            Arrays.stream(trained).max().orElseThrow() - Arrays.stream(trained).min().orElseThrow()
                > 1e-5,
            "Board inference must exercise a nonuniform learned policy");
        var later = CfrModelTest.context(878);
        var firstForced = withActions(base, true);
        var secondForced = withActions(later, true);
        assertParityAndPayload(frozen, List.of(firstForced, secondForced));
        assertParityAndPayload(frozen, List.of(firstForced, later, secondForced, base));
        assertParityAndPayload(frozen, List.of(secondForced));
        assertParityAndPayload(frozen, List.of(later));
      }
    }
  }

  private static void assertParityAndPayload(CfrModel model, List<DecisionContext> contexts) {
    var payloads = contexts.stream().map(CfrObservationCodec::encodePayload).toList();
    var expected = model.evaluateObservations(payloads);
    var actual = model.evaluate(contexts);
    for (int row = 0; row < contexts.size(); row++) {
      Assert.assertEquals(actual.get(row).observationPayload(), payloads.get(row));
      double[] wanted = expected.get(row).targetPolicy(), found = actual.get(row).targetPolicy();
      Assert.assertEquals(found.length, wanted.length);
      if (found.length == 1) Assert.assertEquals(found, new double[] {1});
      for (int slot = 0; slot < found.length; slot++)
        Assert.assertEquals(found[slot], wanted[slot], 1e-6);
    }
  }

  private static DecisionContext withActions(DecisionContext base, boolean forced) {
    return new DecisionContext(
        base.episodeId(),
        base.decisionId(),
        base.seat(),
        base.observation(),
        forced ? base.legalActions().subList(0, 1) : base.legalActions());
  }
}

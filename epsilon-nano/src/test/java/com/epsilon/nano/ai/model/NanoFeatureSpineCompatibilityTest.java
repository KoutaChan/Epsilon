package com.epsilon.nano.ai.model;

import ai.djl.Device;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.google.gson.Gson;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 抽出前の真正Nanoから保存したcheckpointと32局面の出力を固定し、構造と数値の互換性を検証する。 */
public class NanoFeatureSpineCompatibilityTest {
  private record Probe(
      long seed, long decisionId, float[] alternative, float[] candidate, float[] riichi) {}

  private record Fixture(int hidden, List<String> parameterNames, List<Probe> probes) {}

  @Test(groups = "native")
  public void originalNanoCheckpointKeepsParameterNamesAndPolicyScores() throws Exception {
    Path directory =
        Path.of(
                Objects.requireNonNull(getClass().getResource("/nano-policy-head-v1/expected.json"))
                    .toURI())
            .getParent();
    Fixture expected =
        new Gson().fromJson(Files.readString(directory.resolve("expected.json")), Fixture.class);
    try (var model = NetworkFactory.createDecisionModel(Device.cpu(), false, expected.hidden())) {
      var network = (EpsilonDecisionNetwork) model.getBlock();
      Assert.assertEquals(network.getParameters().keys(), expected.parameterNames());
      model.load(directory, "nano");
      int index = 0;
      for (long seed : expected.probes().stream().mapToLong(Probe::seed).distinct().toArray()) {
        var engine = new GameEngine(seed);
        GameStepResult step = engine.stepHanchan();
        while (index < expected.probes().size() && expected.probes().get(index).seed() == seed) {
          while (step instanceof GameStepResult.RoundSettled) step = engine.stepHanchan();
          Assert.assertTrue(step instanceof GameStepResult.AwaitingDecisions);
          var selections = new EngineSelectionBuffer();
          for (var point : ((GameStepResult.AwaitingDecisions) step).decisions()) {
            Probe probe = expected.probes().get(index++);
            Assert.assertEquals(point.id(), probe.decisionId());
            var builder =
                DecisionBatchBuilder.inference(
                    1, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
            builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
            try (var scope = model.getNDManager().newSubManager()) {
              var batch = DecisionBatchTransfer.transferTrainingToDevice(scope, builder.build());
              var scores =
                  network.forwardPolicy(
                      new ParameterStore(scope, false), batch, false, new PairList<>());
              assertScores(scores.alternativeScores().toFloatArray(), probe.alternative());
              assertScores(scores.actionCandidateScores().toFloatArray(), probe.candidate());
              assertScores(scores.riichiGateScores().toFloatArray(), probe.riichi());
            }
            selections.add(point.id(), point.legalActions().getFirst());
          }
          step = engine.commitDecisions(selections);
        }
      }
      Assert.assertEquals(index, 32);
    }
  }

  private static void assertScores(float[] actual, float[] expected) {
    Assert.assertEquals(actual.length, expected.length);
    for (int index = 0; index < actual.length; index++)
      Assert.assertEquals(actual[index], expected[index], 1e-6f);
  }
}

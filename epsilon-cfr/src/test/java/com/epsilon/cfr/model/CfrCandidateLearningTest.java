package com.epsilon.cfr.model;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.network.NetworkDevices;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 出力層をゼロ初期化しても、続く更新で複数遷移の集約重みまで学習されることを検証する。 */
public class CfrCandidateLearningTest {
  @Test(groups = "native")
  public void regressionLearnsTransitionPoolingWithoutUnusedDiscardScorer() {
    DecisionContext context = multiTransitionContext();
    var config =
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
            EpsilonUtilityProfile.TENHOU);
    try (var model = Model.newInstance("cfr-candidate-learning", Device.cpu(), "PyTorch")) {
      model.getNDManager().getEngine().setRandomSeed(1337);
      var network = new CfrNetwork(config);
      model.setBlock(network);
      network.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
      var before = new LinkedHashMap<String, float[]>();
      for (var entry : network.getParameters()) {
        if (entry.getKey().contains("discardCandidateScorer"))
          before.put(entry.getKey(), entry.getValue().getArray().toFloatArray());
      }
      Assert.assertFalse(before.isEmpty());
      int[] actions =
          context.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
      double[] policy = new double[actions.length];
      Arrays.fill(policy, 1d / actions.length);
      var row =
          new CfrTrainingRow(
              new CfrSampleId(1, context.episodeId(), context.decisionId(), context.seat()),
              actions,
              0,
              policy,
              1,
              0,
              0,
              CfrObservationCodec.encodePayload(context));
      try (var learner =
          new CfrLearner(
              config, CfrModel.Role.REGRET, model, network, NetworkDevices.of(Device.cpu()))) {
        for (int update = 1; update <= 3; update++) learner.train(List.of(row), update);
        boolean changed =
            before.entrySet().stream()
                .anyMatch(
                    entry ->
                        !Arrays.equals(
                            entry.getValue(),
                            network.getParameters().get(entry.getKey()).getArray().toFloatArray()));
        Assert.assertTrue(
            changed, "CFR transition pooling must receive candidate regression gradients");
      }
    }
  }

  private static DecisionContext multiTransitionContext() {
    for (long seed = 770; seed < 790; seed++) {
      var engine = new GameEngine(seed);
      GameStepResult step = engine.stepHanchan();
      for (int boundary = 0;
          boundary < 200 && !(step instanceof GameStepResult.HanchanEnded);
          boundary++) {
        if (step instanceof GameStepResult.RoundSettled) {
          step = engine.stepHanchan();
          continue;
        }
        var selections = new EngineSelectionBuffer();
        for (var point : ((GameStepResult.AwaitingDecisions) step).decisions()) {
          if (DecisionBatchBuilder.selectInferenceBucket(point.legalActions())
                  .actionTransitionCapacity()
              > 1) {
            return new DecisionContext(
                seed,
                point.id(),
                point.player(),
                engine.observation(point.player()),
                point.legalActions());
          }
          selections.add(point.id(), point.legalActions().getFirst());
        }
        step = engine.commitDecisions(selections);
      }
    }
    throw new AssertionError("No multi-transition decision found in fixed game seeds");
  }
}

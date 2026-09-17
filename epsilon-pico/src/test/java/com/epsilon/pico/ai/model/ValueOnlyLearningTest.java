package com.epsilon.pico.ai.model;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionPretrainer;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ValueOnlyLearningTest {
  /** 全ブロックを学習可能にしても価値損失の勾配が方策へ流れず、更新後も方策の重みが変わらないことを検証する。 */
  @Test(groups = "native")
  public void valueOnlyBackwardAndAdamUpdateLeaveAllActorWeightsUnchanged() throws Exception {
    DecisionHostBatch host = oneChoiceBatch();
    SettingsLoader config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.devices.learner", "cpu",
                "epsilon.decision.utilityProfile", "TOP",
                "epsilon.decision.weightDecay", "0.5",
                "epsilon.decision.pretrain.computePrecision", "FLOAT32",
                "epsilon.decision.pretrain.tensorTransfer", "DIRECT_BUFFER",
                "epsilon.decision.pretrain.optimizerBatchRows", "1",
                "epsilon.decision.pretrain.maximumDeviceBatchRows", "1",
                "epsilon.decision.pretrain.learningRate", "0.001"));
    try (Model model =
            NetworkFactory.createDecisionModel(Device.cpu(), false, 16, EpsilonUtilityProfile.TOP);
        var execution = new DecisionExecutionContext()) {
      EpsilonDecisionNetwork network = (EpsilonDecisionNetwork) model.getBlock();
      network.prepareForOnlineTraining();
      Set<Parameter> actor = new HashSet<>();
      for (var child : network.getChildren()) {
        if (child.getKey().endsWith("stateEncoder")
            || child.getKey().endsWith("policyStateReadout")
            || child.getKey().endsWith("policyHead")) {
          actor.addAll(child.getValue().getParameters().values());
        }
      }
      Assert.assertFalse(actor.isEmpty());
      Map<String, float[]> before = new LinkedHashMap<>();
      for (var parameter : network.getParameters()) {
        before.put(parameter.getKey(), parameter.getValue().getArray().toFloatArray());
      }
      try (var manager = model.getNDManager().newSubManager()) {
        var input =
            DecisionBatchTransfer.transferStateToDevice(
                manager,
                host.sliceRows(0, host.size()),
                null,
                DecisionTensorTransfer.DIRECT_BUFFER);
        var runtime = EpsilonTileRelationEncoder.createRuntimeParameters(manager);
        try (var collector = manager.getEngine().newGradientCollector()) {
          var output =
              network.forwardValueWithDiagnostics(
                  new ParameterStore(manager, false), input, true, runtime);
          NDArray loss =
              EpsilonDecisionLoss.computeValueLoss(
                  output.valueLogits(),
                  manager.create(new float[] {1}),
                  manager.create(new float[] {1}),
                  EpsilonUtilityProfile.TOP);
          collector.backward(loss);
        }
        double valueGradient = 0;
        for (var parameter : network.getParameters()) {
          try (NDArray gradient = parameter.getValue().getArray().getGradient()) {
            float norm = gradient.abs().sum().getFloat();
            if (actor.contains(parameter.getValue())) {
              Assert.assertEquals(norm, 0f, parameter.getKey());
            } else {
              valueGradient += norm;
            }
            gradient.fillI(0);
          }
        }
        Assert.assertTrue(valueGradient > 0, "Value backward must produce a real gradient");
      }
      try (var trainer = EpsilonDecisionPretrainer.open(model, config, execution)) {
        trainer.trainEpoch(List.of(host, host));
      }
      boolean valueChanged = false;
      for (var parameter : network.getParameters()) {
        float[] actual = parameter.getValue().getArray().toFloatArray();
        float[] original = before.get(parameter.getKey());
        if (actor.contains(parameter.getValue())) {
          Assert.assertEquals(actual, original, parameter.getKey());
        } else {
          valueChanged |= !java.util.Arrays.equals(actual, original);
        }
      }
      Assert.assertTrue(valueChanged, "Value optimizer must update its network parameters");
    }
  }

  private static DecisionHostBatch oneChoiceBatch() {
    GameState state = new GameState(7);
    state.startRound(0, 0, 0, 0);
    state.dealInitialHands();
    int tile = 0;
    while (state.hand(0).count(tile) == 0) tile++;
    var actions = List.of(Action.dahai(tile));
    var bucket = DecisionBatchBuilder.minimumDetachedBucket(state, 0, actions, state.publicState());
    var builder = DecisionBatchBuilder.training(1, bucket);
    builder.addDetachedTrainingRow(
        state,
        0,
        actions,
        state.publicState(),
        DecisionBoundaryContext.uniform(),
        0,
        new float[] {1},
        new float[] {1},
        1,
        0,
        0,
        1);
    return builder.build();
  }
}

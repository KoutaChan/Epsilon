package com.epsilon.pico.ai.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.ai.decision.DecisionBranchGate;
import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.policy.DecisionAlternative;
import com.epsilon.pico.ai.decision.policy.DecisionPolicyScores;
import com.epsilon.pico.ai.decision.training.DecisionOnlineLossConfig;
import java.util.ArrayList;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class BranchGateLossTest {
  @DataProvider
  public Object[][] modes() {
    return new Object[][] {
      {false, DecisionBranchTarget.KYUSHU_ONLY, .125f},
      {true, DecisionBranchTarget.KYUSHU_ONLY, 0f},
      {true, DecisionBranchTarget.completed(DecisionBranchGate.KYUSHU, .5f, 1, -1), -.5f}
    };
  }

  @Test(groups = "native", dataProvider = "modes")
  public void configControlsOnlyTheComparedGate(
      boolean enabled, DecisionBranchTarget target, float expected) {
    GameState state = new GameState(7);
    state.startRound(0, 0, 0, 0);
    state.dealInitialHands();
    ArrayList<Action> actions = new ArrayList<>();
    actions.add(Action.kyushuKyuhai());
    int[] tiles = state.hand(0).copyConcealedTileCounts();
    for (int tile = 0; tile < tiles.length && actions.size() < 3; tile++) {
      if (tiles[tile] > 0) {
        actions.add(Action.dahai(tile));
      }
    }
    var bucket = DecisionBatchBuilder.minimumDetachedBucket(state, 0, actions, state.publicState());
    var builder = DecisionBatchBuilder.training(1, bucket);
    float[] probability = {.5f, .25f, .25f};
    builder.addDetachedTrainingRow(
        state,
        0,
        actions,
        state.publicState(),
        DecisionBoundaryContext.uniform(),
        1,
        probability,
        probability,
        0,
        .25f,
        1,
        1);
    builder.writeBranchTarget(0, target);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
        var constants =
            new EpsilonDecisionHlGauss.DeviceConstants(manager, EpsilonUtilityProfile.TENHOU)) {
      var input = DecisionBatchTransfer.transferTrainingToDevice(manager, builder.build());
      NDArray gates = manager.zeros(new Shape(1, DecisionAlternative.NETWORK_SIZE));
      NDArray leaves = manager.zeros(new Shape(1, bucket.legalActionCapacity()));
      NDArray value = manager.zeros(new Shape(1, EpsilonDecisionHlGauss.BIN_COUNT));
      gates.setRequiresGradient(true);
      leaves.setRequiresGradient(true);
      value.setRequiresGradient(true);
      var output =
          new EpsilonDecisionOutput(
              new DecisionPolicyScores(
                  gates, leaves, manager.zeros(new Shape(1, bucket.legalActionCapacity()))),
              value);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        var loss =
            EpsilonDecisionLoss.computeOnlineTrainingLoss(
                output,
                input,
                DecisionOnlineLossConfig.create().withBranchComparison(enabled),
                true,
                true,
                EpsilonUtilityProfile.TENHOU,
                constants);
        collector.backward(loss.policyGradientLoss());
      }
      Assert.assertEquals(
          gates.getGradient().getFloat(0, DecisionAlternative.KYUSHU.networkIndex()),
          expected,
          1e-6f);
      Assert.assertEquals(leaves.getGradient().getFloat(0, 1), -.125f, 1e-6f);
      Assert.assertEquals(leaves.getGradient().getFloat(0, 2), .125f, 1e-6f);
      Assert.assertEquals(value.getGradient().abs().sum().getFloat(), 0f);
    }
  }
}

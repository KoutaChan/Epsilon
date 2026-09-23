package com.epsilon.major.ai.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.major.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.major.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.major.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.major.ai.decision.policy.DecisionAlternative;
import com.epsilon.major.ai.decision.policy.DecisionPolicyScores;
import com.epsilon.major.ai.decision.training.DecisionOnlineLossConfig;
import java.util.ArrayList;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class SelectedActionLossTest {
  @DataProvider
  public Object[][] selectedActions() {
    return new Object[][] {
      {0, 0.25f, false},
      {0, -0.25f, false},
      {1, 0.25f, false},
      {1, -0.25f, false},
      {0, 0.25f, true},
      {0, -0.25f, true},
      {1, 0.25f, true},
      {1, -0.25f, true}
    };
  }

  /** 重なり方策のアドバンテージ、探索補正、PPO クリッピングから求めた損失と勾配を手計算と比較する。 */
  @Test(groups = "native", dataProvider = "selectedActions")
  public void selectedSurrogateAndGradientUseOverlapPolicy(
      int selected, float advantage, boolean onPolicy) {
    float[] behavior = onPolicy ? new float[] {0.4f, 0.6f} : new float[] {0.2f, 0.8f};
    float[] rollout = {0.4f, 0.6f};
    GameState state = new GameState(7L);
    state.startRound(0, 0, 0, 0);
    state.dealInitialHands();
    ArrayList<Action> actions = new ArrayList<>();
    int[] counts = state.hand(0).copyConcealedTileCounts();
    for (int tile = 0; tile < counts.length && actions.size() < 2; tile++) {
      if (counts[tile] > 0) {
        actions.add(Action.dahai(tile));
      }
    }
    DecisionBucket bucket =
        DecisionBatchBuilder.minimumDetachedBucket(state, 0, actions, state.publicState());
    DecisionBatchBuilder builder = DecisionBatchBuilder.training(1, bucket);
    builder.addDetachedTrainingRow(
        state,
        0,
        actions,
        state.publicState(),
        DecisionBoundaryContext.uniform(),
        selected,
        behavior,
        rollout,
        0.3f,
        advantage,
        1,
        1);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
        EpsilonDecisionHlGauss.DeviceConstants constants =
            new EpsilonDecisionHlGauss.DeviceConstants(manager, EpsilonUtilityProfile.TENHOU)) {
      DecisionDeviceBatch input =
          DecisionBatchTransfer.transferTrainingToDevice(manager, builder.build());
      NDArray logits = manager.zeros(new Shape(1, bucket.legalActionCapacity()));
      logits.setRequiresGradient(true);
      NDArray value = manager.zeros(new Shape(1, EpsilonDecisionHlGauss.BIN_COUNT));
      value.setRequiresGradient(true);
      EpsilonDecisionOutput output =
          new EpsilonDecisionOutput(
              new DecisionPolicyScores(
                  manager.zeros(new Shape(1, DecisionAlternative.NETWORK_SIZE)),
                  logits,
                  manager.zeros(new Shape(1, bucket.legalActionCapacity()))),
              value);
      double firstOverlap = Math.min(behavior[0], 0.9 * rollout[0] + 0.1 * behavior[0]);
      double secondOverlap = Math.min(behavior[1], 0.9 * rollout[1] + 0.1 * behavior[1]);
      double overlapSum = firstOverlap + secondOverlap;
      double firstOld = firstOverlap / overlapSum;
      double secondOld = secondOverlap / overlapSum;
      double firstUpdated = firstOld * 0.5 / rollout[0];
      double secondUpdated = secondOld * 0.5 / rollout[1];
      double updatedSum = firstUpdated + secondUpdated;
      double selectedUpdated =
          selected == 0 ? firstUpdated / updatedSum : secondUpdated / updatedSum;
      double selectedOld = selected == 0 ? firstOld : secondOld;
      double ratio = selectedUpdated / selectedOld;
      double credit = (selected == 0 ? firstOverlap : secondOverlap) / behavior[selected];
      double clipped = Math.max(0.8, Math.min(1.2, ratio));
      double expected = -credit * Math.min(ratio * advantage, clipped * advantage);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        var loss =
            EpsilonDecisionLoss.computeOnlineTrainingLoss(
                output,
                input,
                new DecisionOnlineLossConfig(0.2f, 0.1f, 0, false),
                true,
                true,
                EpsilonUtilityProfile.TENHOU,
                constants);
        Assert.assertEquals(loss.policyGradientLoss().getFloat(), (float) expected, 2e-6f);
        collector.backward(loss.policyGradientLoss());
      }
      boolean clipActive = advantage > 0 && ratio > 1.2 || advantage < 0 && ratio < 0.8;
      double chosenGradient = clipActive ? 0 : -credit * advantage * ratio * (1 - selectedUpdated);
      Assert.assertEquals(
          logits.getGradient().getFloat(0, selected), (float) chosenGradient, 2e-6f);
      Assert.assertEquals(
          logits.getGradient().getFloat(0, 1 - selected), (float) -chosenGradient, 2e-6f);
      Assert.assertEquals(
          value.getGradient().abs().sum().getFloat(),
          0f,
          "Actor backward must not update Value logits");
    }
  }

  /** 全行動の利得が等しいとき、探索補正後も状態価値ベースラインの勾配が消えることを確認する。 */
  @Test(groups = "native")
  public void constantAdvantageHasZeroPolicyGradientAcrossBehaviorActions() {
    float[] behavior = {0.2f, 0.8f};
    float[] rollout = {0.4f, 0.6f};
    GameState state = new GameState(17L);
    state.startRound(0, 0, 0, 0);
    state.dealInitialHands();
    ArrayList<Action> actions = new ArrayList<>();
    int[] counts = state.hand(0).copyConcealedTileCounts();
    for (int tile = 0; tile < counts.length && actions.size() < 2; tile++) {
      if (counts[tile] > 0) {
        actions.add(Action.dahai(tile));
      }
    }
    DecisionBucket bucket =
        DecisionBatchBuilder.minimumDetachedBucket(state, 0, actions, state.publicState());
    DecisionBatchBuilder builder = DecisionBatchBuilder.training(2, bucket);
    for (int chosen = 0; chosen < 2; chosen++) {
      builder.addDetachedTrainingRow(
          state,
          0,
          actions,
          state.publicState(),
          DecisionBoundaryContext.uniform(),
          chosen,
          behavior,
          rollout,
          0.3f,
          0.25f,
          behavior[chosen],
          1.0f);
    }
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
        EpsilonDecisionHlGauss.DeviceConstants constants =
            new EpsilonDecisionHlGauss.DeviceConstants(manager, EpsilonUtilityProfile.TENHOU)) {
      DecisionDeviceBatch input =
          DecisionBatchTransfer.transferTrainingToDevice(manager, builder.build());
      NDArray logits = manager.zeros(new Shape(2, bucket.legalActionCapacity()));
      logits.setRequiresGradient(true);
      EpsilonDecisionOutput output =
          new EpsilonDecisionOutput(
              new DecisionPolicyScores(
                  manager.zeros(new Shape(2, DecisionAlternative.NETWORK_SIZE)),
                  logits,
                  manager.zeros(new Shape(2, bucket.legalActionCapacity()))),
              manager.zeros(new Shape(2, EpsilonDecisionHlGauss.BIN_COUNT)));
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        var loss =
            EpsilonDecisionLoss.computeOnlineTrainingLoss(
                output,
                input,
                new DecisionOnlineLossConfig(0.9f, 0.1f, 0, false),
                true,
                true,
                EpsilonUtilityProfile.TENHOU,
                constants);
        collector.backward(loss.policyGradientLoss());
      }
      Assert.assertEquals(
          logits.getGradient().sum(new int[] {0}).abs().sum().getFloat(), 0.0f, 1e-6f);
    }
  }
}

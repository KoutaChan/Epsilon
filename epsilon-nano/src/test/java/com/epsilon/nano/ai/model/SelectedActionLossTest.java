package com.epsilon.nano.ai.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.nano.ai.decision.policy.DecisionAlternative;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyScores;
import com.epsilon.nano.ai.decision.training.DecisionOnlineLossConfig;
import java.util.ArrayList;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 選択行動に対する PPO のクリップ付き損失と勾配を、式から計算した値と比較する。 */
public class SelectedActionLossTest {
  @DataProvider
  public Object[][] selectedActions() {
    ArrayList<Object[]> cases = new ArrayList<>();
    for (boolean onPolicy : new boolean[] {false, true}) {
      for (float alpha : new float[] {0.1f, 1.0f}) {
        for (int selected = 0; selected < 2; selected++) {
          for (float advantage : new float[] {0.25f, -0.25f}) {
            cases.add(new Object[] {onPolicy, alpha, selected, advantage});
          }
        }
      }
    }
    return cases.toArray(new Object[0][]);
  }

  /** 重なり方策の選択確率、探索分の寄与、PPO のクリッピングから求めた損失と勾配を、手計算と比較する。 */
  @Test(groups = "native", dataProvider = "selectedActions")
  public void selectedSurrogateAndGradientUseOverlapPolicy(
      boolean onPolicy, float alpha, int selected, float advantage) {
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
      double[] overlap = new double[2];
      double[] betaCurrentUnnormalized = new double[2];
      double overlapSum = 0;
      double betaCurrentSum = 0;
      for (int action = 0; action < 2; action++) {
        overlap[action] =
            Math.min(behavior[action], (1.0 - alpha) * rollout[action] + alpha * behavior[action]);
        overlapSum += overlap[action];
        betaCurrentUnnormalized[action] = overlap[action] * 0.5 / rollout[action];
        betaCurrentSum += betaCurrentUnnormalized[action];
      }
      double betaCurrent = betaCurrentUnnormalized[selected] / betaCurrentSum;
      double betaRollout = overlap[selected] / overlapSum;
      double ratio = betaCurrent / betaRollout;
      double credit = overlap[selected] / behavior[selected];
      double clipped = Math.max(0.8, Math.min(1.2, ratio));
      double expected = -credit * Math.min(ratio * advantage, clipped * advantage);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        var loss =
            EpsilonDecisionLoss.computeOnlineTrainingLoss(
                output,
                input,
                new DecisionOnlineLossConfig(0.2f, alpha, 0, false),
                true,
                true,
                EpsilonUtilityProfile.TENHOU,
                constants);
        Assert.assertEquals(loss.policyGradientLoss().getFloat(), (float) expected, 2e-6f);
        collector.backward(loss.policyGradientLoss());
      }
      boolean clipActive = advantage > 0 && ratio > 1.2 || advantage < 0 && ratio < 0.8;
      double chosenGradient = clipActive ? 0 : -credit * advantage * ratio * (1.0 - betaCurrent);
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

  /** 二行動を実際の収集確率で平均したとき、行動に依存しないAdvantageの勾配は相殺される。 */
  @Test(groups = "native")
  public void constantAdvantageBaselineHasZeroExpectedGradient() {
    float[] behavior = {0.2f, 0.8f};
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
    DecisionBatchBuilder builder = DecisionBatchBuilder.training(2, bucket);
    for (int selected = 0; selected < 2; selected++) {
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
          0.25f,
          1.0f,
          behavior[selected]);
    }
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
        EpsilonDecisionHlGauss.DeviceConstants constants =
            new EpsilonDecisionHlGauss.DeviceConstants(manager, EpsilonUtilityProfile.TENHOU)) {
      DecisionDeviceBatch input =
          DecisionBatchTransfer.transferTrainingToDevice(manager, builder.build());
      int capacity = bucket.legalActionCapacity();
      float[] logitsData = new float[2 * capacity];
      for (int row = 0; row < 2; row++) {
        logitsData[row * capacity] = (float) Math.log(rollout[0]);
        logitsData[row * capacity + 1] = (float) Math.log(rollout[1]);
      }
      NDArray logits = manager.create(logitsData).reshape(2, capacity);
      logits.setRequiresGradient(true);
      EpsilonDecisionOutput output =
          new EpsilonDecisionOutput(
              new DecisionPolicyScores(
                  manager.zeros(new Shape(2, DecisionAlternative.NETWORK_SIZE)),
                  logits,
                  manager.zeros(new Shape(2, capacity))),
              manager.zeros(new Shape(2, EpsilonDecisionHlGauss.BIN_COUNT)));
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
        Assert.assertEquals(loss.policyGradientLoss().getFloat(), -0.82f * 0.25f, 2e-6f);
        collector.backward(loss.policyGradientLoss());
      }
      for (int action = 0; action < 2; action++) {
        float aggregateGradient =
            logits.getGradient().getFloat(0, action) + logits.getGradient().getFloat(1, action);
        Assert.assertEquals(aggregateGradient, 0.0f, 2e-6f);
      }
    }
  }
}

package com.epsilon.pico.ai.decision.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.pico.ai.network.NetworkDevices;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.runtime.DecisionExecutionContext;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CheckpointOptimizerResumeTest {
  /** Pico のパラメータと方策・価値関数それぞれの AdamW 状態を保存・復元し、次の更新結果も一致することを検証する。 */
  @Test(groups = "native")
  public void restoredActorAndValueMomentsMatchUninterruptedSecondStep() throws Exception {
    Path directory = Files.createTempDirectory("pico-optimizer-resume-");
    try (var execution = new DecisionExecutionContext();
        Model continuous =
            NetworkFactory.createDecisionModel(Device.cpu(), false, 16, EpsilonUtilityProfile.TOP);
        var parallel =
            EpsilonDecisionDataParallel.openOnline(
                continuous, NetworkDevices.of(Device.cpu()), execution)) {
      Optimizer actor = optimizer();
      Optimizer value = optimizer();
      gradientStep(continuous, parallel, actor, value, 1);
      Path checkpoint = directory.resolve("checkpoint");
      EpsilonDecisionCheckpointManager.save(continuous, checkpoint, 1, 1, 4);
      actor.saveState(directory.resolve("actor.state"));
      value.saveState(directory.resolve("value.state"));

      try (Model restored = EpsilonDecisionCheckpointManager.load(checkpoint, Device.cpu(), false);
          var resumed =
              EpsilonDecisionDataParallel.openOnline(
                  restored, NetworkDevices.of(Device.cpu()), execution)) {
        assertParametersEqual(restored, continuous);
        Optimizer restoredActor = optimizer();
        Optimizer restoredValue = optimizer();
        restoredActor.loadState(restored.getNDManager(), directory.resolve("actor.state"));
        restoredValue.loadState(restored.getNDManager(), directory.resolve("value.state"));
        gradientStep(continuous, parallel, actor, value, 2);
        gradientStep(restored, resumed, restoredActor, restoredValue, 2);
        assertParametersEqual(restored, continuous);
      }
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  private static Optimizer optimizer() {
    return Optimizer.adamW()
        .optLearningRateTracker(Tracker.fixed(2e-4f))
        .optWeightDecays(1e-4f)
        .optClipGrad(0.5f)
        .build();
  }

  private static void gradientStep(
      Model model, EpsilonDecisionDataParallel parallel, Optimizer actor, Optimizer value, int step)
      throws Exception {
    try (var manager = model.getNDManager().newSubManager();
        var collector = manager.getEngine().newGradientCollector()) {
      NDArray loss = manager.create(0f);
      for (var parameter : model.getBlock().getParameters()) {
        if (parameter.getValue().requiresGradient()) {
          float scale = (1 + Math.floorMod(parameter.getKey().hashCode() + step * 13, 31)) * 1e-5f;
          loss = loss.add(parameter.getValue().getArray().sum().mul(step == 1 ? scale : -scale));
        }
      }
      collector.backward(loss);
    }
    parallel.applyFlattenedOnlineOptimizerSteps(actor, 1.0, value, 1.0);
  }

  private static void assertParametersEqual(Model actual, Model expected) {
    var parameters = actual.getBlock().getParameters();
    Assert.assertEquals(parameters.keys(), expected.getBlock().getParameters().keys());
    for (var entry : expected.getBlock().getParameters()) {
      float[] wanted = entry.getValue().getArray().toFloatArray();
      float[] got = parameters.get(entry.getKey()).getArray().toFloatArray();
      Assert.assertEquals(got.length, wanted.length, entry.getKey());
      for (int index = 0; index < wanted.length; index++)
        Assert.assertEquals(got[index], wanted[index], 2e-7f, entry.getKey());
    }
  }
}

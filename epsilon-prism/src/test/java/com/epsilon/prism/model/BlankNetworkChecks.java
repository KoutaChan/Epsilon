package com.epsilon.prism.model;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import java.util.Arrays;
import org.testng.Assert;

/** 同じ種から始めた既定の幹で、初期化と最初の更新が全ての重みに与える結果を比べる。 */
final class BlankNetworkChecks {
  private BlankNetworkChecks() {}

  /** 全候補に目標 1 を与え、Adam で 1 回更新する。 */
  static void update(Model model, HostBatch batch, Precision precision) {
    PrismNetwork network = (PrismNetwork) model.getBlock();
    Optimizer optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.001f)).build();
    try (NDManager scope = model.getNDManager().newSubManager()) {
      NetworkInputs inputs = NetworkInputs.of(scope, PinnedUploads.on(scope), batch);
      boolean[] measured = new boolean[batch.plan().count(Axis.ACTION)];
      Arrays.fill(measured, true);
      try (GradientCollector collector = scope.getEngine().newGradientCollector()) {
        var outputs =
            network.forward(
                new ParameterStore(scope, false), inputs, Pass.training(precision, measured, null));
        collector.backward(outputs.scores().sub(1f).square().mean());
      }
      for (var pair : network.getParameters()) {
        NDArray weights = pair.getValue().getArray();
        if (weights.hasGradient()) {
          try (NDArray gradient = weights.getGradient()) {
            optimizer.update(pair.getValue().getId(), weights, gradient);
          }
        }
      }
      NDArray output =
          network.getParameters().stream()
              .filter(p -> p.getKey().endsWith("discardValue_output"))
              .findFirst()
              .orElseThrow()
              .getValue()
              .getArray();
      Assert.assertTrue(output.abs().max().getFloat() > 0f);
    }
  }
}

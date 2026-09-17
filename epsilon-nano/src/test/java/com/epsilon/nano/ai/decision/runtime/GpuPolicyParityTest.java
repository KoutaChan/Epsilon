package com.epsilon.nano.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionInferenceSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 同一チェックポイントの CPU 推論と GPU 推論で方策が一致することを検証する。 */
public class GpuPolicyParityTest {
  @DataProvider
  public Object[][] precisions() {
    return new Object[][] {{"FLOAT32", 0.0002f}, {"BFLOAT16", 0.02f}};
  }

  /** 同じ保存重みをCPU と GPU の Fusion 実行で評価し、全合法手の確率と価値を比較する。 */
  @Test(groups = "rocm", dataProvider = "precisions")
  public void fusedGpuPredictionsAgreeWithCpu(String precision, float tolerance) throws Exception {
    Engine.getEngine("PyTorch").setRandomSeed(391);
    Path directory = Files.createTempDirectory("epsilon-nano-gpu-parity-");
    var config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.decision.inference.computePrecision",
                precision,
                "epsilon.decision.inference.maxBatch",
                "16",
                "epsilon.decision.inference.multiTransitionMaxBatch",
                "16"));
    var settings = config.bind(DecisionInferenceSettings.class);
    var fusion = config.bind(DecisionInferenceFusionSettings.class);
    try {
      try (Model original =
          NetworkFactory.createDecisionModel(Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
        EpsilonDecisionCheckpointManager.saveInitial(original, directory);
      }
      try (Model cpu =
              EpsilonDecisionCheckpointManager.loadForInference(directory, Device.cpu(), settings);
          Model gpu =
              EpsilonDecisionCheckpointManager.loadForInference(directory, Device.gpu(), settings);
          var reference = EpsilonDecisionInferenceServer.forFrozenModel(cpu, 16, settings, fusion);
          var actual = EpsilonDecisionInferenceServer.forFrozenModel(gpu, 16, settings, fusion)) {
        for (long seed = 700; seed < 716; seed++) {
          var engine = new GameEngine(seed);
          var boundary = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
          var point = boundary.decisions().getFirst();
          var builder =
              DecisionBatchBuilder.inference(
                  1, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
          builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
          var host = builder.build();
          var expected = reference.evaluateBatch(host).getFirst();
          var observed = actual.evaluateBatch(host).getFirst();
          float[] want = expected.policyProbabilities(), got = observed.policyProbabilities();
          Assert.assertEquals(got.length, want.length);
          double sum = 0;
          for (int i = 0; i < want.length; i++) {
            Assert.assertTrue(Float.isFinite(got[i]) && got[i] >= 0, "finite probability");
            Assert.assertEquals(got[i], want[i], tolerance, "seed=" + seed + " action=" + i);
            sum += got[i];
          }
          Assert.assertEquals(sum, 1, 0.00001);
          Assert.assertEquals(observed.valueUtility(), expected.valueUtility(), tolerance);
        }
      }
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }
}

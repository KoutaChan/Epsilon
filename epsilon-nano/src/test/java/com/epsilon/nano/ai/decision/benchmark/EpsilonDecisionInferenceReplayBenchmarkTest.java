package com.epsilon.nano.ai.decision.benchmark;

import static org.testng.Assert.assertEquals;

import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayBenchmark.Classification;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayBenchmark.ConfidenceKind;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayBenchmark.Mode;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayBenchmark.ModeSummary;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayBenchmark.Verdict;
import com.google.gson.Gson;
import java.util.List;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 推論性能測定の容量区分ごとの網羅性と、試行回数に応じた結果判定を検証する。 */
public final class EpsilonDecisionInferenceReplayBenchmarkTest {

  @DataProvider
  public Object[][] boundaryMeasurements() {
    return new Object[][] {
      {55.0, 65.0, 45.0, 55.0, 35.0, 45.0, Classification.MODEL_OR_KERNEL_LIMIT},
      {110.0, 130.0, 70.0, 90.0, 60.0, 80.0, Classification.TRANSFER_STREAM_OR_DJL_LIMIT},
      {140.0, 160.0, 110.0, 130.0, 70.0, 90.0, Classification.JAVA_COMPOSE_OR_SLOT_LIFETIME_LIMIT},
      {140.0, 160.0, 110.0, 130.0, 105.0, 115.0, Classification.INFERENCE_CAPACITY_SUFFICIENT},
      {90.0, 130.0, 95.0, 115.0, 90.0, 110.0, Classification.INCONCLUSIVE}
    };
  }

  @Test(dataProvider = "boundaryMeasurements")
  public void classifiesCapacityFromEachMeasuredBoundary(
      double computeLower,
      double computeUpper,
      double pipelineLower,
      double pipelineUpper,
      double fullLower,
      double fullUpper,
      Classification expected) {
    Verdict verdict =
        EpsilonDecisionInferenceReplayBenchmark.verdict(
            List.of(
                confirmed(Mode.COMPUTE_ONLY, computeLower, computeUpper),
                confirmed(Mode.GPU_PIPELINE, pipelineLower, pipelineUpper),
                confirmed(Mode.FULL_INFERENCE, fullLower, fullUpper)),
            3,
            100.0);

    assertEquals(verdict.classification(), expected);
  }

  @Test
  public void singleRepetitionKeepsPreliminaryClassificationInJson() {
    Verdict verdict =
        EpsilonDecisionInferenceReplayBenchmark.verdict(
            List.of(
                preliminary(Mode.COMPUTE_ONLY, 150.0),
                preliminary(Mode.GPU_PIPELINE, 120.0),
                preliminary(Mode.FULL_INFERENCE, 110.0)),
            1,
            100.0);

    assertEquals(
        verdict.classification(), Classification.PRELIMINARY_INFERENCE_CAPACITY_SUFFICIENT);
    assertEquals(
        new Gson().toJsonTree(verdict).getAsJsonObject().get("classification").getAsString(),
        "PRELIMINARY_INFERENCE_CAPACITY_SUFFICIENT");
  }

  @Test
  public void missingBoundaryRemainsIncomplete() {
    Verdict verdict =
        EpsilonDecisionInferenceReplayBenchmark.verdict(
            List.of(confirmed(Mode.COMPUTE_ONLY, 140.0, 160.0)), 3, 100.0);

    assertEquals(verdict.classification(), Classification.INCOMPLETE);
    assertEquals(verdict.bestMixedByMode().size(), 1);
  }

  private static ModeSummary confirmed(Mode mode, double lower, double upper) {
    return new ModeSummary(
        mode, 2048, 3, Math.sqrt(lower * upper), lower, upper, ConfidenceKind.BONFERRONI_GLOBAL_95);
  }

  private static ModeSummary preliminary(Mode mode, double rowsPerSecond) {
    return new ModeSummary(
        mode, 2048, 1, rowsPerSecond, rowsPerSecond, rowsPerSecond, ConfidenceKind.PRELIMINARY);
  }
}

package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.config.settings.DecisionPretrainValidationSettings;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** 事前学習のチェックポイントについて、対戦評価での採用判定とは独立して、検証データで評価できる条件を満たすかを調べる。 */
final class EpsilonDecisionPretrainValidationAssessment {

  private EpsilonDecisionPretrainValidationAssessment() {}

  static Result evaluate(
      EpsilonDecisionPretrainValidator.ValidationMetrics metrics,
      DecisionPretrainValidationSettings settings) {
    ArrayList<String> failures = new ArrayList<>();
    if (metrics.samples() < settings.minSamples()) {
      failures.add("SAMPLES=" + metrics.samples() + "<" + settings.minSamples());
    }
    if (metrics.policySamples() < settings.minSamples()) {
      failures.add("POLICY_SAMPLES=" + metrics.policySamples() + "<" + settings.minSamples());
    }
    requireFinite(metrics, failures);

    float policyImprovement = metrics.policyUniformNll() - metrics.nll();
    float valueMseImprovement = metrics.valueConstantMse() - metrics.valueMse();
    String diagnostics =
        String.format(
            Locale.ROOT,
            "policyNllImprovement=%.9g;valueMseImprovement=%.9g;"
                + "valueTeacherMse=%.9g;grpBoundaryRps=%.9g;"
                + "grpBoundaryOutcomeNll=%.9g;boundarySamples=%d;"
                + "policyStateCenteredRms=%.9g;"
                + "valueStateCenteredRms=%.9g;valueUtilityCenteredRms=%.9g",
            policyImprovement,
            valueMseImprovement,
            metrics.valueTeacherMse(),
            metrics.grpBoundaryRps(),
            metrics.grpBoundaryOutcomeNll(),
            metrics.boundarySamples(),
            metrics.policyStateCenteredRms(),
            metrics.valueStateCenteredRms(),
            metrics.valueUtilityCenteredRms());
    return failures.isEmpty()
        ? new Result(true, "USABLE;" + diagnostics)
        : new Result(false, "INVALID;" + String.join(",", failures) + ";" + diagnostics);
  }

  private static void requireFinite(
      EpsilonDecisionPretrainValidator.ValidationMetrics metrics, List<String> failures) {
    float[] values = {
      metrics.top1(),
      metrics.dahaiTop1(),
      metrics.riichiTop1(),
      metrics.reactionTop1(),
      metrics.postCallDahaiTop1(),
      metrics.nll(),
      metrics.postCallDahaiNll(),
      metrics.valueTeacherBias(),
      metrics.valueTeacherMse(),
      metrics.valueBias(),
      metrics.valueMse(),
      metrics.grpBoundaryOutcomeNll(),
      metrics.grpBoundaryRps(),
      metrics.policyUniformNll(),
      metrics.valueConstantMse(),
      metrics.policyStateCenteredRms(),
      metrics.valueStateCenteredRms(),
      metrics.valueUtilityCenteredRms()
    };
    for (float value : values) {
      if (!Float.isFinite(value)) {
        failures.add("NON_FINITE_METRIC");
        return;
      }
    }
  }

  record Result(boolean usable, String summary) {}
}

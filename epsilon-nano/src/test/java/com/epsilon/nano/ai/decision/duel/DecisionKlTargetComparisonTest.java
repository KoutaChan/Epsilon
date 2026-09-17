package com.epsilon.nano.ai.decision.duel;

import com.epsilon.ai.decision.duel.EpsilonDecisionFixedSampleConfidenceInterval;
import com.google.gson.Gson;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** KL 制御の比較結果を、差の区間推定がゼロを含むかに基づいて判定することを検証する。 */
public class DecisionKlTargetComparisonTest {
  @DataProvider
  public Object[][] confidenceBounds() {
    return new Object[][] {
      {0.01, 0.1, DecisionKlTargetComparison.Decision.LOWER_TARGET_BETTER},
      {-0.1, -0.01, DecisionKlTargetComparison.Decision.CURRENT_TARGET_BETTER},
      {-0.1, 0.1, DecisionKlTargetComparison.Decision.UNRESOLVED},
      {0.0, 0.1, DecisionKlTargetComparison.Decision.UNRESOLVED},
      {-0.1, 0.0, DecisionKlTargetComparison.Decision.UNRESOLVED}
    };
  }

  @Test(dataProvider = "confidenceBounds")
  public void comparisonDecidesOnlyWhenIntervalExcludesZero(
      double lower, double upper, DecisionKlTargetComparison.Decision expected) {
    var interval =
        new EpsilonDecisionFixedSampleConfidenceInterval.Estimate(
            100, (lower + upper) / 2, 0.01, lower, upper);
    var actual = DecisionKlTargetComparison.decide(interval);
    Assert.assertEquals(actual, expected);
    var gson = new Gson();
    var report =
        new DecisionKlTargetComparison.Result(
            "kl-target-paired-comparison-v1",
            7,
            100,
            1,
            0.05,
            0.025,
            interval.mean(),
            interval.standardError(),
            lower,
            upper,
            actual,
            "FIXED_LOOK_GAUSSIAN_APPROXIMATION",
            "a",
            "b",
            "opponent");
    var restored = gson.fromJson(gson.toJson(report), DecisionKlTargetComparison.Result.class);
    Assert.assertEquals(restored, report);
  }
}

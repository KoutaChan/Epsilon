package com.epsilon.pico.ai.decision.arena;

import com.epsilon.config.settings.DecisionFullSupportSettings.AdaptiveExplorationSettings;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.pico.ai.decision.arena.DecisionAdaptiveExploration.Kind;
import com.epsilon.pico.ai.decision.arena.DecisionAdaptiveExploration.Lookup;
import com.epsilon.pico.ai.decision.arena.DecisionAdaptiveExploration.LookupSource;
import com.epsilon.pico.ai.decision.arena.DecisionAdaptiveExploration.Measurement;
import com.epsilon.pico.ai.decision.arena.DecisionAdaptiveExploration.Observation;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AdaptiveExplorationTelemetryTest {

  /** 校正済・未校正・合法手が一つの判断を混ぜても、確率距離と選択順位を各集計範囲へ正しく記録する。 */
  @Test
  public void aggregatesCalibratedUncalibratedAndForcedSelections() {
    var exploration = new DecisionAdaptiveExploration(AdaptiveExplorationSettings.normal());
    var session = exploration.beginMacro(1);
    var discard =
        new Observation(
            Kind.DISCARD,
            new Measurement(0.8f, 0.6f, 0.7f),
            new Lookup(0.25f, LookupSource.KIND),
            1.5f,
            2);
    var call =
        new Observation(
            Kind.CALL,
            new Measurement(0.4f, 0.2f, 0.3f),
            new Lookup(Float.NaN, LookupSource.NONE),
            1.0f,
            -1);
    var forced = new Observation(Kind.FORCED, null, null, 1.0f, -1);
    session.recordSelection(
        discard, new float[] {0.5f, 0.25f, 0.25f}, new float[] {0.25f, 0.5f, 0.25f}, 1);
    session.recordSelection(
        call, new float[] {0.25f, 0.25f, 0.5f}, new float[] {0.125f, 0.375f, 0.5f}, 2);
    session.recordSelection(forced, new float[] {1.0f}, new float[] {1.0f}, 0);
    session.recordLearningRole(discard, DecisionLearningRole.CAUSAL);
    session.recordLearningRole(call, DecisionLearningRole.PREEMPTED);
    session.recordLearningRole(forced, DecisionLearningRole.FORCED);

    var report = exploration.completeMacro(session);
    var overall = report.overall();
    double discardKl = 0.25 * Math.log(2.0);
    double callKl = 0.125 * Math.log(0.5) + 0.375 * Math.log(1.5);
    Assert.assertEquals(overall.selections(), 3L);
    Assert.assertEquals(overall.nonForcedSelections(), 2L);
    Assert.assertEquals(overall.calibratedSelections(), 1L);
    Assert.assertEquals(overall.kindCalibrations(), 1L);
    Assert.assertEquals(overall.uncalibratedSelections(), 1L);
    Assert.assertEquals(overall.nonTopOneSelections(), 1L);
    Assert.assertEquals(overall.forcedRoles(), 1L);
    Assert.assertEquals(overall.causalRoles(), 1L);
    Assert.assertEquals(overall.preemptedRoles(), 1L);
    Assert.assertEquals(overall.meanTotalVariation(), 0.1875, 0.0);
    Assert.assertEquals(overall.meanBehaviorToRolloutKl(), (discardKl + callKl) / 2.0, 1e-15);
    Assert.assertEquals(overall.meanSelectedRolloutProbability(), 0.375, 0.0);
    Assert.assertEquals(overall.meanSelectedBehaviorProbability(), 0.5, 0.0);
    Assert.assertEquals(overall.meanSelectedRolloutToBehaviorRatio(), 0.75, 0.0);
    Assert.assertEquals(overall.meanSelectedRolloutRank(), 1.5, 0.0);
    Assert.assertEquals(overall.meanTopOneMassChange(), -0.125, 0.0);
    Assert.assertEquals(overall.meanMultiplier(), 1.25, 0.0);

    var discardStats =
        report.byKind().stream()
            .filter(kind -> kind.kind() == Kind.DISCARD)
            .findFirst()
            .orElseThrow()
            .statistics();
    Assert.assertEquals(discardStats.selections(), 1L);
    Assert.assertEquals(discardStats.meanTotalVariation(), 0.25, 0.0);
    Assert.assertEquals(discardStats.meanBehaviorToRolloutKl(), discardKl, 1e-15);
    Assert.assertEquals(discardStats.meanSelectedRolloutRank(), 2.0, 0.0);
    Assert.assertEquals(report.byKindAndDecile().size(), 1);
    Assert.assertEquals(report.byKindAndDecile().getFirst().kind(), Kind.DISCARD);
    Assert.assertEquals(report.byKindAndDecile().getFirst().decile(), 2);
    Assert.assertEquals(report.byKindAndDecile().getFirst().statistics(), discardStats);
  }

  /** 最大確率が同率なら先頭の候補を選び、確率が0の項の KL 計算にだけ既定の下限値を使う。 */
  @Test
  public void retainsTopTieAndZeroProbabilitySemantics() {
    var exploration = new DecisionAdaptiveExploration(AdaptiveExplorationSettings.normal());
    var session = exploration.beginMacro(1);
    var observation =
        new Observation(
            Kind.DISCARD,
            new Measurement(0.6f, 0.5f, 0.55f),
            new Lookup(0.75f, LookupSource.KIND),
            1.0f,
            7);
    session.recordSelection(
        observation, new float[] {0.5f, 0.5f, 0.0f}, new float[] {0.25f, 0.5f, 0.25f}, 2);
    var report = exploration.completeMacro(session);
    var overall = report.overall();
    Assert.assertEquals(overall.meanTotalVariation(), 0.25, 0.0);
    Assert.assertEquals(
        overall.meanBehaviorToRolloutKl(),
        0.25 * Math.log(0.5) + 0.25 * Math.log(0.25 / 1.0e-30),
        1e-14);
    Assert.assertEquals(overall.meanSelectedRolloutRank(), 3.0, 0.0);
    Assert.assertEquals(overall.meanSelectedRolloutToBehaviorRatio(), 0.0, 0.0);
    Assert.assertEquals(overall.meanTopOneMassChange(), -0.25, 0.0);
    Assert.assertEquals(report.byKindAndDecile().getFirst().statistics(), overall);
  }
}

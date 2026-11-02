package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TrustCriteriaTest {

  private static GateReport report(Map<String, Double> measures, GateResult... failures) {
    return new GateReport(
        GateMode.FIRST_REPLACEMENT,
        failures.length == 0 ? GateResult.PASS : failures[0],
        List.of(failures),
        measures,
        null,
        null);
  }

  @Test
  public void readyNeedsTheTrustGames() {
    var config = RefitSchedulerTest.config("phiSource", "SELF_GRP", "phiStart", "SCORE_RANK");
    Assert.assertFalse(TrustCriteria.ready(config, 131071));
    Assert.assertTrue(TrustCriteria.ready(config, 131072));
  }

  @Test
  public void stabilityComesFromTheParityPair() {
    String rms = GateEvaluator.TWIN_RMS_MEASURE;
    GateReport main =
        report(Map.of(rms, 0.0), GateResult.RANK_FIT_WORSE, GateResult.UNSTABLE, GateResult.NOT_ENOUGH_DATA);
    GateReport parity = report(Map.of(rms, 0.02), GateResult.UNSTABLE, GateResult.MONOTONIC_VIOLATIONS);
    GateReport merged = TrustCriteria.merge(main, parity);
    Assert.assertEquals(
        merged.failures(), List.of(GateResult.RANK_FIT_WORSE, GateResult.UNSTABLE, GateResult.NOT_ENOUGH_DATA));
    Assert.assertEquals(merged.result(), GateResult.RANK_FIT_WORSE);
    Assert.assertEquals(merged.measures(), Map.of(rms, 0.0, "parity." + rms, 0.02));
    // 1 回目の UNSTABLE（候補と候補そのもの）は数えない、2 回目の UNSTABLE 以外も数えない
    GateReport pass =
        TrustCriteria.merge(report(Map.of(), GateResult.UNSTABLE), report(Map.of(rms, 0.01), GateResult.SLOPE_OUT_OF_RANGE));
    Assert.assertTrue(pass.passed());
    Assert.assertEquals(pass.failures(), List.of());
    GateReport unstable = TrustCriteria.merge(report(Map.of()), report(Map.of(rms, 0.02), GateResult.UNSTABLE));
    Assert.assertEquals(unstable.result(), GateResult.UNSTABLE);
  }

  /** 双子の出力が壊れていて 2 回目の関門が INVALID_OUTPUT だけを返しても、安定の柱は黙って外れない。 */
  @Test
  public void aBrokenTwinIsNotStable() {
    String rms = GateEvaluator.TWIN_RMS_MEASURE;
    Assert.assertEquals(
        TrustCriteria.merge(report(Map.of()), report(Map.of(), GateResult.INVALID_OUTPUT)).failures(),
        List.of(GateResult.UNSTABLE));
    Assert.assertEquals(
        TrustCriteria.merge(report(Map.of()), report(Map.of(rms, Double.NaN))).result(), GateResult.UNSTABLE);
    Assert.assertEquals(
        TrustCriteria.merge(report(Map.of()), report(Map.of(rms, Double.POSITIVE_INFINITY))).result(), GateResult.UNSTABLE);
  }

  @Test
  public void structuralValidityChecksTheMarginals() {
    double[] uniform = new double[16];
    Arrays.fill(uniform, 0.25);
    double[][] utilities = {{0, 0, 0, 0}};
    Assert.assertTrue(TrustCriteria.structurallyValid(new PhiValues(utilities, new double[][] {uniform})));
    double[] skewed = uniform.clone();
    skewed[0] = 0.5;
    skewed[1] = 0.0;
    // 席の和は 1 のまま、順位の和が崩れる
    Assert.assertFalse(TrustCriteria.structurallyValid(new PhiValues(utilities, new double[][] {skewed})));
    Assert.assertFalse(
        TrustCriteria.structurallyValid(new PhiValues(new double[][] {{0, Double.NaN, 0, 0}}, new double[][] {uniform})));
    Assert.assertFalse(TrustCriteria.structurallyValid(new PhiValues(new double[0][], new double[0][])));
  }

  @Test
  public void onlyAScoreOnlyPhiIsReplacedByTheTrustGate() {
    var config = RefitSchedulerTest.config("phiSource", "SELF_GRP", "phiStart", "SCORE_RANK");
    Assert.assertThrows(
        IllegalArgumentException.class,
        () ->
            TrustCriteria.gate(
                config,
                List.of(),
                List.of(),
                Path.of("x"),
                PhiEntryKind.SELF_CHECKPOINT,
                EpsilonUtilityProfile.TENHOU,
                Duration.ofMinutes(1)));
  }
}

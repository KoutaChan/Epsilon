package com.epsilon.prism.probe;

import com.epsilon.prism.core.DecisionClass;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** probe-nano の集計（{@link ProbeReport#of}）を手計算の値と比べる。 */
public class ProbeReportTest {
  private static final int EARLY = DecisionClass.DISCARD_EARLY.ordinal();

  /** 半荘 0 に ΔY = +0.02・+0.04、半荘 1 に −0.03（ρ = 0.5、4 半荘）。 */
  @Test
  public void aggregatesDeltasClusteredByGame() {
    double[] rates = new double[DecisionClass.COUNT];
    Arrays.fill(rates, 0.5);
    rates[DecisionClass.KAN.ordinal()] = 0;
    long[] floor = new long[DecisionClass.COUNT], agree = floor.clone(), disagree = floor.clone();
    floor[EARLY] = 1;
    agree[EARLY] = 7;
    disagree[EARLY] = 3;
    List<ProbeReport.Disagreement> rows =
        List.of(
            new ProbeReport.Disagreement(0, DecisionClass.DISCARD_EARLY, 0.02),
            new ProbeReport.Disagreement(0, DecisionClass.DISCARD_EARLY, 0.04),
            new ProbeReport.Disagreement(1, DecisionClass.DISCARD_EARLY, -0.03));
    ProbeReport.Report report = ProbeReport.of(4, rates, floor, agree, disagree, rows);
    for (ProbeReport.Line line : List.of(report.classes().get(EARLY), report.overall())) {
      Assert.assertEquals(line.probed(), 10);
      Assert.assertEquals(line.floor(), 1);
      Assert.assertEquals(line.agreementRate(), 0.7, 1e-12);
      Assert.assertEquals(line.meanDelta(), 0.01, 1e-12);
      // r_0 = 0.04、r_1 = −0.04、G = 2: SE = √(2·0.0032) / 3
      Assert.assertEquals(line.meanDeltaError(), 0.08 / 3, 1e-12);
      Assert.assertEquals(line.positiveShare(), 2 / 3.0, 1e-12);
      // S = (0.12, −0.06, 0, 0): 平均 0.015、SE = √(0.0057 / 4)
      Assert.assertEquals(line.perGame(), 0.015, 1e-12);
      Assert.assertEquals(line.perGameError(), Math.sqrt(0.0057 / 4), 1e-12);
      Assert.assertEquals(line.perSeat(), 0.015 / 4, 1e-12);
    }
    ProbeReport.Line kan = report.classes().get(DecisionClass.KAN.ordinal());
    Assert.assertTrue(Double.isNaN(kan.perGame()));
    Assert.assertTrue(Double.isNaN(kan.meanDelta()));
  }

  @Test(expectedExceptions = IllegalStateException.class)
  public void rejectsRowsThatDoNotMatchTheCounts() {
    double[] rates = new double[DecisionClass.COUNT];
    Arrays.fill(rates, 0.5);
    long[] zero = new long[DecisionClass.COUNT];
    ProbeReport.of(
        1,
        rates,
        zero,
        zero,
        zero,
        List.of(new ProbeReport.Disagreement(0, DecisionClass.CALL, 0.01)));
  }
}

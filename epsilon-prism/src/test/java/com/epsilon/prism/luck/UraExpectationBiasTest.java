package com.epsilon.prism.luck;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.DecisionClass;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 乱数の種を固定した 10 万局の自己対局（四人打ちと三人打ち）で、立直の和了者の実際の値と期待値の差の平均が、槓の有無ごとに標準誤差の
 * 2 倍以内に収まることを確かめる。全ての局末で数え上げの基準とも比べ、監査の集計が、テストで足した平均と z に合うことも確かめる。監査は
 * 局を束に振り分けて標準誤差を求めるので、z は行を独立とみなしたテストの z と 5% の幅で比べる。
 */
public class UraExpectationBiasTest {
  private static final Logger log = LoggerFactory.getLogger(UraExpectationBiasTest.class);

  private static final int HAND_ENDS = 100_000;

  private static final double Z_LINE = 2;

  @Test(groups = "slow", timeOut = 3_600_000)
  public void fourPlayerDifferencesStayWithinTwoStandardErrors() {
    assertUnbiased(Variant.TENHOU_4P, 20261109L);
  }

  @Test(groups = "slow", timeOut = 3_600_000)
  public void threePlayerDifferencesStayWithinTwoStandardErrors() {
    assertUnbiased(Variant.TENHOU_3P, 20261110L);
  }

  private static void assertUnbiased(Variant variant, long seed) {
    UraAudit audit = new UraAudit();
    UraExpectationPlayTest.Tally tally = UraExpectationPlayTest.play(variant, seed, HAND_ENDS, audit);
    UraAudit.Report report = audit.report();
    log.info(
        String.format(
            Locale.ROOT,
            "URA %s %s z(no kan)=%.3f z(kan)=%.3f us/averaged=%.2f",
            variant,
            tally,
            tally.z(false),
            tally.z(true),
            report.microsPerAveraged()));
    Assert.assertEquals(report.handEnds(), HAND_ENDS);
    Assert.assertEquals(report.riichiWins(), tally.riichiWins);
    Assert.assertEquals(report.averaged(), tally.riichiWins);
    Assert.assertTrue(tally.jointCompared > 0, tally.toString());
    for (boolean kan : new boolean[] {false, true}) {
      double[] cell = tally.winner[kan ? 1 : 0];
      Assert.assertTrue(cell[0] > 1_000, "kan " + kan + ": " + tally);
      double z = tally.z(kan);
      Assert.assertTrue(Math.abs(z) <= Z_LINE, "kan " + kan + ": z " + z + ", " + tally);
      UraAudit.Cell audited = cell(report, kan);
      Assert.assertEquals(audited.rows(), (long) cell[0]);
      Assert.assertEquals(audited.meanDifference(), cell[1] / cell[0], 1e-6);
      Assert.assertEquals(audited.z(), z, 0.05 * Math.abs(z) + 1e-3);
    }
  }

  private static UraAudit.Cell cell(UraAudit.Report report, boolean kan) {
    for (UraAudit.Cell cell : report.cells()) {
      if (cell.decisionClass() == DecisionClass.END && cell.kan() == kan) {
        return cell;
      }
    }
    throw new AssertionError("No audit cell for kan " + kan);
  }
}

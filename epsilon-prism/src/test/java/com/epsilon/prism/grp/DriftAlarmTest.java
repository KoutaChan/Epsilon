package com.epsilon.prism.grp;

import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DriftAlarmTest {
  /** 統計の線だけの緩い線（割合の下限なし、傾きの |z| 3.2、連続 2）。 */
  private static final DriftAlarm.Lines K = new DriftAlarm.Lines(4, 0, 3.2, 2);
  private static final DriftAlarm.Baseline PRODUCTION = DriftAlarm.Baseline.PRODUCTION;

  private static PhiMonitorLine line(int cohort, String phi, double dSpg, double slopeZ, long games) {
    double[] z = new double[HandGroup.values().length];
    Arrays.fill(z, Double.NaN);
    z[2] = slopeZ;
    CellReport report =
        new CellReport(games, games * 40, 0.25, dSpg, 0, new double[8], new double[8], new double[8], z, 1.2, 0.6, 0);
    return new PhiMonitorLine(cohort, phi, "sha", report, Map.of(), dSpg, 1, 1);
  }

  private static PhiMonitorLine line(int cohort, double dSpg) {
    return line(cohort, "g000", dSpg, 0.5, 65536);
  }

  @Test
  public void theMarginKeepsSmallDriftsQuiet() {
    var strict = new DriftAlarm.Lines(4, 0.5, 8, 3);
    double drifted = DriftAlarm.FLOOR * 1.25;
    DriftAlarm.State loose = null, held = null;
    for (int t = 1; t <= 40; t++) {
      loose = DriftAlarm.next(loose, line(t, drifted), K, PRODUCTION, true);
      held = DriftAlarm.next(held, line(t, drifted), strict, PRODUCTION, true);
    }
    Assert.assertTrue(loose.on(), "the statistical line alone turns on at +25%");
    Assert.assertFalse(held.on(), "a +25% drift stays under the 50% margin");
    Assert.assertEquals(held.onLine(), DriftAlarm.FLOOR * 1.5, 1e-15);
    Assert.assertEquals(held.offLine(), DriftAlarm.FLOOR * 1.25, 1e-15);
  }

  @Test
  public void theSlopeLineAndTheStreakComeFromTheSettings() {
    var strict = new DriftAlarm.Lines(4, 0.5, 8, 3);
    DriftAlarm.State tilted = null;
    for (int t = 1; t <= 2; t++)
      tilted = DriftAlarm.next(tilted, line(t, "g000", DriftAlarm.FLOOR, 9, 65536), strict, PRODUCTION, true);
    Assert.assertFalse(tilted.on(), "two tilted cohorts are not enough");
    tilted = DriftAlarm.next(tilted, line(3, "g000", DriftAlarm.FLOOR, 9, 65536), strict, PRODUCTION, true);
    Assert.assertTrue(tilted.on(), "the third tilted cohort turns it on");
    DriftAlarm.State mild = null;
    for (int t = 1; t <= 5; t++)
      mild = DriftAlarm.next(mild, line(t, "g000", DriftAlarm.FLOOR, 7, 65536), strict, PRODUCTION, true);
    Assert.assertEquals(mild.streak(), 0, "|z| 7 is within the slope line 8");
  }

  @Test
  public void linesFollowTheEwmaVariance() {
    double sd = DriftAlarm.COHORT_SD;
    DriftAlarm.State first = DriftAlarm.next(null, line(10, 9e-5), K, PRODUCTION, true);
    Assert.assertEquals(first.dSpgEwma(), 9e-5, 0);
    Assert.assertEquals(first.offLine(), DriftAlarm.FLOOR + 2 * sd, 1e-15);
    Assert.assertEquals(first.onLine(), DriftAlarm.FLOOR + K.k() * sd, 1e-15);
    DriftAlarm.State second = DriftAlarm.next(first, line(11, 7e-5), K, PRODUCTION, true);
    double lambda = PhiMonitorLine.EWMA_WEIGHT;
    Assert.assertEquals(second.dSpgEwma(), (1 - lambda) * 9e-5 + lambda * 7e-5, 1e-18);
    double variance = (1 - lambda) * (1 - lambda) * sd * sd + lambda * lambda * sd * sd;
    Assert.assertEquals(second.offLine(), DriftAlarm.FLOOR + 2 * Math.sqrt(variance), 1e-15);
    // 長く続けると SD_EWMA は SD·√(λ/(2−λ)) に近づく
    DriftAlarm.State state = second;
    for (int t = 12; t < 200; t++) state = DriftAlarm.next(state, line(t, 8e-5), K, PRODUCTION, true);
    Assert.assertEquals(
        (state.offLine() - DriftAlarm.FLOOR) / 2, sd * Math.sqrt(lambda / (2 - lambda)), 1e-12);
    Assert.assertEquals(DriftAlarm.cohortSd(16384), 2 * sd, 1e-15);
  }

  @Test
  public void twoConsecutiveExceedancesTurnItOnUntilTheOffLine() {
    DriftAlarm.State state = null;
    for (int t = 1; t <= 60; t++) state = DriftAlarm.next(state, line(t, DriftAlarm.FLOOR), K, PRODUCTION, true);
    Assert.assertFalse(state.on());
    double high = state.onLine() * 3;
    DriftAlarm.State once = DriftAlarm.next(state, line(61, high), K, PRODUCTION, true);
    Assert.assertEquals(once.streak(), once.dSpgEwma() >= once.onLine() ? 1 : 0);
    DriftAlarm.State a = once;
    int t = 62;
    while (a.streak() < 1) a = DriftAlarm.next(a, line(t++, high), K, PRODUCTION, true);
    Assert.assertFalse(a.on());
    DriftAlarm.State on = DriftAlarm.next(a, line(t++, high), K, PRODUCTION, true);
    Assert.assertEquals(on.streak(), 2);
    Assert.assertTrue(on.on());
    // 切り替えが入りきっていなければ ON にしない
    Assert.assertFalse(DriftAlarm.next(a, line(t, high), K, PRODUCTION, false).on());
    // OFF の線まで下がるまで ON のまま
    DriftAlarm.State staying = on;
    while (staying.dSpgEwma() > staying.offLine()) {
      Assert.assertTrue(staying.on());
      staying = DriftAlarm.next(staying, line(t++, DriftAlarm.FLOOR), K, PRODUCTION, true);
    }
    Assert.assertFalse(staying.on());
  }

  @Test
  public void slopeOutsideTheBonferroniLineCountsToo() {
    DriftAlarm.State first = DriftAlarm.next(null, line(1, "g000", DriftAlarm.FLOOR, -3.3, 65536), K, PRODUCTION, true);
    Assert.assertEquals(first.maxSlopeZ(), 3.3, 1e-12);
    Assert.assertEquals(first.streak(), 1);
    Assert.assertFalse(first.on());
    DriftAlarm.State second = DriftAlarm.next(first, line(2, "g000", DriftAlarm.FLOOR, 3.4, 65536), K, PRODUCTION, true);
    Assert.assertTrue(second.on());
    DriftAlarm.State back = DriftAlarm.next(second, line(3, "g000", DriftAlarm.FLOOR, 1.0, 65536), K, PRODUCTION, true);
    Assert.assertFalse(back.on());
    Assert.assertEquals(DriftAlarm.next(first, line(2, "g000", DriftAlarm.FLOOR, 3.1, 65536), K, PRODUCTION, true).streak(), 0);
  }

  @Test
  public void aNewEntryRestartsAndScoreOnlyNeverAlarms() {
    DriftAlarm.State a = DriftAlarm.next(null, line(1, "g000", 1e-3, 0, 65536), K, PRODUCTION, true);
    DriftAlarm.State b = DriftAlarm.next(a, line(2, "g000", 1e-3, 0, 65536), K, PRODUCTION, true);
    Assert.assertTrue(b.on());
    DriftAlarm.State switched = DriftAlarm.next(b, line(3, "g2-a1", 1e-3, 0, 65536), K, PRODUCTION, true);
    Assert.assertEquals(switched.streak(), 1);
    Assert.assertFalse(switched.on());
    Assert.assertEquals(switched.dSpgEwma(), 1e-3, 0);
    DriftAlarm.State score = DriftAlarm.next(null, line(1, "g000", 1, 9, 8192), K, null, true);
    Assert.assertFalse(score.on());
    Assert.assertTrue(Double.isNaN(score.onLine()));
    DriftAlarm.State after = DriftAlarm.next(score, line(2, "g000", 1e-3, 0, 8192), K, PRODUCTION, true);
    Assert.assertEquals(after.streak(), 1);
  }

  /**
   * 自己対局で学んだ項目は、公開のときに測ったその候補の較正からの増分で決める: D_SPG が本番の床より大きくても安定していれば ON にならず、
   * 区分の傾きも基準からの z で見る。
   */
  @Test
  public void aSelfPlayEntryIsJudgedFromItsOwnBaseline() {
    double[] slope = new double[HandGroup.values().length], slopeSe = new double[slope.length];
    Arrays.fill(slope, 1.05);
    Arrays.fill(slopeSe, 0.002);
    var baseline =
        DriftAlarm.Baseline.of(
            new CellReport(65536, 65536 * 40L, 0.25, 1.3e-4, 0, new double[8], slope, slopeSe, new double[8], 1.2, 0.6, 0));
    DriftAlarm.State state = null;
    for (int t = 1; t <= 40; t++) {
      PhiMonitorLine line = selfLine(t, 1.3e-4, 1.05, 0.002);
      state = DriftAlarm.next(state, line, K, baseline, true);
      Assert.assertFalse(state.on(), "t=" + t);
      // 大域の床で比べると ON のままになる
      Assert.assertTrue(DriftAlarm.next(null, line, K, PRODUCTION, true).dSpgEwma() > DriftAlarm.next(null, line, K, PRODUCTION, true).onLine());
    }
    Assert.assertEquals(state.floor(), 1.3e-4, 0);
    Assert.assertEquals(state.maxSlopeZ(), 0, 1e-9);
    // 傾きが基準から離れれば数える
    DriftAlarm.State tilted = DriftAlarm.next(state, selfLine(41, 1.3e-4, 1.02, 0.002), K, baseline, true);
    Assert.assertTrue(tilted.maxSlopeZ() > K.slopeZ(), String.valueOf(tilted.maxSlopeZ()));
    Assert.assertEquals(tilted.streak(), 1);
    // 床から大きく離れれば ON
    DriftAlarm.State high = state;
    for (int t = 42; t < 60 && !high.on(); t++) high = DriftAlarm.next(high, selfLine(t, 4e-4, 1.05, 0.002), K, baseline, true);
    Assert.assertTrue(high.on());
  }

  private static PhiMonitorLine selfLine(int cohort, double dSpg, double slope, double slopeSe) {
    double[] slopes = new double[HandGroup.values().length], errors = new double[slopes.length], z = new double[slopes.length];
    Arrays.fill(slopes, slope);
    Arrays.fill(errors, slopeSe);
    Arrays.fill(z, (slope - 1) / slopeSe);
    CellReport report = new CellReport(65536, 65536 * 40L, 0.25, dSpg, 0, new double[8], slopes, errors, z, 1.2, 0.6, 0);
    return new PhiMonitorLine(cohort, "g048-a1", "sha", report, Map.of(), dSpg, 1, 1);
  }

  @Test
  public void updateAppendsOneLinePerCohort() throws IOException {
    Path run = Files.createTempDirectory("drift");
    GrpFiles files = new GrpFiles(run);
    PrismConfig fixed = RefitSchedulerTest.config("phiDriftConsecutive", "2");
    Assert.assertFalse(DriftAlarm.isOn(files));
    DriftAlarm.update(files, fixed, line(5, "fixed", 1e-3, 0, 65536));
    DriftAlarm.State second = DriftAlarm.update(files, fixed, line(6, "fixed", 1e-3, 0, 65536));
    Assert.assertTrue(second.on());
    Assert.assertTrue(DriftAlarm.isOn(files));
    Assert.assertEquals(DriftAlarm.update(files, fixed, line(6, "fixed", 0, 0, 65536)), second);
    Assert.assertEquals(PhiHistory.lines(files.drift()).size(), 2);
    Files.writeString(files.drift(), "{\"cohort\":7,\"ph", StandardOpenOption.APPEND);
    Assert.assertTrue(DriftAlarm.isOn(files));
    // 書きかけの行は切り捨てて足す
    DriftAlarm.update(files, fixed, line(7, "fixed", 1e-3, 0, 65536));
    Assert.assertEquals(PhiHistory.lines(files.drift()).size(), 3);
    Assert.assertTrue(DriftAlarm.isOn(files));
    PrismConfig score = RefitSchedulerTest.config("phiSource", "SCORE_SMOOTH");
    Assert.assertFalse(DriftAlarm.update(files, score, line(8, "score-smooth", 1, 9, 65536)).on());
  }
}

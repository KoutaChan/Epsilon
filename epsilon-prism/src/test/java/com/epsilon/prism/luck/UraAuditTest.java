package com.epsilon.prism.luck;

import com.epsilon.grp.Phi;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.prism.core.DecisionClass;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 裏ドラの監査の集計を確かめる: 区分ごとの式、局末の数え方、局で束ねた標準誤差、足す順とスレッドに依らないこと。 */
public class UraAuditTest {

  private static final double EPSILON = 1e-12;

  /** 監査に数える行。 */
  record AuditRow(DecisionClass decisionClass, int seat) implements UraAudit.Row {}

  /** 判断のクラスが同じで、席を並べた順の行。 */
  static List<UraAudit.Row> rows(DecisionClass decisionClass, int... seats) {
    List<UraAudit.Row> rows = new ArrayList<>(seats.length);
    for (int seat : seats) rows.add(new AuditRow(decisionClass, seat));
    return rows;
  }

  /** 入る束が互いに違う局を {@code count} 個返す。[i] → 半荘 ID で、局の序数はどれも 0。 */
  static long[] games(int count) {
    long[] games = new long[count];
    Set<Integer> clusters = new HashSet<>();
    long game = 0;
    for (int found = 0; found < count; game++) if (clusters.add(UraAudit.cluster(game, 0))) games[found++] = game;
    return games;
  }

  /** 区分ごとの式: 平均と局で束ねた分散から、標準誤差（G/(G−1) を掛ける）・z・分散が減る割合を求める。 */
  @Test
  public void aCellFollowsItsFormulas() {
    UraAudit.Cell cell = UraAudit.Cell.of(DecisionClass.KAN, true, 60, 30, 0.5, 0.1, 0.75, 0.2, 1);
    Assert.assertEquals(cell.decisionClass(), DecisionClass.KAN);
    Assert.assertTrue(cell.kan());
    Assert.assertEquals(cell.rows(), 60);
    Assert.assertEquals(cell.clusters(), 30);
    Assert.assertEquals(cell.meanDifference(), 0.5, EPSILON);
    double error = Math.sqrt(0.1 * 30 / (30 - 1));
    Assert.assertEquals(cell.standardError(), error, EPSILON);
    Assert.assertEquals(cell.z(), 0.5 / error, EPSILON);
    Assert.assertEquals(cell.realizedVariance(), 0.75, EPSILON);
    Assert.assertEquals(cell.expectedVariance(), 0.2, EPSILON);
    Assert.assertEquals(cell.labelVariance(), 1.0, EPSILON);
    Assert.assertEquals(cell.reduction(), 0.2, EPSILON);
  }

  /** 束が 1 つなら、行がいくつあっても標準誤差と z は NaN。ラベルの分散が 0 なら、分散が減る割合は NaN。 */
  @Test
  public void aSingleClusterHasNoStandardError() {
    UraAudit.Cell cell = UraAudit.Cell.of(DecisionClass.END, false, 3, 1, 0.25, 0.01, 0.0625, 0.01, 0);
    Assert.assertTrue(Double.isNaN(cell.standardError()));
    Assert.assertTrue(Double.isNaN(cell.z()));
    Assert.assertTrue(Double.isNaN(cell.reduction()));
  }

  /**
   * 束が少ない区分の z は、行がいくつあっても当てにならないので求めず、要約にも書かない（束が 2 つで 30 行の区分など）。標準誤差は束が
   * 2 つあれば求める。束が {@link UraAudit#MIN_CLUSTERS} 個あれば z も求める。
   */
  @Test
  public void fewClustersHaveNoZ() {
    UraAudit.Cell two = UraAudit.Cell.of(DecisionClass.KAN, true, 30, 2, 0.5, 0.1, 0.75, 0.2, 1);
    UraAudit.Cell below =
        UraAudit.Cell.of(DecisionClass.KAN, false, 500, UraAudit.MIN_CLUSTERS - 1, 0.5, 0.1, 0.75, 0.2, 1);
    UraAudit.Cell enough =
        UraAudit.Cell.of(DecisionClass.END, false, 500, UraAudit.MIN_CLUSTERS, 0.5, 0.1, 0.75, 0.2, 1);
    for (UraAudit.Cell cell : List.of(two, below)) {
      Assert.assertTrue(cell.standardError() > 0, cell.toString());
      Assert.assertTrue(Double.isNaN(cell.z()), cell.toString());
    }
    Assert.assertEquals(enough.z(), 0.5 / enough.standardError(), EPSILON);
    List<UraAudit.CellSummary> cells =
        new UraAudit.Report(3, 3, 3, new long[UraAudit.GROUP_BINS], 0, List.of(two, below, enough)).summary().cells();
    Assert.assertNull(cells.get(0).z(), two.toString());
    Assert.assertNull(cells.get(1).z(), below.toString());
    Assert.assertEquals(cells.get(2).z(), enough.z());
    Assert.assertEquals(cells.get(0).reduction(), two.reduction());
  }

  /** 局末の数え方: 期待値を求めた局末だけ、基本点の組の数のビンと時間を数える。期待値を求めていない局末の行は数えない。 */
  @Test
  public void handEndsAreCountedByWhatWasComputed() {
    Phi phi = Phi.scoreSmooth(UraScenes.VARIANT, RankUtility.TENHOU);
    UraScenes.Played played = UraScenes.plainRon();
    UraExpectation averaged = new UraExpectation(phi);
    averaged.evaluate(played.engine().full(), played.engine().roundEnd(), played.start(), null, true);
    UraExpectation realized = new UraExpectation(phi);
    realized.evaluate(played.engine().full(), played.engine().roundEnd(), played.start(), null, false);
    UraAudit audit = new UraAudit();
    audit.handEnd(averaged);
    audit.handEnd(realized);
    audit.rows(realized, 0, 0, rows(DecisionClass.END, 2));
    UraAudit.Report report = audit.report();
    Assert.assertEquals(report.handEnds(), 2);
    Assert.assertEquals(report.riichiWins(), 2);
    Assert.assertEquals(report.averaged(), 1);
    Assert.assertEquals(report.groups()[averaged.groups()], 1);
    Assert.assertEquals(report.nanos(), averaged.nanos());
    Assert.assertTrue(report.cells().isEmpty());
    audit.rows(averaged, 0, 0, rows(DecisionClass.END, 2));
    List<UraAudit.Cell> cells = audit.report().cells();
    Assert.assertEquals(cells.size(), 1);
    Assert.assertEquals(cells.get(0).decisionClass(), DecisionClass.END);
    Assert.assertFalse(cells.get(0).kan());
    Assert.assertEquals(cells.get(0).rows(), 1);
    Assert.assertEquals(cells.get(0).clusters(), 1);
    Assert.assertEquals(cells.get(0).meanDifference(), averaged.actual(2) - averaged.expected(2), 1e-9);
    Assert.assertEquals(cells.get(0).expectedVariance(), averaged.variance(2), 1e-9);
  }

  /**
   * 束が互いに違う局ごとに行を 1 つずつ足した区分の平均と分散は、同じ行の値から double のまま 2 パスで求めた値と、丸めの幅（2^−32）の
   * 中で合う。どの束も行が 1 つなので、標準誤差は行を独立とみなした値 √(Σ(d − d̄)² / (N·(N−1))) になる。
   */
  @Test
  public void singleRowClustersKeepTheRowStandardError() {
    Phi phi = Phi.scoreSmooth(UraScenes.VARIANT, RankUtility.TENHOU);
    int seats = UraScenes.VARIANT.seats();
    UraAudit audit = new UraAudit();
    Map<Boolean, List<UraExpectation>> byKan = new HashMap<>();
    List<UraScenes.Played> scenes =
        List.of(UraScenes.plainRon(), UraScenes.chankan(), UraScenes.chankanWithPendingIndicator());
    long[] games = games(scenes.size() * seats);
    int added = 0;
    for (UraScenes.Played played : scenes) {
      UraExpectation values = new UraExpectation(phi);
      Assert.assertTrue(values.evaluate(played.engine().full(), played.engine().roundEnd(), played.start(), null, true));
      for (int seat = 0; seat < seats; seat++) {
        audit.rows(values, games[added++], 0, rows(DecisionClass.END, seat));
      }
      byKan.computeIfAbsent(values.kan(), kan -> new ArrayList<>()).add(values);
    }
    List<UraAudit.Cell> cells = audit.report().cells();
    Assert.assertEquals(cells.size(), byKan.size());
    for (UraAudit.Cell cell : cells) {
      List<UraExpectation> ends = byKan.get(cell.kan());
      int n = ends.size() * seats;
      double meanDifference = 0, meanActual = 0, meanVariance = 0;
      for (UraExpectation values : ends) {
        for (int seat = 0; seat < seats; seat++) {
          meanDifference += (values.actual(seat) - values.expected(seat)) / n;
          meanActual += values.actual(seat) / n;
          meanVariance += values.variance(seat) / n;
        }
      }
      double differenceVariance = 0, squared = 0, labelVariance = 0;
      for (UraExpectation values : ends) {
        for (int seat = 0; seat < seats; seat++) {
          double difference = values.actual(seat) - values.expected(seat);
          differenceVariance += (difference - meanDifference) * (difference - meanDifference) / n;
          squared += difference * difference / n;
          labelVariance += (values.actual(seat) - meanActual) * (values.actual(seat) - meanActual) / n;
        }
      }
      Assert.assertEquals(cell.rows(), n);
      Assert.assertEquals(cell.clusters(), n);
      Assert.assertEquals(cell.meanDifference(), meanDifference, 1e-9);
      Assert.assertEquals(cell.standardError(), Math.sqrt(differenceVariance / (n - 1)), 1e-9);
      Assert.assertEquals(cell.realizedVariance(), squared, 1e-9);
      Assert.assertEquals(cell.expectedVariance(), meanVariance, 1e-9);
      Assert.assertEquals(cell.labelVariance(), labelVariance, 1e-9);
      Assert.assertTrue(cell.labelVariance() > 0, cell.toString());
    }
  }

  /**
   * 同じ局の行は束ねて数える。束が互いに違う局ごとに行の組（同じ席が重なる組や、片方のクラスに行の無い局を含む）を変えて 2 つのクラスに
   * 足すと、区分ごとの標準誤差は、局ごとの d の和 D_g と行の数 m_g から手で求めた √(G/(G−1)·Σ_g (D_g − m_g·d̄)² / N²) と合い、行を
   * 独立とみなした値とは違う。
   */
  @Test
  public void rowsSharingAHandAreClustered() {
    Phi phi = Phi.scoreSmooth(UraScenes.VARIANT, RankUtility.TENHOU);
    UraScenes.Played played = UraScenes.plainRon();
    UraExpectation values = new UraExpectation(phi);
    Assert.assertTrue(values.evaluate(played.engine().full(), played.engine().roundEnd(), played.start(), null, true));
    // [局] → 行の席
    int[][] discards = {{0, 1, 2, 3}, {2, 2, 2}, {1, 3}, {2}, {0, 0, 1}};
    int[][] wins = {{2}, {2, 1}, {}, {2, 2}, {3}};
    long[] games = games(discards.length);
    UraAudit audit = new UraAudit();
    for (int hand = 0; hand < discards.length; hand++) {
      List<UraAudit.Row> rows = rows(DecisionClass.END, wins[hand]);
      rows.addAll(rows(DecisionClass.DISCARD_MID, discards[hand]));
      audit.rows(values, games[hand], 0, rows);
    }
    Map<DecisionClass, int[][]> seatsByClass = Map.of(DecisionClass.DISCARD_MID, discards, DecisionClass.END, wins);
    List<UraAudit.Cell> cells = audit.report().cells();
    Assert.assertEquals(cells.size(), seatsByClass.size());
    for (UraAudit.Cell cell : cells) {
      int[][] hands = seatsByClass.get(cell.decisionClass());
      int n = 0, g = 0;
      double sum = 0;
      for (int[] seats : hands) {
        n += seats.length;
        if (seats.length > 0) g++;
        for (int seat : seats) sum += values.actual(seat) - values.expected(seat);
      }
      double mean = sum / n, clustered = 0, independent = 0;
      for (int[] seats : hands) {
        double handSum = 0;
        for (int seat : seats) {
          double difference = values.actual(seat) - values.expected(seat);
          handSum += difference;
          independent += (difference - mean) * (difference - mean);
        }
        clustered += (handSum - seats.length * mean) * (handSum - seats.length * mean);
      }
      double error = Math.sqrt(g / (g - 1.0) * clustered / ((double) n * n));
      double rowError = Math.sqrt(independent / ((double) n * (n - 1)));
      String where = cell + " row error " + rowError;
      Assert.assertFalse(cell.kan(), where);
      Assert.assertEquals(cell.rows(), n, where);
      Assert.assertEquals(cell.clusters(), g, where);
      Assert.assertEquals(cell.meanDifference(), mean, 1e-9, where);
      Assert.assertEquals(cell.standardError(), error, 1e-9, where);
      Assert.assertTrue(Math.abs(error - rowError) > 0.01 * rowError, where);
    }
  }

  /** 局末を足す順、局末の中の行の順、スレッドを変えても、同じ局に足した行が同じなら、報告に写した和はビットまで同じ。 */
  @Test
  public void sumsDoNotDependOnTheOrder() throws Exception {
    Phi phi = Phi.scoreSmooth(UraScenes.VARIANT, RankUtility.TENHOU);
    List<UraExpectation> ends = new ArrayList<>();
    for (UraScenes.Played played :
        List.of(UraScenes.plainRon(), UraScenes.chankan(), UraScenes.chankanWithPendingIndicator())) {
      UraExpectation values = new UraExpectation(phi);
      values.evaluate(played.engine().full(), played.engine().roundEnd(), played.start(), null, true);
      ends.add(values);
    }
    UraAudit forward = new UraAudit();
    for (int repeat = 0; repeat < 50; repeat++) {
      for (int index = 0; index < ends.size(); index++) {
        forward.rows(ends.get(index), repeat, index, rows(DecisionClass.END, 0, 1, 2, 3));
      }
    }
    UraAudit threaded = new UraAudit();
    ExecutorService pool = Executors.newFixedThreadPool(4);
    try {
      List<Future<?>> tasks = new ArrayList<>();
      for (int repeat = 0; repeat < 50; repeat++) {
        for (int index = ends.size() - 1; index >= 0; index--) {
          UraExpectation values = ends.get(index);
          long game = repeat;
          int hand = index;
          tasks.add(pool.submit(() -> threaded.rows(values, game, hand, rows(DecisionClass.END, 3, 2, 1, 0))));
        }
      }
      for (Future<?> task : tasks) {
        task.get();
      }
    } finally {
      pool.shutdown();
      Assert.assertTrue(pool.awaitTermination(1, TimeUnit.MINUTES));
    }
    List<UraAudit.Cell> expected = forward.report().cells();
    List<UraAudit.Cell> actual = threaded.report().cells();
    Assert.assertEquals(actual.size(), 2);
    Assert.assertEquals(actual, expected);
  }
}

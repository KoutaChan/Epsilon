package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntPredicate;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 終盤の報告を確かめる。測るのは数え上げる局末だけで、両方のモードの較正は、元の列で数え上げる局末を選んで測った較正と同じになる。
 * 決まり具合の区間は、GRP のモードで今トップの席が 1 着で終わる確率で分け、区間は下の境を含む。
 * 作った値（LATE は Φ ＝ u で最終順位の確率を 1、GRP のモードは Φ ＝ u/2）では次のようになる。対の MSE の差は −u²/4 の
 * 平均、順位の NLL の差は GRP のモードでの最終順位の確率の対数の平均になる。区間ごとの終盤の 2 区分の傾きは、LATE が 1、GRP の
 * モードが 2。b4 は、LATE が 0、GRP のモードが今最下位の席の u/2 の平均。数え上げの集計は、数え上げで求めた値なら数え上げる
 * 局末での平均になり、作った値では 0 になる。
 */
public class LateReportTest {

  private static final RankUtility UTILITY = RankUtility.TENHOU;

  /** GRP のモードで、今トップの席が 1 着で終わる確率。局末の番号の順に繰り返して使う。区間の境の 0.5 と 0.9 も入れる。 */
  private static final double[] TOPS = {0.2, 0.4999, 0.5, 0.7, 0.8999, 0.9, 0.97, 0.99};

  /** {@link #TOPS} のそれぞれが入る区間。 */
  private static final int[] TOP_BINS = {0, 0, 1, 1, 1, 2, 2, 2};

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void builtValuesGiveKnownBinsAndDifferences(Variant variant) {
    HandEnds ends = HandEnds.of(variant, UTILITY, TrainFixtures.simulated(variant, 400, 0, 53));
    int seats = variant.seats();
    int lastGroup = ends.groups() - 1;
    HandEndValues.Builder lateBuilder = HandEndValues.builder(ends);
    HandEndValues.Builder grpBuilder = HandEndValues.builder(ends);
    double[] latePhi = new double[seats];
    double[] grpPhi = new double[seats];
    double[] lateRanks = new double[seats * seats];
    double[] grpRanks = new double[seats * seats];
    for (int h = 0; h < ends.handEnds(); h++) {
      double top = TOPS[h % TOPS.length];
      Arrays.fill(lateRanks, 0);
      for (int seat = 0; seat < seats; seat++) {
        double u = ends.utility(h, seat);
        latePhi[seat] = u;
        grpPhi[seat] = u / 2;
        lateRanks[seat * seats + ends.finalRank(h, seat)] = 1;
        boolean leader = ends.rank(h, seat) == 0;
        for (int place = 0; place < seats; place++) {
          grpRanks[seat * seats + place] = !leader ? 1.0 / seats : place == 0 ? top : (1 - top) / (seats - 1);
        }
      }
      lateBuilder.set(h, latePhi, lateRanks);
      grpBuilder.set(h, grpPhi, grpRanks);
    }
    HandEndValues late = lateBuilder.build();
    HandEndValues grp = grpBuilder.build();
    LateReport report = LateReport.measure(ends, late, grp);

    IntPredicate covered = h -> FinalRound.covers(ends.state(h));
    long handEnds = 0;
    long[] binEnds = new long[3];
    long[][] groupSeats = new long[3][2];
    double[] binSquares = new double[3];
    double[][] lastSum = new double[3][2];
    long[][] lastCount = new long[3][2];
    double squares = 0;
    double rankLogs = 0;
    for (int h = 0; h < ends.handEnds(); h++) {
      if (!covered.test(h)) {
        continue;
      }
      handEnds++;
      int bin = TOP_BINS[h % TOPS.length];
      int lateGroup = ends.group(h) - (lastGroup - 1);
      Assert.assertTrue(lateGroup == 0 || lateGroup == 1, ends.state(h).toString());
      binEnds[bin]++;
      groupSeats[bin][lateGroup] += seats;
      for (int seat = 0; seat < seats; seat++) {
        double u = ends.utility(h, seat);
        squares -= u * u / 4;
        binSquares[bin] -= u * u / 4;
        rankLogs += Math.log(grp.marginal(h, seat, ends.finalRank(h, seat)));
        if (ends.rank(h, seat) == seats - 1) {
          lastSum[bin][lateGroup] += u / 2;
          lastCount[bin][lateGroup]++;
        }
      }
    }
    Assert.assertTrue(handEnds > 300, variant + " " + handEnds);
    Assert.assertEquals(report.handEnds(), handEnds);
    Assert.assertEquals(report.late(), GrpMetrics.measure(ends, late, covered));
    Assert.assertEquals(report.grp(), GrpMetrics.measure(ends, grp, covered));
    long seatHands = handEnds * seats;
    Assert.assertEquals(report.mse().mean(), squares / seatHands, 1e-12, report.mse().toString());
    Assert.assertEquals(report.rankNll().mean(), rankLogs / seatHands, 1e-12, report.rankNll().toString());

    Assert.assertEquals(report.bins().size(), 3);
    double[] edges = {0, 0.5, 0.9, 1};
    for (int bin = 0; bin < 3; bin++) {
      LateReport.Bin measured = report.bins().get(bin);
      String context = variant + " bin " + bin + " " + measured;
      Assert.assertEquals(measured.low(), edges[bin], 0, context);
      Assert.assertEquals(measured.high(), edges[bin + 1], 0, context);
      Assert.assertEquals(measured.handEnds(), binEnds[bin], context);
      Assert.assertEquals(measured.mse().mean(), binSquares[bin] / (binEnds[bin] * seats), 1e-12, context);
      Assert.assertEquals(measured.late().size(), 2, context);
      Assert.assertEquals(measured.grp().size(), 2, context);
      for (int group = 0; group < 2; group++) {
        GrpReport.Fit lateFit = measured.late().get(group);
        GrpReport.Fit grpFit = measured.grp().get(group);
        Assert.assertEquals(lateFit.seatHands(), groupSeats[bin][group], context);
        Assert.assertEquals(grpFit.seatHands(), groupSeats[bin][group], context);
        Assert.assertEquals(lateFit.slope(), 1, 1e-9, context);
        Assert.assertEquals(lateFit.b4(), 0, 1e-12, context);
        Assert.assertEquals(grpFit.slope(), 2, 1e-9, context);
        Assert.assertEquals(grpFit.b4(), lastSum[bin][group] / lastCount[bin][group], 1e-12, context);
      }
    }
    Assert.assertEquals(report.enumeration(), new LateReport.Enumeration(0, 0, 0, 0));

    LateReport same = LateReport.measure(ends, grp, grp);
    Assert.assertEquals(same.mse().mean(), 0, 0);
    Assert.assertEquals(same.rankNll().mean(), 0, 0);
    for (LateReport.Bin bin : same.bins()) {
      Assert.assertEquals(bin.mse().mean(), 0, 0, bin.toString());
    }
    HandEnds other = HandEnds.of(variant, UTILITY, TrainFixtures.simulated(variant, 20, 0, 59));
    GrpNetwork leaf = LateHandEndValuesTest.round(variant, 0.5).leaf();
    HandEndValues foreign = HandEndValues.of(other, leaf, GrpWorkspace.of(leaf.shape()));
    Assert.assertThrows(IllegalArgumentException.class, () -> LateReport.measure(ends, foreign, grp));
    Assert.assertThrows(IllegalArgumentException.class, () -> LateReport.measure(ends, late, foreign));
  }

  /**
   * 数え上げの Φ の値と、同じ GRP の葉のネットワークの値を比べる。両方のモードの較正と対の MSE は、元の列で数え上げる局末を選んで
   * 測った値と同じになる。区間の局末の数は、葉のネットワークで見たトップの席の 1 着の確率で分けた数になる。区間ごとの対の MSE は、
   * その区間の局末を選んで測った値と同じ。数え上げの集計は、数え上げる局末での、埋めた確率・半荘が終わる組の確率・精算した組の数・
   * 半荘が続く葉の数の平均。
   */
  @Test(dataProvider = "variants")
  public void enumeratedValuesAreComparedOnCoveredHandEnds(Variant variant) {
    HandEnds ends = HandEnds.of(variant, UTILITY, TrainFixtures.simulated(variant, 300, 0, 41));
    FinalRound round = LateHandEndValuesTest.round(variant, 0.5);
    HandEndValues late = HandEndValues.of(ends, Phi.late(round, UTILITY, 256));
    HandEndValues grp = HandEndValues.of(ends, round.leaf(), GrpWorkspace.of(round.leaf().shape()));
    LateReport report = LateReport.measure(ends, late, grp);
    IntPredicate covered = h -> FinalRound.covers(ends.state(h));
    Assert.assertEquals(report.late(), GrpMetrics.measure(ends, late, covered));
    Assert.assertEquals(report.grp(), GrpMetrics.measure(ends, grp, covered));
    Assert.assertEquals(report.mse(), GrpMetrics.pairedMse(ends, grp, late, covered));

    int[] bins = new int[ends.handEnds()];
    long[] binEnds = new long[3];
    long handEnds = 0;
    double filled = 0;
    double terminal = 0;
    double settled = 0;
    double leaves = 0;
    for (int h = 0; h < ends.handEnds(); h++) {
      if (!covered.test(h)) {
        continue;
      }
      int leader = 0;
      while (ends.rank(h, leader) != 0) {
        leader++;
      }
      double top = grp.marginal(h, leader, 0);
      bins[h] = top < 0.5 ? 0 : top < 0.9 ? 1 : 2;
      binEnds[bins[h]]++;
      handEnds++;
      filled += late.filledMass(h);
      terminal += late.terminalMass(h);
      settled += late.settled(h);
      leaves += late.leaves(h);
      Assert.assertTrue(late.leaves(h) <= late.settled(h), ends.state(h).toString());
    }
    Assert.assertEquals(report.handEnds(), handEnds);
    for (int bin = 0; bin < 3; bin++) {
      int chosen = bin;
      LateReport.Bin measured = report.bins().get(bin);
      Assert.assertEquals(measured.handEnds(), binEnds[bin], measured.toString());
      Assert.assertEquals(
          measured.mse(), GrpMetrics.pairedMse(ends, grp, late, h -> covered.test(h) && bins[h] == chosen));
    }
    LateReport.Enumeration enumeration = report.enumeration();
    Assert.assertEquals(enumeration.filled(), filled / handEnds, 1e-15);
    Assert.assertEquals(enumeration.terminal(), terminal / handEnds, 1e-15);
    Assert.assertEquals(enumeration.settled(), settled / handEnds, 1e-12);
    Assert.assertEquals(enumeration.leaves(), leaves / handEnds, 1e-12);
    Assert.assertTrue(enumeration.filled() > 0 && enumeration.terminal() > 0, enumeration.toString());
    Assert.assertTrue(enumeration.leaves() > 0 && enumeration.settled() > enumeration.leaves(), enumeration.toString());
    Assert.assertEquals(LateReport.measure(ends, grp, late).enumeration(), new LateReport.Enumeration(0, 0, 0, 0));
    List<LateReport.Bin> again = LateReport.measure(ends, late, grp).bins();
    Assert.assertEquals(again, report.bins());
  }
}

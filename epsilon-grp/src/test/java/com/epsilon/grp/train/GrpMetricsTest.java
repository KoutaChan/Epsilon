package com.epsilon.grp.train;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpPhase;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpTensor;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 答えの分かる合成データで評価の指標を確かめる。最終順位は卓に依らず一様なので、較正された Φ は E[u] = 0、周辺確率は 1/4 になる。
 * D_SPG は、仕込んだずれを 10% 以内の誤差で取り戻す。ノイズの分は v_c で引かれるので、ずれの無い Φ の D は 0 に近い。b4 は、ずれを
 * 仕込んだ区分でだけ動く。Φ = u ＋ ノイズのとき、u を Φ に回帰した傾きは var(u)/(var(u) ＋ σ²) になる。
 */
public class GrpMetricsTest {

  private static final int GAMES = 20_000;
  private static final int HANDS = 6;
  private static final double NOISE = 0.2;
  private static final double BIAS = 0.2;

  /** u は 2/3・1/3・0・−1 が等しく出るので E[u] = 0、var(u) = 7/18。 */
  private static final double VARIANCE = 7.0 / 18;

  /** ずれを仕込む区分。終わった局が東 3 局の局末で、そのとき最下位の席の Φ にずれを入れる。 */
  private static final int BIASED_GROUP = 2;

  @Test
  public void dSpgRecoversAnInjectedBiasWithoutNoise() {
    HandEnds ends = uniformEnds(3);
    GrpReport clean = GrpMetrics.measure(ends, constant(ends, 0, 0));
    GrpReport biased = GrpMetrics.measure(ends, constant(ends, BIAS, 0));
    // ずれがあるのは区分 2 で最下位の席だけなので、席と局末の組の 1/24 に当たる
    double expected = BIAS * BIAS / (HANDS * 4);
    Assert.assertEquals(biased.dSpg(), expected, 0.1 * expected, "D_SPG " + biased.dSpg());
    Assert.assertEquals(biased.dSb(), expected, 0.1 * expected, "D_SB " + biased.dSb());
    Assert.assertTrue(Math.abs(clean.dSpg()) < 0.05 * expected, "clean D_SPG " + clean.dSpg());
    Assert.assertTrue(Math.abs(clean.dSb()) < 0.05 * expected, "clean D_SB " + clean.dSb());
    Assert.assertEquals(clean.mse(), VARIANCE, 0.005);
    Assert.assertEquals(biased.games(), GAMES);
    Assert.assertEquals(biased.seatHands(), (long) GAMES * HANDS * 4);
  }

  @Test
  public void b4MovesOnlyInTheBiasedGroup() {
    HandEnds ends = uniformEnds(3);
    GrpReport report = GrpMetrics.measure(ends, constant(ends, BIAS, 0));
    for (int group = 0; group < HANDS; group++) {
      double expected = group == BIASED_GROUP ? BIAS : 0;
      Assert.assertEquals(report.groups().get(group).b4(), expected, 0.015, "b4 of group " + group);
    }
    Assert.assertTrue(Double.isNaN(report.groups().get(7).b4()), "group 7 has no hand ends");
    Paired residual = GrpMetrics.residual(
        ends, constant(ends, BIAS, 0), (group, rank, gap) -> group == BIASED_GROUP && rank == 3);
    Assert.assertEquals(residual.mean(), BIAS, 0.015);
    GrpReport subset = GrpMetrics.measure(ends, constant(ends, BIAS, 0), h -> ends.group(h) == BIASED_GROUP);
    Assert.assertEquals(subset.games(), GAMES);
    Assert.assertEquals(subset.seatHands(), (long) GAMES * 4);
    Assert.assertEquals(subset.groups().get(BIASED_GROUP).b4(), report.groups().get(BIASED_GROUP).b4(), 1e-12);
    Assert.assertEquals(subset.groups().get(BIASED_GROUP - 1).seatHands(), 0);
  }

  @Test
  public void pairedMseIsTheMseDifference() {
    HandEnds ends = uniformEnds(3);
    HandEndValues clean = constant(ends, 0, 0);
    HandEndValues biased = constant(ends, BIAS, 0);
    Paired paired = GrpMetrics.pairedMse(ends, clean, biased);
    double difference = GrpMetrics.mse(ends, biased) - GrpMetrics.mse(ends, clean);
    Assert.assertEquals(paired.mean(), difference, 1e-12);
    double reported = GrpMetrics.measure(ends, biased).mse() - GrpMetrics.measure(ends, clean).mse();
    Assert.assertEquals(reported, difference, 1e-12);
    Assert.assertTrue(paired.standardError() > 0 && paired.z() > 10, paired.toString());
    Assert.assertEquals(GrpMetrics.wobble(ends, clean, constant(ends, 0, 0.1)), 0.1, 1e-12);
  }

  @Test
  public void slopesOfNoisyPhiShrinkByTheNoise() {
    HandEnds ends = uniformEnds(3);
    GrpReport report = GrpMetrics.measure(ends, noisy(ends, new SplittableRandom(5)));
    double slope = VARIANCE / (VARIANCE + NOISE * NOISE);
    for (int group = 0; group < HANDS; group++) {
      GrpReport.Fit fit = report.groups().get(group);
      Assert.assertEquals(fit.slope(), slope, 0.02, "slope of group " + group);
      Assert.assertTrue(fit.slopeZ() < -3.2 && fit.slopeSe() > 0, "group " + group + ": " + fit);
    }
    // 次局の番号（東 1 局が 0）が 1〜3 なら東場、4・5 なら南 1〜2 局、6 なら南 3 局の段階に入る
    int[] handsPerPhase = {3, 2, 1, 0};
    for (GrpPhase phase : GrpPhase.values()) {
      GrpReport.Fit fit = report.phases().get(phase.ordinal());
      Assert.assertEquals(fit.seatHands(), (long) GAMES * 4 * handsPerPhase[phase.ordinal()], phase.toString());
      if (handsPerPhase[phase.ordinal()] > 0) {
        Assert.assertEquals(fit.slope(), slope, 0.02, "slope of " + phase);
      }
    }
    HandEnds few = HandEnds.of(Variant.TENHOU_4P, RankUtility.TENHOU,
        TrainFixtures.uniform(Fits.MIN_SLOPE_GAMES - 1, HANDS, new SplittableRandom(3)));
    GrpReport small = GrpMetrics.measure(few, noisy(few, new SplittableRandom(5)));
    for (int group = 0; group < HANDS; group++) {
      Assert.assertTrue(Double.isNaN(small.groups().get(group).slopeSe()), "SE of group " + group);
      Assert.assertTrue(Double.isNaN(small.groups().get(group).slopeZ()), "z of group " + group);
    }
  }

  @Test
  public void exactRanksHaveNoLoss() {
    HandEnds ends = HandEnds.of(
        Variant.TENHOU_4P, RankUtility.TENHOU, TrainFixtures.uniform(500, HANDS, new SplittableRandom(4)));
    HandEndValues.Builder builder = HandEndValues.builder(ends);
    for (int h = 0; h < ends.handEnds(); h++) {
      double[] phi = new double[4];
      double[] ranks = new double[16];
      for (int seat = 0; seat < 4; seat++) {
        phi[seat] = ends.utility(h, seat);
        ranks[4 * seat + ends.finalRank(h, seat)] = 1;
      }
      builder.set(h, phi, ranks, 0);
    }
    GrpReport report = GrpMetrics.measure(ends, builder.build());
    Assert.assertEquals(report.mse(), 0, 1e-15);
    Assert.assertEquals(report.rankNll(), 0, 1e-12);
    Assert.assertEquals(report.brier(), 0, 1e-15);
    Assert.assertEquals(report.fourthEce(), 0, 1e-15);
    Assert.assertEquals(report.jointNll(), 0, 0);
    Assert.assertTrue(Double.isNaN(GrpMetrics.measure(ends, constant(ends, 0, 0)).jointNll()));
  }

  @Test
  public void calibratedLateCellsStayWithinTheirLines() {
    HandEnds ends = lateEnds(new SplittableRandom(8), false);
    GrpReport report = GrpMetrics.measure(ends, constant(ends, 0, 0));
    Assert.assertEquals(report.closeCells().size(), 8);
    for (GrpReport.CloseCell cell : report.closeCells()) {
      Assert.assertTrue(cell.group() >= 6 && cell.rank() >= 2 && cell.seatHands() > 1_000, cell.toString());
      Assert.assertTrue(Math.abs(cell.z()) < 4, cell.toString());
    }
    Assert.assertEquals(report.jumps().size(), 2 * 4 * 2);
    double chiSquare = 0;
    for (GrpReport.JumpSeries series : report.jumps()) {
      Assert.assertTrue(series.pairs() >= 20 && series.maxAbsZ() < 5, series.toString());
      chiSquare += series.chiSquarePerDf() / report.jumps().size();
    }
    Assert.assertTrue(chiSquare > 0.6 && chiSquare < 1.5, "mean chi-square per df " + chiSquare);
  }

  @Test
  public void aKeepStepShowsAsAJumpAndAMissedCell() {
    HandEnds ends = lateEnds(new SplittableRandom(9), true);
    GrpReport report = GrpMetrics.measure(ends, constant(ends, 0, 0));
    GrpReport.JumpSeries leader = report.jumps().stream()
        .filter(series -> series.group() == 6 && series.rank() == 0)
        .filter(series -> series.measure() == GrpReport.JumpMeasure.KEEP)
        .findFirst()
        .orElseThrow();
    Assert.assertTrue(leader.maxAbsZ() > 10 && !leader.within(), leader.toString());
    // 首位が 2 着を 12,000 点以上離した局末では、首位は必ず首位のまま終わる。そのため、最下位になる予測 1/4 は端の基準を外れる
    GrpReport.Tail leaderLast = report.tails().leaderLast();
    Assert.assertTrue(leaderLast.rows() > 0 && leaderLast.realized() == 0, leaderLast.toString());
    Assert.assertFalse(leaderLast.within(), leaderLast.toString());
  }

  /**
   * 95% 以上の周辺確率は、外れる側の率で比べる。1 着の確率を 0.952 と予測して実測が 0.9965 なら、外れる側の予測 0.048 が
   * 実測 0.0035 の 2 倍を超えるので、基準を外れる。0.999 と予測して実測が 0.99 なら、外れる側の予測 0.001 が実測 0.01 の半分に
   * 届かないので、やはり外れる。
   */
  @Test
  public void highProbabilitiesAreJudgedOnTheirMisses() {
    HandEnds ends = uniformEnds(13);
    HandEndValues underConfident = leaderValues(ends, 0.952, 0.9965, 7);
    GrpReport.Tail under = GrpMetrics.measure(ends, underConfident).tails().highProbability();
    Assert.assertTrue(under.rows() > 1_000, under.toString());
    Assert.assertEquals(under.predicted(), 0.048, 1e-9);
    Assert.assertEquals(under.realized(), 0.0035, 0.0015);
    Assert.assertFalse(under.within(), under.toString());
    GrpReport.Tail over = GrpMetrics.measure(ends, leaderValues(ends, 0.999, 0.99, 8)).tails().highProbability();
    Assert.assertEquals(over.predicted(), 0.001, 1e-9);
    Assert.assertFalse(over.within(), over.toString());
    GrpReport.Tail calibrated = GrpMetrics.measure(ends, leaderValues(ends, 0.97, 0.97, 9)).tails().highProbability();
    Assert.assertTrue(calibrated.within(), calibrated.toString());
  }

  @Test
  public void acceptanceLinesOfTheReport() {
    // 実測 0.01・SE 0.0002 なら基準は [0.0046, 0.0204]
    Assert.assertTrue(new GrpReport.Tail(100, 0.012, 0.01, 0.002).within());
    Assert.assertTrue(new GrpReport.Tail(100, 0.024, 0.01, 0.002).within());
    Assert.assertTrue(new GrpReport.Tail(100, 0.0047, 0.01, 0.0002).within());
    Assert.assertFalse(new GrpReport.Tail(100, 0.0205, 0.01, 0.0002).within());
    Assert.assertFalse(new GrpReport.Tail(100, 0.0045, 0.01, 0.0002).within());
    Assert.assertTrue(new GrpReport.Tail(0, Double.NaN, Double.NaN, Double.NaN).within());
    Assert.assertTrue(new GrpReport.CloseCell(6, 3, 0, 2_000, 10, 0.009, 0.001).within());
    Assert.assertTrue(new GrpReport.CloseCell(6, 3, 0, 2_000, 10, 0.05, 0.02).within());
    Assert.assertFalse(new GrpReport.CloseCell(6, 3, 0, 2_000, 10, 0.05, 0.01).within());
    Assert.assertTrue(new GrpReport.Fit(10, -0.004, 1.009, 0.001, 9).within());
    Assert.assertFalse(new GrpReport.Fit(10, -0.006, 1.0, 0.001, 0).within());
    Assert.assertFalse(new GrpReport.Fit(10, 0, 1.011, 0.001, 11).within());
    Assert.assertTrue(new GrpReport.JumpSeries(6, 0, GrpReport.JumpMeasure.KEEP, 23, 1.4, 3.9, 1.9).within());
    Assert.assertFalse(new GrpReport.JumpSeries(6, 0, GrpReport.JumpMeasure.KEEP, 23, 1.4, 4.1, 1.9).within());
    Assert.assertFalse(new GrpReport.JumpSeries(6, 0, GrpReport.JumpMeasure.KEEP, 23, 1.6, 3.9, 1.9).within());
    Assert.assertFalse(new GrpReport.JumpSeries(6, 0, GrpReport.JumpMeasure.KEEP, 23, 1.4, 3.9, 2.1).within());
    Assert.assertTrue(new GrpReport.JumpSeries(6, 0, GrpReport.JumpMeasure.PHI, 23, 1.4, 3.9, Double.NaN).within());
  }

  @Test
  public void networkValuesMatchPhiAndCarryTheJointLikelihood() {
    for (Variant variant : Variant.values()) {
      List<GameRecord> games = TrainFixtures.simulated(variant, 300, 0, 17);
      HandEnds ends = HandEnds.of(variant, RankUtility.TENHOU, games);
      GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(variant, 12, true), 19, 0.7);
      GrpWorkspace workspace = GrpWorkspace.of(network.shape());
      HandEndValues values = HandEndValues.of(ends, network, workspace);
      Assert.assertTrue(values.finite() && values.joint());
      int seats = variant.seats();
      double[] phi = new double[seats];
      for (int h = 0; h < ends.handEnds(); h++) {
        GrpState state = ends.state(h);
        network.phi(state, RankUtility.TENHOU, workspace, phi);
        for (int seat = 0; seat < seats; seat++) {
          Assert.assertEquals(Double.doubleToRawLongBits(values.phi(h, seat)), Double.doubleToRawLongBits(phi[seat]));
          double row = 0;
          double column = 0;
          for (int other = 0; other < seats; other++) {
            row += values.marginal(h, seat, other);
            column += values.marginal(h, other, seat);
          }
          Assert.assertEquals(row, 1, 1e-12);
          Assert.assertEquals(column, 1, 1e-12);
        }
        Assert.assertEquals(
            values.jointLog(h), network.logProbability(state, ends.permutation(h), workspace), 1e-9);
      }
      GrpReport report = GrpMetrics.measure(ends, values);
      Assert.assertEquals(report.groups().size(), variant.lastRound() + 1);
      Assert.assertEquals(report.closeCells().size(), 8);
      Assert.assertEquals(report.jumps().size(), 2 * seats * 2);
      Assert.assertTrue(Double.isFinite(report.jointNll()) && report.jointNll() > 0);
      Assert.assertTrue(Double.isFinite(report.dSpg()) && Double.isFinite(report.mse()));
    }
  }

  @Test
  public void tieJumpsOfAFlatNetworkAreZero() {
    List<GameRecord> games = TrainFixtures.simulated(Variant.TENHOU_4P, 5_000, 0, 23);
    HandEnds ends = HandEnds.of(Variant.TENHOU_4P, RankUtility.TENHOU, games);
    GrpNetwork random = GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 8, false), 29, 1.0);
    float[] weights = random.weights();
    int[] starts = GrpTensor.starts(random.shape());
    Arrays.fill(weights, starts[GrpTensor.W3.ordinal()], starts[GrpTensor.W3.ordinal() + 1], 0f);
    GrpNetwork flat = GrpNetwork.of(random.shape(), weights);
    TieJump jump = TieJumps.measure(ends, flat, GrpWorkspace.of(flat.shape()));
    Assert.assertTrue(jump.rows() > 100, jump.rows() + " near ties");
    Assert.assertEquals(jump.predicted(), 0, 0);
    Assert.assertEquals(jump.predictedCurve().length, 11);
    for (double value : jump.predictedCurve()) {
      Assert.assertEquals(value, jump.predictedCurve()[0], 1e-12);
    }
    // 2 席の同点（d = 0）はどの局末でも起きる。終局の点数が同点なら起家に近い席が上なので、率は 1/2 より上に寄る
    double tie = jump.realizedCurve()[5];
    Assert.assertTrue(tie > 0.5 && tie <= 1, "tie rate " + tie);
    TieJump learned = TieJumps.measure(ends, random, GrpWorkspace.of(random.shape()));
    Assert.assertEquals(learned.rows(), jump.rows());
    Assert.assertTrue(Double.isFinite(learned.predicted()), learned.toString());
    for (double value : learned.predictedCurve()) {
      Assert.assertTrue(value >= 0 && value <= 1, Double.toString(value));
    }
  }

  /** 最終順位が一様な四人打ちの半荘から局末を作る。20,000 半荘 × 6 局末で、終わった局の区分は 0〜5。 */
  private static HandEnds uniformEnds(long seed) {
    List<GameRecord> games = TrainFixtures.uniform(GAMES, HANDS, new SplittableRandom(seed));
    return HandEnds.of(Variant.TENHOU_4P, RankUtility.TENHOU, games);
  }

  /**
   * 終盤の区分の局末が多い、四人打ちの半荘の局末を作る。局末 0〜2 の次局は南 3・南 4・西 1 なので、局末 1・2 の終わった局は
   * 南 3（区分 6）と南 4（区分 7）になる。この 2 つの局末の点数は同じ。
   *
   * @param stepped true なら、局末 1 の首位が 2 着を 2,000 点以上離しているとき、首位のまま終わる（ほかの席の順は一様）。
   *     false なら全ての順位が一様
   */
  private static HandEnds lateEnds(SplittableRandom random, boolean stepped) {
    List<GameRecord> games = new ArrayList<>(GAMES);
    for (int game = 0; game < GAMES; game++) {
      int[] scores = wide(random);
      List<GrpState> handEnds = List.of(
          new GrpState(Variant.TENHOU_4P, 6, 0, 0, wide(random)),
          new GrpState(Variant.TENHOU_4P, 7, 0, 0, scores),
          new GrpState(Variant.TENHOU_4P, 8, 0, 0, scores));
      int[] order = {0, 1, 2, 3};
      for (int index = 3; index > 0; index--) {
        int other = random.nextInt(index + 1);
        int held = order[index];
        order[index] = order[other];
        order[other] = held;
      }
      GrpState middle = handEnds.get(1);
      int leader = 0;
      for (int seat = 1; seat < 4; seat++) {
        if (middle.rank(seat) == 0) {
          leader = seat;
        }
      }
      int second = 0;
      for (int seat = 0; seat < 4; seat++) {
        if (middle.rank(seat) == 1) {
          second = seat;
        }
      }
      if (stepped && middle.score(leader) - middle.score(second) >= 2_000) {
        for (int seat = 0; seat < 4; seat++) {
          if (order[seat] == 0) {
            order[seat] = order[leader];
          }
        }
        order[leader] = 0;
      }
      int[] finals = {40_000, 30_000, 20_000, 10_000};
      int[] finalScores = new int[4];
      for (int seat = 0; seat < 4; seat++) {
        finalScores[seat] = finals[order[seat]];
      }
      games.add(new GameRecord(Variant.TENHOU_4P, game, handEnds, finalScores));
    }
    return HandEnds.of(Variant.TENHOU_4P, RankUtility.TENHOU, games);
  }

  /** 点数を広く散らして配る。3 席は 25,000 点 ± 12,000 点から 100 点刻みで一様に選び、残りの 1 席で和を 100,000 点にする。 */
  private static int[] wide(SplittableRandom random) {
    int[] scores = new int[4];
    int sum = 0;
    for (int seat = 0; seat < 3; seat++) {
      scores[seat] = 25_000 + 100 * (random.nextInt(241) - 120);
      sum += scores[seat];
    }
    scores[3] = 100_000 - sum;
    return scores;
  }

  /**
   * 全ての席の Φ を同じ値にし、区分 2 の局末でだけ、そのとき最下位の席の Φ を下げる。周辺確率は一様にする。
   *
   * @param bias 区分 2 で最下位の席の Φ を下げる幅
   * @param shift 全ての席の Φ
   */
  private static HandEndValues constant(HandEnds ends, double bias, double shift) {
    HandEndValues.Builder builder = HandEndValues.builder(ends);
    double[] uniform = new double[16];
    Arrays.fill(uniform, 0.25);
    for (int h = 0; h < ends.handEnds(); h++) {
      double[] phi = new double[4];
      for (int seat = 0; seat < 4; seat++) {
        phi[seat] = shift - (ends.group(h) == BIASED_GROUP && ends.rank(h, seat) == 3 ? bias : 0);
      }
      builder.set(h, phi, uniform);
    }
    return builder.build();
  }

  /** Φ = u ＋ N(0, σ) とした局末の値を作る。周辺確率は一様にする。 */
  private static HandEndValues noisy(HandEnds ends, SplittableRandom random) {
    HandEndValues.Builder builder = HandEndValues.builder(ends);
    double[] uniform = new double[16];
    Arrays.fill(uniform, 0.25);
    for (int h = 0; h < ends.handEnds(); h++) {
      double[] phi = new double[4];
      for (int seat = 0; seat < 4; seat++) {
        phi[seat] = ends.utility(h, seat) + NOISE * random.nextGaussian();
      }
      builder.set(h, phi, uniform);
    }
    return builder.build();
  }

  /**
   * 局末ごとに 1 席を選び、その席の 1 着の確率を指定した値を作る。残りの確率はほかの順位と席に等分し、Φ は 0 にする。
   *
   * @param p 選んだ席の 1 着の確率
   * @param rate 実際に 1 着で終わった席を選ぶ率。選ばないときは、ほかの 3 席から一様に選ぶ
   */
  private static HandEndValues leaderValues(HandEnds ends, double p, double rate, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    HandEndValues.Builder builder = HandEndValues.builder(ends);
    double[] phi = new double[4];
    double[] marginals = new double[16];
    for (int h = 0; h < ends.handEnds(); h++) {
      int first = 0;
      while (ends.finalRank(h, first) != 0) {
        first++;
      }
      int chosen = random.nextDouble() < rate ? first : (first + 1 + random.nextInt(3)) % 4;
      for (int seat = 0; seat < 4; seat++) {
        double top = seat == chosen ? p : (1 - p) / 3;
        marginals[seat * 4] = top;
        for (int rank = 1; rank < 4; rank++) {
          marginals[seat * 4 + rank] = (1 - top) / 3;
        }
      }
      builder.set(h, phi, marginals);
    }
    return builder.build();
  }
}

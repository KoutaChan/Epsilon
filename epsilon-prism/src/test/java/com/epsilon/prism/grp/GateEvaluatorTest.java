package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 関門の各 enum に届く。真の Φ は次局の点数の Plackett–Luce（席 i の重み exp((s_i − s̄)/τ)、τ = 6000）で、取り置きの最終順位はそこから引く。
 * 使用中・候補は τ を変えたもの（小さい τ は自信過剰）・のこぎりの波を足したもの・NaN を返すもの。
 */
public class GateEvaluatorTest {
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;
  private static final double TRUTH = 6000;
  private static final List<RecordedGame> HOLDOUT = holdout(new SplittableRandom(1), 40_000);

  @Test
  public void refitPassesACalibratedCandidate() throws IOException {
    GateReport report = refit(new PlackettLuce(5700), new PlackettLuce(TRUTH));
    Assert.assertEquals(report.result(), GateResult.PASS, report.failures() + " " + report.measures());
    Assert.assertTrue(report.passed());
    Assert.assertTrue(report.measures().get("mseDifference") < 0);
    Assert.assertEquals(report.measures().get("monotonicCandidate"), 0.0);
  }

  @Test
  public void refitDecidesOnAccuracyAndRecordsTheRest() throws IOException {
    Assert.assertEquals(refit(new PlackettLuce(5700), new PlackettLuce(5700)).result(), GateResult.NOT_BETTER);
    Assert.assertEquals(refit(new PlackettLuce(5700), new PlackettLuce(12000)).result(), GateResult.NOT_BETTER);
    // 効用は真の値で順位分布だけ自信過剰: 当たるので入れ、順位の当てはまりは記録だけ
    GateReport overconfident = refit(new PlackettLuce(5700), new PlackettLuce(TRUTH, 1500, 0));
    Assert.assertEquals(overconfident.result(), GateResult.PASS, overconfident.failures() + " " + overconfident.measures());
    Assert.assertTrue(overconfident.measures().containsKey("nllDifference"));
    // 使用中から大きく変わっても当たるなら入れ、変化の幅と傾きは記録だけ
    GateReport far = refit(new PlackettLuce(3000), new PlackettLuce(TRUTH));
    Assert.assertEquals(far.result(), GateResult.PASS, far.failures() + " " + far.measures());
    Assert.assertTrue(far.measures().containsKey("diffShift") && far.measures().containsKey("slopeRatio"));
    Assert.assertTrue(
        refit(new PlackettLuce(5700), new PlackettLuce(TRUTH, TRUTH, 0.08))
            .failures()
            .contains(GateResult.MONOTONIC_VIOLATIONS));
    HandEndPhi broken =
        inputs -> {
          PhiValues values = new PlackettLuce(TRUTH).values(inputs);
          values.utilities()[0][0] = Double.NaN;
          return values;
        };
    Assert.assertEquals(refit(new PlackettLuce(5700), broken).result(), GateResult.INVALID_OUTPUT);
    GateReport invalid =
        GateEvaluator.evaluate(
            new GateInput(GateMode.REFIT, HOLDOUT, new PlackettLuce(5700), new PlackettLuce(TRUTH), false, 1, 1),
            PROFILE);
    Assert.assertEquals(invalid.result(), GateResult.INVALID_OUTPUT);
    GateReport empty =
        GateEvaluator.evaluate(
            new GateInput(GateMode.REFIT, List.of(), new PlackettLuce(5700), new PlackettLuce(TRUTH), true, 1, 1),
            PROFILE);
    Assert.assertEquals(empty.result(), GateResult.NOT_ENOUGH_DATA);
  }

  @Test
  public void firstReplacementComparesWithTheScoreOnlyPhi() throws IOException {
    HandEndPhi truth = new PlackettLuce(TRUTH);
    GateReport pass = first(truth, 200_000, 131_072);
    Assert.assertEquals(pass.result(), GateResult.PASS, pass.failures() + " " + pass.measures());
    Assert.assertTrue(pass.measures().get("mseDifference") < 0, pass.measures().toString());
    Assert.assertTrue(pass.measures().containsKey("calibrationRms"), "the calibration is recorded");
    // 外れた GRP でも、較正は記録だけで落とさない（点数だけの Φ より当たれば通る）
    GateReport sharp = first(new PlackettLuce(7500), 200_000, 131_072);
    Assert.assertEquals(sharp.result(), GateResult.PASS, sharp.failures() + " " + sharp.measures());
    Assert.assertEquals(first(score(), 200_000, 131_072).result(), GateResult.NOT_BETTER);
    HandEndPhi saw = new PlackettLuce(TRUTH, TRUTH, 0.08);
    Assert.assertTrue(first(saw, 200_000, 131_072).failures().contains(GateResult.MONOTONIC_VIOLATIONS));
    Assert.assertEquals(first(truth, 100_000, 131_072).result(), GateResult.NOT_ENOUGH_DATA);
  }

  private static GateReport refit(HandEndPhi active, HandEndPhi candidate) throws IOException {
    return GateEvaluator.evaluate(new GateInput(GateMode.REFIT, HOLDOUT, active, candidate, true, 0, 1), PROFILE);
  }

  private static GateReport first(HandEndPhi candidate, long window, int trust) throws IOException {
    return GateEvaluator.evaluate(
        new GateInput(GateMode.FIRST_REPLACEMENT, HOLDOUT, score(), candidate, true, window, trust), PROFILE);
  }

  private static HandEndPhi score() {
    return ScorePhi.offline(PhiEntryKind.SCORE_RANK, PROFILE);
  }

  /** 1 局末だけ続く半荘（次局は南 1 局、点数は 25,000 のまわりに SD 約 5,000）。最終順位は真の Plackett–Luce から引く。 */
  private static List<RecordedGame> holdout(SplittableRandom random, int count) {
    List<RecordedGame> games = new ArrayList<>(count);
    int[] finals = {40000, 30000, 20000, 10000};
    for (int g = 0; g < count; g++) {
      int[] scores = new int[4];
      int sum = 0;
      for (int seat = 0; seat < 3; seat++) {
        scores[seat] = 25000 + 100 * (int) Math.round(50 * gaussian(random));
        sum += scores[seat];
      }
      scores[3] = 100_000 - sum;
      double[] weights = PlackettLuce.weights(scores, TRUTH);
      boolean[] placed = new boolean[4];
      int[] finalScores = new int[4];
      for (int rank = 0; rank < 4; rank++) {
        double total = 0;
        for (int seat = 0; seat < 4; seat++) if (!placed[seat]) total += weights[seat];
        double draw = random.nextDouble() * total;
        int chosen = -1;
        for (int seat = 0; seat < 4 && chosen < 0; seat++) {
          if (placed[seat]) continue;
          draw -= weights[seat];
          if (draw <= 0) chosen = seat;
        }
        if (chosen < 0) for (int seat = 3; seat >= 0 && chosen < 0; seat--) if (!placed[seat]) chosen = seat;
        placed[chosen] = true;
        finalScores[chosen] = finals[rank];
      }
      games.add(new RecordedGame(g, List.of(new HandStart(4, 0, 0, scores)), finalScores));
    }
    return games;
  }

  private static double gaussian(SplittableRandom random) {
    double u1 = 1 - random.nextDouble(), u2 = random.nextDouble();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * 次局の点数の Plackett–Luce の Φ。効用は τ の順位分布の期待効用にのこぎりの波 saw·((s_i mod 2000)/2000 − 1/2) を足したもの、順位分布は
   * τ_rank のもの。
   */
  private record PlackettLuce(double tau, double rankTau, double saw) implements HandEndPhi {
    private static final int[][] ORDERS = orders();

    PlackettLuce(double tau) {
      this(tau, tau, 0);
    }

    @Override
    public PhiValues values(List<HandEndInput> inputs) {
      double[][] utilities = new double[inputs.size()][4], ranks = new double[inputs.size()][];
      for (int i = 0; i < inputs.size(); i++) {
        int[] scores = inputs.get(i).next().scores();
        double[] marginals = marginals(scores, tau);
        for (int seat = 0; seat < 4; seat++) {
          for (int rank = 0; rank < 4; rank++)
            utilities[i][seat] += marginals[4 * seat + rank] * PROFILE.utilityForRankDouble(rank);
          utilities[i][seat] += saw * (Math.floorMod(scores[seat], 2000) / 2000.0 - 0.5);
        }
        ranks[i] = rankTau == tau ? marginals : marginals(scores, rankTau);
      }
      return new PhiValues(utilities, ranks);
    }

    static double[] weights(int[] scores, double tau) {
      double mean = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;
      double[] weights = new double[4];
      for (int seat = 0; seat < 4; seat++) weights[seat] = Math.exp((scores[seat] - mean) / tau);
      return weights;
    }

    private static double[] marginals(int[] scores, double tau) {
      double[] weights = weights(scores, tau);
      double[] marginals = new double[16];
      for (int[] order : ORDERS) {
        double probability = 1;
        for (int rank = 0; rank < 4; rank++) {
          // 残りの重みは引き算せずに足し直す（重みの桁が離れると引き算で 0 や負になる）
          double remaining = 0;
          for (int later = rank; later < 4; later++) remaining += weights[order[later]];
          probability *= weights[order[rank]] / remaining;
        }
        for (int rank = 0; rank < 4; rank++) marginals[4 * order[rank] + rank] += probability;
      }
      return marginals;
    }

    private static int[][] orders() {
      List<int[]> orders = new ArrayList<>();
      for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
          for (int c = 0; c < 4; c++)
            if (a != b && b != c && a != c) orders.add(new int[] {a, b, c, 6 - a - b - c});
      return orders.toArray(int[][]::new);
    }
  }
}

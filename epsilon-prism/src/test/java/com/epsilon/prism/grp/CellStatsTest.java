package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.ScoreRanking;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 較正の測り: 較正された Φ（最終順位が一様なので Φ = E[u] = 0）の升目に仕込んだずれを D_SPG・D_SB が 10% 以内で取り戻し（雑音の分は v_c で
 * 引かれ、ずれのない Φ の D は 0 の近く）、B4 はずれの区分だけが動く。傾きは Φ = u + 雑音で var(u)/(var(u) + σ²) になる。対の差は MSE の差
 * そのもの。
 */
public class CellStatsTest {
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;
  private static final int GAMES = 20_000, HANDS = 6;
  private static final double NOISE = 0.2, BIAS = 0.2;

  /** u は 2/3・1/3・0・−1 が等しく出るので E[u] = 0、var(u) = 7/18。 */
  private static final double VARIANCE = 7.0 / 18;

  /** 東 3 局が終わった局末で今 4 位の席。 */
  private static boolean biased(HandGroup group, int rank) {
    return group == HandGroup.E3 && rank == 3;
  }

  @Test
  public void recoversAnInjectedCellBias() {
    List<RecordedGame> games = games(new SplittableRandom(3));
    PhiValues clean = biasedZero(games, 0);
    PhiValues shifted = biasedZero(games, BIAS);
    CellReport none = CellStats.measure(games, clean, PROFILE);
    CellReport report = CellStats.measure(games, shifted, PROFILE);
    // ずれの升目は E3 の 4 位の席全て（全席局の 1/24）。D_SB でもずれた席は別の Φ の区間に入る
    double expected = BIAS * BIAS / (HANDS * 4);
    Assert.assertEquals(report.dSpg(), expected, 0.1 * expected, "D_SPG " + report.dSpg());
    Assert.assertEquals(report.dSb(), expected, 0.1 * expected, "D_SB " + report.dSb());
    Assert.assertTrue(Math.abs(none.dSpg()) < 0.05 * expected, "clean D_SPG " + none.dSpg());
    Assert.assertTrue(Math.abs(none.dSb()) < 0.05 * expected, "clean D_SB " + none.dSb());
    Assert.assertEquals(report.b4()[HandGroup.E3.ordinal()], BIAS, 0.015);
    Assert.assertEquals(report.b4()[HandGroup.E2.ordinal()], 0, 0.015);
    Assert.assertEquals(none.mse(), VARIANCE, 0.005);
    Assert.assertEquals(report.games(), GAMES);
    Assert.assertEquals(report.seatHands(), (long) GAMES * HANDS * 4);
    var paired = CellStats.paired(games, clean, shifted, PROFILE);
    Assert.assertEquals(paired.mseDifference(), report.mse() - none.mse(), 1e-12);
    Assert.assertTrue(paired.standardError() > 0 && paired.mseDifference() > 10 * paired.standardError());
    var residual = CellStats.residual(games, shifted, PROFILE, (group, rank, gap) -> biased(group, rank));
    Assert.assertEquals(residual.mseDifference(), BIAS, 0.015);
  }

  @Test
  public void slopeOfNoisyPhiShrinks() {
    List<RecordedGame> games = games(new SplittableRandom(3));
    CellReport report = CellStats.measure(games, noisy(games, new SplittableRandom(5)), PROFILE);
    double slope = VARIANCE / (VARIANCE + NOISE * NOISE);
    for (int group = 0; group < HANDS; group++) {
      Assert.assertEquals(report.slope()[group], slope, 0.02, "slope of group " + group);
      Assert.assertTrue(report.slopeZ()[group] < -3.2, "z of group " + group);
      Assert.assertTrue(report.slopeSe()[group] > 0, "SE of group " + group);
    }
    // 半荘が少ない区分は SE と z を出さない（SE がほぼ 0 になり z が発散する）
    List<RecordedGame> few = games.subList(0, CellStats.MIN_SLOPE_GAMES - 1);
    CellReport small = CellStats.measure(few, noisy(few, new SplittableRandom(5)), PROFILE);
    for (int group = 0; group < HANDS; group++) {
      Assert.assertTrue(Double.isNaN(small.slopeZ()[group]), "z of group " + group);
      Assert.assertTrue(Double.isNaN(small.slopeSe()[group]), "SE of group " + group);
    }
  }

  @Test
  public void exactRanksHaveNoRankLoss() {
    List<RecordedGame> games = games(new SplittableRandom(4)).subList(0, 500);
    int handEnds = games.size() * HANDS;
    double[][] utilities = new double[handEnds][], ranks = new double[handEnds][16];
    int h = 0;
    for (RecordedGame game : games)
      for (int k = 0; k < HANDS; k++, h++) {
        utilities[h] = game.finalUtilities(PROFILE);
        int[] finalRanks = game.finalRanks();
        for (int seat = 0; seat < 4; seat++) ranks[h][4 * seat + finalRanks[seat]] = 1;
      }
    CellReport report = CellStats.measure(games, new PhiValues(utilities, ranks), PROFILE);
    Assert.assertEquals(report.mse(), 0, 1e-15);
    Assert.assertEquals(report.rankNll(), 0, 1e-12);
    Assert.assertEquals(report.brier(), 0, 1e-15);
    Assert.assertEquals(report.fourthEce(), 0, 1e-15);
  }

  @Test
  public void bootstrapSpreadIsPositiveAndRepeatable() {
    List<RecordedGame> games = games(new SplittableRandom(6)).subList(0, 4000);
    PhiValues phi = noisy(games, new SplittableRandom(7));
    double sd = CellStats.bootstrapDSpgSd(games, phi, PROFILE, 50, 11);
    Assert.assertTrue(sd > 0);
    Assert.assertEquals(CellStats.bootstrapDSpgSd(games, phi, PROFILE, 50, 11), sd);
  }

  /** 局末 k の次局は局番号 k + 1（終わった局は東 1〜南 2）、点数は 25,000 のまわりに散らし、最終順位は一様。 */
  static List<RecordedGame> games(SplittableRandom random) {
    List<RecordedGame> games = new ArrayList<>(GAMES);
    int[] finals = {40000, 30000, 20000, 10000};
    for (int g = 0; g < GAMES; g++) {
      List<HandStart> starts = new ArrayList<>(HANDS);
      for (int k = 0; k < HANDS; k++) starts.add(new HandStart(k + 1, 0, 0, scores(random)));
      int[] order = {0, 1, 2, 3};
      for (int i = 3; i > 0; i--) {
        int j = random.nextInt(i + 1);
        int swap = order[i];
        order[i] = order[j];
        order[j] = swap;
      }
      int[] finalScores = new int[4];
      for (int seat = 0; seat < 4; seat++) finalScores[seat] = finals[order[seat]];
      games.add(new RecordedGame(g, starts, finalScores));
    }
    return games;
  }

  private static int[] scores(SplittableRandom random) {
    int[] scores = new int[4];
    int sum = 0;
    for (int seat = 0; seat < 3; seat++) {
      scores[seat] = 25000 + 100 * (random.nextInt(201) - 100);
      sum += scores[seat];
    }
    scores[3] = 100_000 - sum;
    return scores;
  }

  /** Φ = 0（較正された値）、ずれの升目の席だけ −bias（残差の平均が bias）。順位分布は一様。 */
  private static PhiValues biasedZero(List<RecordedGame> games, double bias) {
    return values(games, (game, k, seat, now) -> biased(CellStats.endedGroup(game, k), now[seat]) ? -bias : 0);
  }

  /** Φ = u + N(0, σ)。 */
  private static PhiValues noisy(List<RecordedGame> games, SplittableRandom random) {
    return values(games, (game, k, seat, now) -> game.finalUtilities(PROFILE)[seat] + NOISE * gaussian(random));
  }

  private interface SeatValue {
    double of(RecordedGame game, int k, int seat, int[] now);
  }

  private static PhiValues values(List<RecordedGame> games, SeatValue value) {
    List<double[]> utilities = new ArrayList<>();
    List<double[]> ranks = new ArrayList<>();
    for (RecordedGame game : games)
      for (int k = 0; k < game.continuingHandEnds(); k++) {
        int[] now = ScoreRanking.byScoreThenSeat(game.starts().get(k).scores());
        double[] values = new double[4];
        for (int seat = 0; seat < 4; seat++) values[seat] = value.of(game, k, seat, now);
        utilities.add(values);
        double[] uniform = new double[16];
        Arrays.fill(uniform, 0.25);
        ranks.add(uniform);
      }
    return new PhiValues(utilities.toArray(double[][]::new), ranks.toArray(double[][]::new));
  }

  private static double gaussian(SplittableRandom random) {
    double u1 = 1 - random.nextDouble(), u2 = random.nextDouble();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

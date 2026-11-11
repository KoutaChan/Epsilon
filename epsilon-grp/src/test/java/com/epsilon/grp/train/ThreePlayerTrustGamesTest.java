package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * {@link TrustGames} が見積もる半荘の数を、三人打ちの半荘で確かめる。これは、点数だけの Φ を初めて置き換える関門で、2 SE の判定に
 * 要る半荘の数。
 * 半荘ごとの局末の数と、対の MSE の差が半荘ごとにどれだけ散らばるかを決めて半荘を作るので、差・標準偏差・要る半荘の数の答えを
 * テストの側で求められる。
 *
 * <p>使用中の Φ は、全ての席・局末で u ＋ 1（残差の二乗は 1）。候補の Φ は、半荘 g の全ての席・局末で u ＋ √(1 ＋ x_g)（残差の
 * 二乗は 1 ＋ x_g）。x_g は平均 {@value #DIFFERENCE}、標準偏差 {@value #SPREAD} の正規分布に従う。そのため対の差は、半荘 g の
 * 席・局末ごとに x_g になる。
 */
public class ThreePlayerTrustGamesTest {

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final RankUtility UTILITY = RankUtility.TENHOU;

  /** 半荘を作るときの差（候補 − 使用中、席・局末あたり）。 */
  private static final double DIFFERENCE = -0.3;

  /** 差が半荘ごとに散らばる大きさ（標準偏差）。 */
  private static final double SPREAD = 0.15;

  /** 局末を持つ半荘の数。 */
  private static final int GAMES = 4000;

  /** 局末を持たない半荘の数。これらの半荘は数に入らない。 */
  private static final int EMPTY = 25;

  private static final int RESAMPLES = 4000;

  /** 最終の点の組。半荘ごとに乱数で並べ替えて配る。 */
  private static final int[] FINALS = {50_000, 35_000, 20_000};

  /**
   * 局末を 1〜8 個持つ三人打ちの半荘 {@code count} 個と、局末を持たない半荘 {@code empty} 個（1 以上）を、等しい間隔で混ぜて並べる。
   * 局末 k の次局は k ＋ 1。最終の点は、局末の点とは関係なく乱数で並べ替える。
   */
  static List<GameRecord> games(int count, int empty, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    int stride = (count + empty) / empty;
    List<GameRecord> games = new ArrayList<>(count + empty);
    for (int game = 0; game < count + empty; game++) {
      boolean none = game % stride == 0 && game / stride < empty;
      int hands = none ? 0 : 1 + random.nextInt(8);
      List<GrpState> handEnds = new ArrayList<>(hands);
      for (int hand = 0; hand < hands; hand++) {
        int first = 35_000 + 100 * (random.nextInt(201) - 100);
        int second = 35_000 + 100 * (random.nextInt(201) - 100);
        handEnds.add(new GrpState(THREE, hand + 1, 0, 0, new int[] {first, second, 105_000 - first - second}));
      }
      games.add(new GameRecord(THREE, game, handEnds, shuffled(random)));
    }
    return games;
  }

  private static int[] shuffled(SplittableRandom random) {
    int[] scores = FINALS.clone();
    for (int index = scores.length - 1; index > 0; index--) {
      int other = random.nextInt(index + 1);
      int held = scores[index];
      scores[index] = scores[other];
      scores[other] = held;
    }
    return scores;
  }

  /** 使用中と候補の局末の値、半荘ごとの差の和、半荘ごとの席・局末の数。和と数は、局末を持つ半荘の分だけを持つ。 */
  private record Synthetic(HandEnds ends, HandEndValues used, HandEndValues candidate, double[] sums, long[] counts) {}

  private static Synthetic synthetic(HandEnds ends, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    int seats = ends.seats();
    HandEndValues.Builder used = HandEndValues.builder(ends);
    HandEndValues.Builder candidate = HandEndValues.builder(ends);
    double[] ranks = new double[seats * seats];
    Arrays.fill(ranks, 1.0 / seats);
    double[] usedPhi = new double[seats];
    double[] candidatePhi = new double[seats];
    double[] sums = new double[ends.games()];
    long[] counts = new long[ends.games()];
    int games = 0;
    for (int game = 0; game < ends.games(); game++) {
      int first = ends.firstHand(game);
      int last = ends.firstHand(game + 1);
      if (first == last) {
        continue;
      }
      double error = Math.sqrt(Math.max(0, 1 + DIFFERENCE + SPREAD * random.nextGaussian()));
      for (int h = first; h < last; h++) {
        for (int seat = 0; seat < seats; seat++) {
          double u = ends.utility(h, seat);
          usedPhi[seat] = u + 1;
          candidatePhi[seat] = u + error;
          double ofCandidate = u - candidatePhi[seat];
          double ofUsed = u - usedPhi[seat];
          sums[games] += ofCandidate * ofCandidate - ofUsed * ofUsed;
        }
        counts[games] += seats;
        used.set(h, usedPhi, ranks);
        candidate.set(h, candidatePhi, ranks);
      }
      games++;
    }
    return new Synthetic(
        ends, used.build(), candidate.build(), Arrays.copyOf(sums, games), Arrays.copyOf(counts, games));
  }

  private static Synthetic standard() {
    return synthetic(HandEnds.of(THREE, UTILITY, games(GAMES, EMPTY, 3)), 5);
  }

  /**
   * 差は、半荘ごとの和をまとめた比 Σ S_g / Σ n_g になる。ブートストラップの標準偏差は、半荘をクラスタとしたデルタ法の値
   * √Σ (S_g − m·n_g)² / Σ n_g と、ブートストラップの揺れ（相対で 1/√(2B)）の範囲で合う。要る半荘の数は G·(z·SD/d)² で、
   * 作るときの散らばりから求めた値とも SE の範囲で合う。
   */
  @Test
  public void gamesForTheLineMatchTheFormulaWithinTheirStandardError() {
    Synthetic synthetic = standard();
    TrustGames trust = TrustGames.measure(synthetic.ends(), synthetic.used(), synthetic.candidate(), RESAMPLES, 11);
    double[] sums = synthetic.sums();
    long[] counts = synthetic.counts();
    int games = sums.length;
    Assert.assertEquals(games, GAMES);
    Assert.assertEquals(trust.games(), GAMES, "games without hand ends are not counted");
    Assert.assertEquals(trust.resamples(), RESAMPLES);
    Assert.assertEquals(trust.handEnds(), synthetic.ends().handEnds());
    Assert.assertEquals(trust.handEndsPerGame(), (double) synthetic.ends().handEnds() / GAMES, 1e-12);

    double total = 0;
    double weight = 0;
    double squaredCounts = 0;
    for (int game = 0; game < games; game++) {
      total += sums[game];
      weight += counts[game];
      squaredCounts += (double) counts[game] * counts[game];
    }
    double mean = total / weight;
    Assert.assertEquals(trust.difference().mean(), mean, 1e-12);
    Assert.assertEquals(trust.difference().mean(), DIFFERENCE, 4 * SPREAD * Math.sqrt(squaredCounts) / weight);

    double squares = 0;
    for (int game = 0; game < games; game++) {
      double residual = sums[game] - mean * counts[game];
      squares += residual * residual;
    }
    double delta = Math.sqrt(squares) / weight;
    double bootstrapSe = 1 / Math.sqrt(2.0 * RESAMPLES);
    Assert.assertEquals(trust.bootstrapSd() / delta, 1, 4 * bootstrapSe, "bootstrap SD against the delta method");
    Assert.assertEquals(
        trust.bootstrapSd() / trust.difference().standardError(), 1, 4 * bootstrapSe,
        "bootstrap SD against the clustered SE of the gate");

    // 式どおり
    double line = GAMES * Math.pow(2 * trust.bootstrapSd() / trust.difference().mean(), 2);
    double power = GAMES * Math.pow((2 + 1.6448536269514722) * trust.bootstrapSd() / trust.difference().mean(), 2);
    Assert.assertEquals(trust.atLine(), line, line * 1e-12);
    Assert.assertEquals(trust.withPower(), power, power * 1e-12);
    Assert.assertTrue(trust.withPower() > trust.atLine());

    // 作るときの散らばりから求めた値: SD = τ·√Σ n_g² / Σ n_g、差 = d
    double designedSd = SPREAD * Math.sqrt(squaredCounts) / weight;
    double designedLine = GAMES * Math.pow(2 * designedSd / DIFFERENCE, 2);
    double relativeSe =
        2 * Math.sqrt(bootstrapSe * bootstrapSe + 1.0 / (2 * GAMES) + Math.pow(designedSd / DIFFERENCE, 2));
    Assert.assertEquals(
        trust.atLine() / designedLine, 1, 4 * relativeSe,
        "games at the line " + trust.atLine() + " against the designed " + designedLine);
  }

  /** 乱数の種が同じなら同じ値になり、違えばブートストラップの揺れの範囲で違う値になる。差は種に依らない。 */
  @Test
  public void theSameSeedGivesTheSameValue() {
    Synthetic synthetic = standard();
    TrustGames first = TrustGames.measure(synthetic.ends(), synthetic.used(), synthetic.candidate(), 500, 23);
    TrustGames again = TrustGames.measure(synthetic.ends(), synthetic.used(), synthetic.candidate(), 500, 23);
    Assert.assertEquals(again, first);
    Assert.assertEquals(Double.doubleToRawLongBits(again.atLine()), Double.doubleToRawLongBits(first.atLine()));
    Synthetic rebuilt = standard();
    TrustGames rebuiltAgain = TrustGames.measure(rebuilt.ends(), rebuilt.used(), rebuilt.candidate(), 500, 23);
    Assert.assertEquals(rebuiltAgain, first, "the same games and values give the same measurement");
    TrustGames other = TrustGames.measure(synthetic.ends(), synthetic.used(), synthetic.candidate(), 500, 24);
    Assert.assertNotEquals(other.bootstrapSd(), first.bootstrapSd());
    Assert.assertEquals(other.bootstrapSd() / first.bootstrapSd(), 1, 8 / Math.sqrt(2.0 * 500));
    Assert.assertEquals(other.difference(), first.difference(), "the difference does not depend on the seed");
  }

  /** 候補が良くなっていなければ（差が 0 以上なら）、何半荘で比べても関門を通らず、要る半荘の数は +∞ になる。 */
  @Test
  public void aCandidateThatIsNotBetterNeedsInfinitelyManyGames() {
    Synthetic synthetic = standard();
    TrustGames worse = TrustGames.measure(synthetic.ends(), synthetic.candidate(), synthetic.used(), 200, 1);
    Assert.assertTrue(worse.difference().mean() > 0);
    Assert.assertEquals(worse.atLine(), Double.POSITIVE_INFINITY);
    Assert.assertEquals(worse.withPower(), Double.POSITIVE_INFINITY);
    TrustGames same = TrustGames.measure(synthetic.ends(), synthetic.used(), synthetic.used(), 200, 1);
    Assert.assertEquals(same.difference().mean(), 0.0);
    Assert.assertEquals(same.bootstrapSd(), 0.0);
    Assert.assertEquals(same.atLine(), Double.POSITIVE_INFINITY);
  }

  @Test
  public void badInputsAreRefusedInEnglish() {
    Synthetic synthetic = standard();
    IllegalArgumentException resamples =
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> TrustGames.measure(synthetic.ends(), synthetic.used(), synthetic.candidate(), 1, 1));
    Assert.assertEquals(resamples.getMessage(), "The bootstrap needs at least two resamples: 1");
    Synthetic other = synthetic(HandEnds.of(THREE, UTILITY, games(GAMES, EMPTY, 3)), 5);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> TrustGames.measure(synthetic.ends(), synthetic.used(), other.candidate(), 10, 1));
    // 局末を持つ半荘が 1 つだけ
    List<GameRecord> lone = new ArrayList<>(games(1, 1, 7));
    Synthetic single = synthetic(HandEnds.of(THREE, UTILITY, lone), 1);
    IllegalArgumentException lonely =
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> TrustGames.measure(single.ends(), single.used(), single.candidate(), 10, 1));
    Assert.assertEquals(lonely.getMessage(), "The bootstrap needs at least two games with hand ends, got 1");
  }
}

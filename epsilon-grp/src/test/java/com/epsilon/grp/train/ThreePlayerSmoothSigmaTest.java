package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.Phi;
import com.epsilon.grp.ScoreSmooth;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalDouble;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 点数からなめらかに順位を求めるときの σ の係数を、半荘の記録から測る（{@link SmoothSigma}）。局末の点は、最終の点から
 * σ·√(残りの局の数)·z だけずらして作る（z は標準正規分布に従う）。そのため、測った係数は作るときに使った σ に戻る。三人打ちの係数は
 * 持たないので、三人打ちでなめらかな Φ を作ろうとすると例外を投げる（メッセージは英語）。
 */
public class ThreePlayerSmoothSigmaTest {

  private static final Variant THREE = Variant.TENHOU_3P;

  /** 半荘を作るときに使う係数（点）。 */
  private static final double SIGMA = 8_000;

  /** 最終の点の組。半荘ごとに席を 1 つずつずらして配る。 */
  private static final int[] FINALS = {50_000, 35_000, 20_000};

  /**
   * 局末を 1〜8 個持つ半荘を {@code count} 個作る。局末ごとに、次局を東 1 局から最後の局（三人打ちなら西 3 局）まで乱数で選び、
   * 局末の点を最終の点から σ·√(残りの局の数)·z だけずらす。
   */
  static List<GameRecord> moves(Variant variant, double sigma, int count, long firstId, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    int seats = variant.seats();
    List<GameRecord> games = new ArrayList<>(count);
    for (int game = 0; game < count; game++) {
      int[] finals = new int[seats];
      for (int seat = 0; seat < seats; seat++) {
        finals[seat] = seats == FINALS.length ? FINALS[(seat + game) % seats] : 25_000 + 10_000 * (seat - 1);
      }
      int hands = 1 + random.nextInt(8);
      List<GrpState> handEnds = new ArrayList<>(hands);
      for (int hand = 0; hand < hands; hand++) {
        int round = random.nextInt(variant.finalRound() + 1);
        double scale = sigma * Math.sqrt(ScoreSmooth.remainingRounds(variant, round));
        int[] scores = new int[seats];
        for (int seat = 0; seat < seats; seat++) {
          scores[seat] = finals[seat] - (int) Math.round(scale * random.nextGaussian());
        }
        handEnds.add(new GrpState(variant, round, 0, 0, scores));
      }
      games.add(new GameRecord(variant, firstId + game, handEnds, finals));
    }
    return games;
  }

  @Test
  public void theSigmaUsedToMakeTheMovesComesBack() {
    List<GameRecord> games = moves(THREE, SIGMA, 6_000, 0, 31);
    SmoothSigma sigma = SmoothSigma.builder(THREE).add(games).build();
    Assert.assertEquals(sigma.variant(), THREE);
    // 全ての席と局末を数える。係数は (最終の点 − 局末の点)/√(残りの局の数) の不偏標準偏差
    long seatHands = 0;
    double sum = 0;
    for (GameRecord game : games) {
      for (GrpState state : game.handEnds()) {
        for (int seat = 0; seat < 3; seat++) {
          sum += (game.finalScore(seat) - state.score(seat)) / Math.sqrt(ScoreSmooth.remainingRounds(THREE, state.round()));
        }
        seatHands += 3;
      }
    }
    double mean = sum / seatHands;
    double squares = 0;
    for (GameRecord game : games) {
      for (GrpState state : game.handEnds()) {
        for (int seat = 0; seat < 3; seat++) {
          double value =
              (game.finalScore(seat) - state.score(seat)) / Math.sqrt(ScoreSmooth.remainingRounds(THREE, state.round()));
          squares += (value - mean) * (value - mean);
        }
      }
    }
    Assert.assertEquals(sigma.seatHands(), seatHands);
    Assert.assertEquals(sigma.perHand(), Math.sqrt(squares / (seatHands - 1)), 1e-6);
    Assert.assertEquals(sigma.perHand(), SIGMA, 4 * SIGMA / Math.sqrt(2.0 * seatHands), "sigma per hand");

    // 次局ごと: 東 1 局から西 3 局まで全てそろう。残りの局の数は南 3 局までを数え、西入なら 1
    int[] remaining = {6, 5, 4, 3, 2, 1, 1, 1, 1};
    Assert.assertEquals(sigma.rounds().size(), THREE.finalRound() + 1);
    long counted = 0;
    for (SmoothSigma.Round round : sigma.rounds()) {
      Assert.assertEquals(round.remaining(), remaining[round.round()], "round " + round.round());
      double se = 4 / Math.sqrt(2.0 * round.seatHands());
      Assert.assertEquals(round.spread() / (SIGMA * Math.sqrt(round.remaining())), 1, se, "spread of " + round);
      Assert.assertEquals(round.perHand(), round.spread() / Math.sqrt(round.remaining()), 1e-9);
      Assert.assertEquals(round.perHand() / SIGMA, 1, se, "per hand of " + round);
      counted += round.seatHands();
    }
    Assert.assertEquals(counted, seatHands);
  }

  /** コホートごとに足しても、まとめて足しても同じ値になる。ルールの違う半荘が混ざった一覧は例外で断り、その一覧からは 1 件も足さない。 */
  @Test
  public void cohortsAddUpAndAnotherVariantIsRefused() {
    List<GameRecord> first = moves(THREE, SIGMA, 300, 0, 41);
    List<GameRecord> second = moves(THREE, SIGMA, 300, 300, 43);
    List<GameRecord> all = new ArrayList<>(first);
    all.addAll(second);
    SmoothSigma together = SmoothSigma.builder(THREE).add(all).build();
    SmoothSigma.Builder builder = SmoothSigma.builder(THREE).add(first);
    List<GameRecord> mixed = new ArrayList<>(moves(Variant.TENHOU_4P, SIGMA, 2, 0, 1));
    mixed.addAll(second);
    IllegalArgumentException refused = Assert.expectThrows(IllegalArgumentException.class, () -> builder.add(mixed));
    Assert.assertEquals(refused.getMessage(), "A TENHOU_4P game among TENHOU_3P games");
    SmoothSigma inParts = builder.add(second).build();
    Assert.assertEquals(inParts, together);
    SmoothSigma none = SmoothSigma.builder(THREE).build();
    Assert.assertEquals(none.seatHands(), 0);
    Assert.assertTrue(Double.isNaN(none.perHand()));
    Assert.assertTrue(none.rounds().isEmpty());
  }

  /** 三人打ちの係数は持たない。四人打ちの係数は測った値の 5140 点。 */
  @Test
  public void onlyFourPlayerHasAMeasuredSigma() {
    Assert.assertEquals(ScoreSmooth.sigmaPerHand(THREE), OptionalDouble.empty());
    Assert.assertEquals(ScoreSmooth.sigmaPerHand(Variant.TENHOU_4P), OptionalDouble.of(5140));
  }

  /** 三人打ちでなめらかな Φ を作ろうとすると、例外を投げる。メッセージは英語。 */
  @Test
  public void threePlayerScoreSmoothIsRefusedInEnglish() {
    IllegalArgumentException refused =
        Assert.expectThrows(IllegalArgumentException.class, () -> Phi.scoreSmooth(THREE, RankUtility.TENHOU));
    Assert.assertEquals(refused.getMessage(), "The score-smooth sigma is not measured for TENHOU_3P");
  }
}

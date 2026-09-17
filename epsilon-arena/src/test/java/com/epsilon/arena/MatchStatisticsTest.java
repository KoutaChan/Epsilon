package com.epsilon.arena;

import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 同じ牌山の席替え結果をまとめた信頼区間と、同点時の順位・固定シードの扱いを検証する。 */
public class MatchStatisticsTest {
  @Test
  public void confidenceUsesThreeWallsRatherThanTwelveCorrelatedGames() {
    MatchStatistics statistics = new MatchStatistics();
    int[][] participantScores = {
      {40000, 20000, 20000, 20000}, {10000, 30000, 30000, 30000}, {30000, 40000, 20000, 10000}
    };
    for (int game : new int[] {11, 3, 4, 0, 7, 8, 2, 10, 1, 6, 9, 5}) {
      statistics.add(game, rotate(participantScores[game / 4], game % 4));
    }
    var paired = statistics.paired();
    double rankError = 4 * Math.sqrt(7) / 9;
    Assert.assertEquals(paired.walls(), 3);
    Assert.assertEquals(paired.rankAdvantage(), 2.0 / 9, 1e-12);
    Assert.assertEquals(paired.rankStandardError(), rankError, 1e-12);
    Assert.assertEquals(paired.rankLower95(), 2.0 / 9 - 1.96 * rankError, 1e-12);
    Assert.assertEquals(paired.scoreAdvantage(), 20000.0 / 9, 1e-9);
    Assert.assertEquals(paired.scoreStandardError(), rankError * 10000, 1e-9);
    Assert.assertEquals(paired.scoreLower95(), 20000.0 / 9 - 1.96 * rankError * 10000, 1e-9);
    var first = statistics.players(List.of("candidate", "a", "b", "c")).getFirst();
    Assert.assertEquals(first.averageRank(), 7.0 / 3, 1e-12);
    Assert.assertEquals(first.first(), 4);
    Assert.assertEquals(first.second(), 4);
    Assert.assertEquals(first.fourth(), 4);
  }

  @Test
  public void equalScoresRespectSeatOrderAndCancelAcrossAllRotations() {
    MatchStatistics statistics = new MatchStatistics();
    for (int game = 0; game < 4; game++)
      statistics.add(game, new int[] {25000, 25000, 25000, 25000});
    var paired = statistics.paired();
    Assert.assertEquals(paired.rankAdvantage(), 0.0, 1e-12);
    Assert.assertEquals(paired.scoreAdvantage(), 0.0);
    Assert.assertEquals(paired.rankStandardError(), 0.0);
    for (var player : statistics.players(List.of("a", "b", "c", "d"))) {
      Assert.assertEquals(player.averageRank(), 2.5);
      Assert.assertEquals(player.first(), 1);
      Assert.assertEquals(player.fourth(), 1);
    }
  }

  @Test
  public void incompleteWallCannotProduceAConfidenceInterval() {
    MatchStatistics statistics = new MatchStatistics();
    statistics.add(0, new int[] {40000, 20000, 20000, 20000});
    Assert.expectThrows(IllegalStateException.class, statistics::paired);
  }

  @Test
  public void fixedWallSeedsAndWarmupOffsetMatchTheOriginalEvalVsSequence() {
    RunSettings first = new RunSettings(2304, 128, 4, 98200000L, 0, 0);
    Assert.assertEquals(first.wallSeed(0), -5662750408236166856L);
    Assert.assertEquals(first.wallSeed(3), -5662750408236166856L);
    Assert.assertEquals(first.wallSeed(4), 6677185565226558319L);
    RunSettings measured = new RunSettings(2048, 128, 4, 98200000L, 0, 64);
    Assert.assertEquals(measured.wallSeed(0), 985275713442325221L);
    Assert.assertEquals(measured.wallSeed(2047), -8523246863023800583L);
    Assert.assertEquals(measured.wallSeed(0), first.wallSeed(256));
  }

  private static int[] rotate(int[] participantScores, int rotation) {
    int[] scores = new int[4];
    for (int participant = 0; participant < 4; participant++)
      scores[(participant + rotation) % 4] = participantScores[participant];
    return scores;
  }
}

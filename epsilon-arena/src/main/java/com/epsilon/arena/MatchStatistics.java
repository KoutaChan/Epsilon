package com.epsilon.arena;

import com.epsilon.core.ScoreRanking;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** 完了した対局を集計し、同じ牌山で4通りの席替えを終えた対局群を、信頼区間の独立標本とする。 */
final class MatchStatistics {
  private final long[] rankTotals = new long[4];
  private final long[] scoreTotals = new long[4];
  private final int[][] placements = new int[4][4];
  private final int[] ranks = new int[4];
  private final Map<Integer, Wall> incomplete = new HashMap<>();
  private int games;
  private int walls;
  private double rankMean, rankM2, scoreMean, scoreM2;

  void add(int gameIndex, int[] scores) {
    int rotation = gameIndex % 4;
    ScoreRanking.byScoreThenSeat(scores, ranks);
    double opponentRanks = 0, opponentScores = 0;
    for (int seat = 0; seat < 4; seat++) {
      int participant = (seat - rotation + 4) % 4;
      int rank = ranks[seat] + 1;
      rankTotals[participant] += rank;
      scoreTotals[participant] += scores[seat];
      placements[participant][rank - 1]++;
      if (participant != 0) {
        opponentRanks += rank;
        opponentScores += scores[seat];
      }
    }
    Wall wall = incomplete.computeIfAbsent(gameIndex / 4, ignored -> new Wall());
    wall.rank += (opponentRanks / 3 - (ranks[rotation] + 1)) / 4;
    wall.score += (scores[rotation] - opponentScores / 3) / 4;
    games++;
    if (++wall.rotations == 4) {
      incomplete.remove(gameIndex / 4);
      walls++;
      double rankDelta = wall.rank - rankMean;
      rankMean += rankDelta / walls;
      rankM2 += rankDelta * (wall.rank - rankMean);
      double scoreDelta = wall.score - scoreMean;
      scoreMean += scoreDelta / walls;
      scoreM2 += scoreDelta * (wall.score - scoreMean);
    }
  }

  ArenaResult.PairedResult paired() {
    if (!incomplete.isEmpty() || games == 0)
      throw new IllegalStateException("Paired statistics require complete four-seat walls");
    double rankError = walls > 1 ? Math.sqrt(rankM2 / (walls - 1) / walls) : 0;
    double scoreError = walls > 1 ? Math.sqrt(scoreM2 / (walls - 1) / walls) : 0;
    return new ArenaResult.PairedResult(
        walls,
        rankMean,
        rankError,
        rankMean - 1.96 * rankError,
        scoreMean,
        scoreError,
        scoreMean - 1.96 * scoreError);
  }

  List<ArenaResult.PlayerResult> players(List<String> names) {
    List<ArenaResult.PlayerResult> players = new ArrayList<>(4);
    for (int player = 0; player < 4; player++)
      players.add(
          new ArenaResult.PlayerResult(
              names.get(player),
              (double) rankTotals[player] / games,
              (double) scoreTotals[player] / games,
              placements[player][0],
              placements[player][1],
              placements[player][2],
              placements[player][3]));
    return players;
  }

  private static final class Wall {
    int rotations;
    double rank, score;
  }
}
